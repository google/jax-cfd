"""Network modules that interface with numerical methods."""

import functools
import itertools
from typing import Callable, Optional, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers
import numpy as np


def _identity(grid, dt, physics_specs):
  del grid, dt, physics_specs  # unused.
  return lambda x: x


@gin.register
def split_to_aligned_field(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    network_offsets: Optional[Tuple[Tuple[float, float], ...]] = None,
):
  """Returns module that splits inputs along last axis into GridArrayVector."""
  del dt  # unused.
  if hasattr(physics_specs, "combo_offsets"):
    data_offsets = physics_specs.combo_offsets()
  else:
    data_offsets = grid.cell_faces

  if hasattr(physics_specs, "combo_boundaries"):
    boundary_conditions = physics_specs.combo_boundaries()
  else:
    boundary_conditions = tuple(
        boundaries.periodic_boundary_conditions(grid.ndim)
        for _ in range(grid.ndim))
  network_offsets = network_offsets or data_offsets
  def process(inputs):
    split_inputs = array_utils.split_axis(inputs, -1)
    output = tuple(
        grids.GridVariable(grids.GridArray(x, offset, grid), bc) for x, offset,
        bc in zip(split_inputs, network_offsets, boundary_conditions))
    output = tuple(
        interpolation.linear(x, offset)
        for x, offset in zip(output, data_offsets))
    return output
  return hk.to_module(process)()


@gin.configurable()
def interpolate_gridvar(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    final_offsets: Optional[Tuple[Tuple[float, float], ...]] = None,
    process_fn: Optional[Callable] = lambda x: x,  # pylint: disable=g-bare-generic
):
  """Returns module that splits inputs along last axis into GridArrayVector."""
  del dt  # unused.
  if hasattr(physics_specs, "combo_offsets"):
    data_offsets = physics_specs.combo_offsets()
  else:
    data_offsets = grid.cell_faces
  final_offsets = final_offsets or data_offsets

  def process(inputs):
    inputs = process_fn(inputs)
    inputs = tuple(
        interpolation.linear(x, offset)
        for x, offset in zip(inputs, final_offsets))
    return inputs

  return hk.to_module(process)()


@gin.register
def aligned_field_from_split_divergence(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
):
  """Returns module that splits inputs along last axis into GridArrayVector."""
  del dt, physics_specs  # unused.

  def _shift_offset(offset, axis):
    return tuple(o + 0.5 if i == axis else o for i, o in enumerate(offset))

  flux_offsets = tuple(
      _shift_offset(o, i) for i in range(grid.ndim)  # pylint: disable=g-complex-comprehension
      for o in grid.cell_faces
  )

  def _to_grid_variables(grid_arrays):
    # TODO(dkochkov) make boundary conditions configurable.
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    return tuple(grids.GridVariable(array, bc) for array in grid_arrays)

  def process(inputs):
    split_inputs = array_utils.split_axis(inputs, -1)
    split_inputs = tuple(grids.GridArray(x, o, grid)
                         for x, o in zip(split_inputs, flux_offsets))
    # below we combine `grid.ndim`-sized sequences of arrays into a tuples.
    # we do that by iterating over a `grid.ndim`-sized zip of the same iterator.
    # For example:
    # a = [1, 2, 3, 4]
    # tuple(zip(*([iter(a)] * 2))) >>> ((1, 2), (3, 4))
    split_inputs = tuple(zip(*[iter(split_inputs)] * grid.ndim))
    tensor_inputs = grids.GridArrayTensor(split_inputs)
    # to compute divergence we need to convert fluxes to GridVariable class.
    grid_array_field = tuple(
        -finite_differences.divergence(_to_grid_variables(tensor_inputs[i, :]))
        for i in range(grid.ndim))
    # since divergence removes the boundary conditions, we add them back.
    return _to_grid_variables(grid_array_field)

  return hk.to_module(process)()


@gin.register
def stack_aligned_field_with_neighbors(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    n_neighbors: int = 1,
):
  """Returns a module that stacks input field with neighbors along channels."""
  del dt, physics_specs  # unused.
  shifts = [i for i in np.arange(-n_neighbors, n_neighbors + 1) if i != 0]
  shifts_and_axis = list(itertools.product(shifts, np.arange(grid.ndim)))
  shifts_and_axis.append([0, 0])

  def process(inputs):
    inputs = tuple(jnp.expand_dims(x.data, axis=-1) for x in inputs)
    array = array_utils.concat_along_axis(jax.tree_leaves(inputs), axis=-1)
    arrays = tuple(
        jnp.roll(array, *shift_and_axis) for shift_and_axis in shifts_and_axis)
    return array_utils.concat_along_axis(arrays, axis=-1)

  return hk.to_module(process)()


@gin.register
def stack_aligned_field(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
):
  """Returns a module that stacks GridArrayVector along the last axis."""
  del grid, dt, physics_specs  # unused.

  def process(inputs):
    inputs = tuple(jnp.expand_dims(x.data, axis=-1) for x in inputs)
    return array_utils.concat_along_axis(jax.tree_leaves(inputs), axis=-1)

  return hk.to_module(process)()


@gin.configurable
def tower_module(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: towers.TowerFactory,
    pre_process_module: Callable = _identity,  # pylint: disable=g-bare-generic
    post_process_module: Callable = _identity,  # pylint: disable=g-bare-generic
    num_output_channels: Optional[int] = None,
    name: Optional[str] = None,
):
  """Constructs tower module with configured number of output channels."""
  pre_process = pre_process_module(grid, dt, physics_specs)
  post_process = post_process_module(grid, dt, physics_specs)

  def forward_pass(x):
    x = pre_process(x)
    if num_output_channels is None:
      network = tower_factory(x.shape[-1], grid.ndim)
    else:
      network = tower_factory(num_output_channels, grid.ndim)
    return post_process(network(x))

  return hk.to_module(forward_pass)(name=name)


@gin.configurable
def velocity_corrector_network_w_boundaries(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: towers.TowerFactory,
    network_offsets: Tuple[Tuple[float, ...], ...],
    num_output_channels: int,
    name: Optional[str] = None,
    process_fn: Optional[Callable] = _identity,  # pylint: disable=g-bare-generic
):
  """Returns a module that computes corrections to the velocity field."""
  pre_process = functools.partial(
      interpolate_gridvar, final_offsets=network_offsets, process_fn=process_fn)
  post_process = interpolate_gridvar
  return tower_module(
      grid=grid, dt=dt, physics_specs=physics_specs,
      tower_factory=tower_factory, pre_process_module=pre_process,
      post_process_module=post_process, num_output_channels=num_output_channels,
      name=name)


@gin.register
def velocity_corrector_network(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: towers.TowerFactory,
    name: Optional[str] = None,
):
  """Returns a module that computes corrections to the velocity field."""
  pre_process_module = stack_aligned_field
  post_process_module = split_to_aligned_field
  return tower_module(
      grid=grid, dt=dt, physics_specs=physics_specs,
      tower_factory=tower_factory, pre_process_module=pre_process_module,
      post_process_module=post_process_module, num_output_channels=grid.ndim,
      name=name)


@gin.register
def flux_corrector_network(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: towers.TowerFactory,
    pre_process_module: Callable = stack_aligned_field,  # pylint: disable=g-bare-generic
    name: Optional[str] = None,
):
  """Returns a module that computes corrections to the velocity fluxes."""
  post_process_module = aligned_field_from_split_divergence
  num_output_channels = grid.ndim ** 2
  return tower_module(
      grid=grid, dt=dt, physics_specs=physics_specs,
      tower_factory=tower_factory, pre_process_module=pre_process_module,
      post_process_module=post_process_module,
      num_output_channels=num_output_channels, name=name)
