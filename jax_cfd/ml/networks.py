"""Network modules that interface with numerical methods."""

from typing import Callable, Optional, Tuple
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import finite_differences
from jax_cfd.base import grids
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers


def _identity(grid, dt, physics_specs):
  del grid, dt, physics_specs  # unused.
  return lambda x: x


@gin.configurable
def split_to_aligned_field(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
):
  """Returns module that splits inputs along last axis into AlignedField."""
  del dt, physics_specs  # unused.
  data_offsets = data_offsets or grid.cell_faces

  def process(inputs):
    split_inputs = array_utils.split_axis(inputs, -1)
    return tuple(grids.AlignedArray(x, offset)
                 for x, offset in zip(split_inputs, data_offsets))

  return hk.to_module(process)()


@gin.configurable
def aligned_field_from_split_divergence(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
):
  """Returns module that splits inputs along last axis into AlignedField."""
  del dt, physics_specs  # unused.

  def _shift_offset(offset, axis):
    return tuple(o + 0.5 if i == axis else o for i, o in enumerate(offset))

  flux_offsets = tuple(
      _shift_offset(o, i) for i in range(grid.ndim)  # pylint: disable=g-complex-comprehension
      for o in grid.cell_faces
  )

  def process(inputs):
    split_inputs = array_utils.split_axis(inputs, -1)
    split_inputs = tuple(grids.AlignedArray(x, o)
                         for x, o in zip(split_inputs, flux_offsets))
    # below we combine `grid.ndim`-sized sequences of arrays into a tuples.
    # we do that by iterating over a `grid.ndim`-sized zip of the same iterator.
    # For example:
    # a = [1, 2, 3, 4]
    # tuple(zip(*([iter(a)] * 2))) >>> ((1, 2), (3, 4))
    split_inputs = tuple(zip(*[iter(split_inputs)] * grid.ndim))
    tensor_inputs = grids.Tensor(split_inputs)
    return tuple(-finite_differences.divergence(tensor_inputs[i, :], grid)
                 for i in range(grid.ndim))

  return hk.to_module(process)()


@gin.configurable
def stack_aligned_field(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
):
  """Returns a module that stacks AlignedField along the last axis."""
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


@gin.configurable
def flux_corrector_network(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: towers.TowerFactory,
    name: Optional[str] = None,
):
  """Returns a module that computes corrections to the velocity fluxes."""
  pre_process_module = stack_aligned_field
  post_process_module = aligned_field_from_split_divergence
  num_output_channels = grid.ndim ** 2
  return tower_module(
      grid=grid, dt=dt, physics_specs=physics_specs,
      tower_factory=tower_factory, pre_process_module=pre_process_module,
      post_process_module=post_process_module,
      num_output_channels=num_output_channels, name=name)
