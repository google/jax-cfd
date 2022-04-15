"""Models for closure terms and effective viscosities."""

import functools
from typing import Any, Callable, Optional, Tuple, Union

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import subgrid_models
from jax_cfd.ml import interpolations
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers
import numpy as np

Array = Union[np.ndarray, jnp.DeviceArray]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
InterpolationModule = interpolations.InterpolationModule
ViscosityFn = Callable[[grids.GridArrayTensor, GridArrayVector, grids.Grid],
                       grids.GridArrayTensor]
ViscosityModule = Callable[..., ViscosityFn]


@gin.register
def smagorinsky_viscosity(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    viscosity_scale: Optional[float] = None,
    cs: float = 0.2,
    interpolation_module: InterpolationModule = interpolations.linear,
) -> ViscosityFn:
  """Constructs a Smagorinsky viscosity model."""
  del viscosity_scale  # unused.
  interpolate = interpolation_module(grid, dt, physics_specs)
  viscosity_fn = functools.partial(
      subgrid_models.smagorinsky_viscosity, dt=dt, cs=cs,
      interpolate_fn=interpolate)
  return hk.to_module(viscosity_fn)(name='smagorinsky_viscosity')


@gin.register
def learned_scalar_viscosity(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    viscosity_scale: float,
    interpolate_module: InterpolationModule = interpolations.linear,
    tower_factory: Callable[..., Any] = towers.forward_tower_factory,
) -> ViscosityFn:
  """Constructs an learned, scalar-valued viscosity model."""
  def interpolate(
      c: GridArray,
      offset: Tuple[float, ...],
      v: Optional[GridVariableVector] = None,
      dt: Optional[float] = None,
  ) -> grids.GridArray:
    """Interpolation method wrapped for GridArray using periodic BC."""
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    c_bc = grids.GridVariable(c, bc)
    interp_var = interpolate_module(grid, dt, physics_specs)(
        c_bc, offset, v, dt)
    return interp_var.array

  def viscosity_fn(
      s_ij: grids.GridArrayTensor,
      v: GridArrayVector,
  ) -> grids.GridArrayTensor:
    """Computes effective eddy viscosity using learned components.

    This viscosity model computes parametric scalar viscosity that is
    interpolated to the offsets of the strain rate tensor.

    Args:
      s_ij: strain rate tensor that is equal to the forward finite difference
        derivatives of the velocity field `(d(u_i)/d(x_j) + d(u_j)/d(x_i)) / 2`.
      v: velocity field.

    Returns:
      tensor containing values of the eddy viscosity at the same grid offsets
      as the strain tensor `s_ij`.
    """
    s_ij_offsets = [array.offset for array in s_ij.ravel()]
    unique_offsets = list(set(s_ij_offsets))
    viscosity_net = tower_factory(1, grid.ndim)
    inputs = jnp.stack([u.data for u in v], axis=-1)
    predicted_viscosity = (viscosity_scale + 1e-6) * viscosity_net(inputs)
    predicted_viscosity = grids.GridArray(
        data=jnp.squeeze(predicted_viscosity, -1),
        offset=grid.cell_center, grid=grid)
    interpolated_viscosities = {
        offset: interpolate(predicted_viscosity, offset, v, dt)
        for offset in unique_offsets}
    viscosities = [interpolated_viscosities[offset] for offset in s_ij_offsets]
    tree_def = jax.tree_util.tree_structure(s_ij)
    return jax.tree_unflatten(tree_def, [x.data for x in viscosities])

  return hk.to_module(viscosity_fn)()


@gin.register
def learned_scalar_viscosity_from_gradients(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    viscosity_scale: float,
    interpolate_module: InterpolationModule = interpolations.linear,
    tower_factory: Callable[..., Any] = towers.forward_tower_factory,
) -> ViscosityFn:
  """Constructs a scalar viscosity model predicted from velocity gradients."""
  def interpolate(
      c: GridArray,
      offset: Tuple[float, ...],
      v: Optional[GridVariableVector] = None,
      dt: Optional[float] = None,
  ) -> grids.GridArray:
    """Interpolation method wrapped for GridArray using periodic BC."""
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    c_bc = grids.GridVariable(c, bc)
    interp_var = interpolate_module(grid, dt, physics_specs)(
        c_bc, offset, v, dt)
    return interp_var.array

  def viscosity_fn(
      s_ij: grids.GridArrayTensor,
      v: GridArrayVector,
  ) -> grids.GridArrayTensor:
    """Computes effective eddy viscosity using learned components.

    This viscosity model computes parametric scalar viscosity that is
    interpolated to the offsets of the strain rate tensor.

    Args:
      s_ij: strain rate tensor that is equal to the forward finite difference
        derivatives of the velocity field `(d(u_i)/d(x_j) + d(u_j)/d(x_i)) / 2`.
      v: velocity field.

    Returns:
      tensor containing values of the eddy viscosity at the same grid offsets
      as the strain tensor `s_ij`.
    """
    s_ij_offsets = [array.offset for array in s_ij.ravel()]
    unique_offsets = list(set(s_ij_offsets))
    viscosity_net = tower_factory(1, grid.ndim)
    cell_center = grid.cell_center
    interpolate_to_center = lambda x: interpolate(x, cell_center, v, dt)
    centered_s_ij = np.vectorize(interpolate_to_center)(s_ij)
    inputs = jnp.stack([array.data for array in centered_s_ij.ravel()], axis=-1)
    predicted_viscosity = (viscosity_scale + 1e-6) * viscosity_net(inputs)
    predicted_viscosity = grids.GridArray(
        data=jnp.squeeze(predicted_viscosity, -1),
        offset=grid.cell_center, grid=grid)
    interpolated_viscosities = {
        offset: interpolate(predicted_viscosity, offset, v, dt)
        for offset in unique_offsets}
    viscosities = [interpolated_viscosities[offset] for offset in s_ij_offsets]
    tree_def = jax.tree_util.tree_structure(s_ij)
    return jax.tree_unflatten(tree_def, [x.data for x in viscosities])

  return hk.to_module(viscosity_fn)()


@gin.register
def learned_tensor_viscosity(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    viscosity_scale: float,
    tower_factory: Callable[..., Any] = towers.forward_tower_factory,
) -> ViscosityFn:
  """Constructs an learned, tensor-valued viscosity model."""
  del dt, physics_specs

  def viscosity_fn(
      s_ij: grids.GridArrayTensor,
      v: GridArrayVector,
  ) -> grids.GridArrayTensor:
    """Computes effective eddy viscosity using learned components.

    This viscosity model computes parametric tensor viscosity that predicts
    independent values at all offsets of the strain rate tensor.

    Args:
      s_ij: strain rate tensor that is equal to the forward finite difference
        derivatives of the velocity field `(d(u_i)/d(x_j) + d(u_j)/d(x_i)) / 2`.
      v: velocity field.

    Returns:
      tensor containing values of the eddy viscosity at the same grid offsets
      as the strain tensor `s_ij`.
    """
    s_ij_offsets = [array.offset for array in s_ij.ravel()]
    unique_offsets = list(set(s_ij_offsets))
    num_offsets = len(unique_offsets)
    viscosity_net = tower_factory(num_offsets, grid.ndim)
    inputs = jnp.stack([u.data for u in v], axis=-1)
    viscosities = (viscosity_scale + 1e-6) * viscosity_net(inputs)
    viscosities = jnp.split(viscosities, np.arange(1, num_offsets), axis=-1)
    viscosities_dict = {
        offset: jnp.squeeze(visc, axis=-1)  # remove channel dimension.
        for offset, visc in zip(unique_offsets, viscosities)}
    viscosities = [viscosities_dict[offset] for offset in s_ij_offsets]
    tree_def = jax.tree_util.tree_structure(s_ij)
    return jax.tree_unflatten(tree_def, viscosities)

  return hk.to_module(viscosity_fn)()


@gin.register
def eddy_viscosity_model(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    v: Optional[GridArrayVector] = None,
    viscosity_scale: Optional[float] = None,
    viscosity_model: ViscosityModule = smagorinsky_viscosity,
):
  """Constructs eddy viscosity model that computes accelerations.

  Eddy viscosity models compute a turbulence closure term as a divergence of the
  subgrid-scale stress tensor, which is expressed as velocity dependent
  viscosity times the rate of strain tensor. This module delegates computation
  of the eddy-viscosity to `viscosity_model` function. Note that if outputs of
  the `viscosity_model` are not unrestricted to interpolations of a scalar
  field, this model can represent almost arbitrary stress tensor.
  For details see go/whirl-evm.

  Args:
    grid: grid on which the Navier-Stokes equation is discretized.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation module.
    v: optional velocity field that is used to precompute values in models.
    viscosity_scale: the kinematic viscosity of the fluid.
    viscosity_model: function that generates a `viscosity_fn`.

  Returns:
    Function that computes accelerations due to eddy viscosity model.
  """
  del v  # unused.
  if viscosity_scale is None:
    viscosity_scale = physics_specs.viscosity
  viscosity = viscosity_model(
      grid, dt, physics_specs, viscosity_scale=viscosity_scale)
  evm_fn = functools.partial(subgrid_models.evm_model, viscosity_fn=viscosity)
  return hk.to_module(evm_fn)(name='eddy_viscosity_model')
