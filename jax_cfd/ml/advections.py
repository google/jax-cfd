"""Models for advection and convection components."""
import functools

from typing import Callable, Optional
import gin
from jax_cfd.base import advection
from jax_cfd.base import grids
from jax_cfd.ml import interpolations
from jax_cfd.ml import physics_specifications


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationModule = interpolations.InterpolationModule
AdvectFn = Callable[[GridVariable, GridVariableVector, float], GridArray]
AdvectionModule = Callable[..., AdvectFn]
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
ConvectionModule = Callable[..., ConvectFn]


@gin.register
def modular_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    c_interpolation_module: InterpolationModule = interpolations.upwind,
    u_interpolation_module: InterpolationModule = interpolations.linear,
    **kwargs
) -> AdvectFn:
  """Modular advection module based on `advection_diffusion.advect_general`."""
  c_interpolate_fn = c_interpolation_module(grid, dt, physics_specs, **kwargs)
  u_interpolate_fn = u_interpolation_module(grid, dt, physics_specs, **kwargs)

  def advect(
      c: GridVariable,
      v: GridVariableVector,
      dt: Optional[float] = None
  ) -> GridArray:
    return advection.advect_general(
        c, v, u_interpolate_fn, c_interpolate_fn, dt)

  return advect


@gin.register
def modular_self_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    interpolation_module: InterpolationModule,
    **kwargs
) -> AdvectFn:
  """Modular self advection using a single interpolation module."""
  # TODO(jamieas): Replace this entire function once
  # `single_tower_navier_stokes` is in place.
  interpolate_fn = interpolation_module(grid, dt, physics_specs, **kwargs)
  c_interpolate_fn = functools.partial(interpolate_fn, tag='c')
  u_interpolate_fn = functools.partial(interpolate_fn, tag='u')

  def advect(
      c: GridVariable,
      v: GridVariableVector,
      dt: Optional[float] = None
  ) -> GridArray:
    return advection.advect_general(
        c, v, u_interpolate_fn, c_interpolate_fn, dt)

  return advect


@gin.register
def self_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    advection_module: AdvectionModule = modular_advection,
    **kwargs
) -> ConvectFn:
  """Convection module based on simultaneous self-advection of velocities."""
  advect_fn = advection_module(grid, dt, physics_specs, **kwargs)

  def convect(v: GridVariableVector) -> GridArrayVector:
    return tuple(advect_fn(u, v, dt) for u in v)

  return convect
