"""Models for advection and convection components."""
import functools

from typing import Callable, Optional, Tuple
import gin
from jax_cfd.base import advection
from jax_cfd.base import grids
from jax_cfd.ml import interpolations
from jax_cfd.ml import physics_specifications


AlignedArray = grids.AlignedArray
Grid = grids.Grid
AlignedField = Tuple[AlignedArray, ...]
InterpolationModule = interpolations.InterpolationModule
AdvectFn = Callable[[AlignedArray, AlignedField, Grid, float], AlignedArray]
AdvectionModule = Callable[..., AdvectFn]
ConvectFn = Callable[[AlignedField, Grid], AlignedField]
ConvectionModule = Callable[..., ConvectFn]


@gin.configurable
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
      c: AlignedArray,
      v: AlignedField,
      grid: grids.Grid,
      dt: Optional[float] = None
  ) -> AlignedArray:
    return advection.advect_general(
        c, v, grid, u_interpolate_fn, c_interpolate_fn, dt)

  return advect


@gin.configurable
def modular_self_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    interpolation_module: InterpolationModule = (
        interpolations.FusedLearnedInterpolation),
    **kwargs
) -> AdvectFn:
  """Modular self advection using a single interpolation module."""
  # TODO(jamieas): Replace this entire function once
  # `single_tower_navier_stokes` is in place.
  interpolate_fn = interpolation_module(grid, dt, physics_specs, **kwargs)
  c_interpolate_fn = functools.partial(interpolate_fn, tag='c')
  u_interpolate_fn = functools.partial(interpolate_fn, tag='u')

  def advect(
      c: AlignedArray,
      v: AlignedField,
      grid: grids.Grid,
      dt: Optional[float] = None
  ) -> AlignedArray:
    return advection.advect_general(
        c, v, grid, u_interpolate_fn, c_interpolate_fn, dt)

  return advect


@gin.configurable
def self_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    advection_module: AdvectionModule = modular_advection,
    **kwargs
) -> ConvectFn:
  """Convection module based on simultaneous self-advection of velocities."""
  advect_fn = advection_module(grid, dt, physics_specs, **kwargs)

  def convect(v: AlignedField, grid: grids.Grid) -> AlignedField:
    return tuple(advect_fn(u, v, grid, dt) for u in v)

  return convect
