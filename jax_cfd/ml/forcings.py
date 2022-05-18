"""Components that apply forcing. See jax_cfd.base.forcings for forcing API."""

from typing import Callable

from typing import Optional, Tuple
import gin
from jax import numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import boundaries
from jax_cfd.base import equations
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import utils as spectral_utils

ForcingFn = forcings.ForcingFn
ForcingModule = Callable[..., ForcingFn]


def sum_forcings(*forces: ForcingFn) -> ForcingFn:
  """Sum multiple forcing functions."""
  def forcing(v):
    return equations.sum_fields(*[forcing(v) for forcing in forces])
  return forcing


@gin.register
def filtered_linear_forcing(grid: grids.Grid,
                            scale: float,
                            lower_wavenumber: float = 0,
                            upper_wavenumber: float = 4) -> ForcingFn:
  return forcings.filtered_linear_forcing(lower_wavenumber,
                                          upper_wavenumber,
                                          coefficient=scale,
                                          grid=grid)


@gin.register
def linear_forcing(grid: grids.Grid,
                   scale: float) -> ForcingFn:
  return forcings.linear_forcing(grid, scale)


@gin.register
def spectral_kolmogorov_forcing(grid):
  return forcings.kolmogorov_forcing(
      grid, 1.0, k=4, swap_xy=False, offsets=((0.0, 0.0), (0.0, 0.0)))


@gin.register
def vorticity_space_forcing(grid: grids.Grid, forcing_module: ForcingModule):
  forcing_fn = forcing_module(grid, offsets=((0.0, 0.0), (0.0, 0.0)))
  velocity_solve = spectral_utils.vorticity_to_velocity(grid)
  kx, ky = grid.rfft_mesh()
  fft, ifft = jnp.fft.rfft2, jnp.fft.irfft2
  bc = boundaries.periodic_boundary_conditions(grid.ndim)
  offset = (0.0, 0.0)  # TODO(dresdner) do not hard code

  def forcing_fn_ret(vorticity):
    vorticity, = array_utils.split_axis(vorticity, axis=-1)  # channel dim = 1
    v = tuple(
        grids.GridVariable(grids.GridArray(ifft(u), offset, grid), bc)
        for u in velocity_solve(fft(vorticity)))
    fhatu, fhatv = tuple(fft(u) for u in forcing_fn(v))
    fhat_vorticity = 2j * jnp.pi * (fhatv * kx - fhatu * ky)
    return ifft(fhat_vorticity)

  return forcing_fn_ret


@gin.register
def kolmogorov_forcing(grid: grids.Grid,  # pylint: disable=missing-function-docstring
                       scale: float = 0,
                       wavenumber: int = 2,
                       linear_coefficient: float = 0,
                       swap_xy: bool = False,
                       offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
                       ) -> ForcingFn:
  force_fn = forcings.kolmogorov_forcing(
      grid, scale, wavenumber, swap_xy, offsets=offsets)
  if linear_coefficient != 0:
    linear_force_fn = forcings.linear_forcing(grid, linear_coefficient)
    force_fn = forcings.sum_forcings(force_fn, linear_force_fn)
  return force_fn


@gin.register
def taylor_green_forcing(grid: grids.Grid,
                         scale: float = 0,
                         wavenumber: int = 2,
                         linear_coefficient: float = 0) -> ForcingFn:
  force_fn = forcings.taylor_green_forcing(grid, scale, wavenumber)
  if linear_coefficient != 0:
    linear_force_fn = forcings.linear_forcing(grid, linear_coefficient)
    force_fn = forcings.sum_forcings(force_fn, linear_force_fn)
  return force_fn


@gin.register
def no_forcing(grid: grids.Grid) -> ForcingFn:
  return forcings.no_forcing(grid)
