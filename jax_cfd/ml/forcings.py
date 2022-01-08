"""Components that apply forcing. See jax_cfd.base.forcings for forcing API."""

from typing import Callable
import gin

from jax_cfd.base import equations
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import forcings as spectral_forcings

ForcingFn = forcings.ForcingFn
ForcingModule = Callable[..., ForcingFn]


gin.external_configurable(spectral_forcings.kolmogorov_forcing_fn)
gin.external_configurable(spectral_forcings.spectral_no_forcing)


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
def kolmogorov_forcing(grid: grids.Grid,  # pylint: disable=missing-function-docstring
                       scale: float = 0,
                       wavenumber: int = 2,
                       linear_coefficient: float = 0,
                       swap_xy: bool = False) -> ForcingFn:
  force_fn = forcings.kolmogorov_forcing(grid, scale, wavenumber, swap_xy)
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
