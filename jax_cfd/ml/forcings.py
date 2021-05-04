"""Components that apply forcing. See jax_cfd.base.forcings for forcing API."""

from typing import Callable, Tuple
import gin

from jax_cfd.base import equations
from jax_cfd.base import forcings
from jax_cfd.base import grids

AlignedArray = grids.AlignedArray
AlignedField = Tuple[AlignedArray, ...]
Grid = grids.Grid
ForcingFunction = forcings.ForcingFunction
ForcingModule = Callable[..., ForcingFunction]


def sum_forcings(*forces: ForcingFunction) -> ForcingFunction:
  """Sum multiple forcing functions."""
  def forcing(v, grid):
    return equations.sum_fields(*[forcing(v, grid) for forcing in forces])
  return forcing


@gin.configurable
def filtered_linear_forcing(grid: grids.Grid,
                            scale: float,
                            lower_wavenumber: float = 0,
                            upper_wavenumber: float = 4) -> ForcingFunction:
  del grid  # unused
  return forcings.filtered_linear_forcing(lower_wavenumber,
                                          upper_wavenumber,
                                          scale)


@gin.configurable
def linear_forcing(grid: grids.Grid,
                   scale: float) -> ForcingFunction:
  del grid  # unused
  return forcings.linear_forcing(scale)


@gin.configurable
def kolmogorov_forcing(grid: grids.Grid,  # pylint: disable=missing-function-docstring
                       scale: float = 0,
                       wavenumber: int = 2,
                       linear_coefficient: float = 0,
                       swap_xy: bool = False) -> ForcingFunction:
  force_fn = forcings.kolmogorov_forcing(grid, scale, wavenumber, swap_xy)
  if linear_coefficient != 0:
    linear_force_fn = forcings.linear_forcing(linear_coefficient)
    force_fn = forcings.sum_forcings(force_fn, linear_force_fn)
  return force_fn


@gin.configurable
def taylor_green_forcing(grid: grids.Grid,
                         scale: float = 0,
                         wavenumber: int = 2,
                         linear_coefficient: float = 0) -> ForcingFunction:
  force_fn = forcings.taylor_green_forcing(grid, scale, wavenumber)
  if linear_coefficient != 0:
    linear_force_fn = forcings.linear_forcing(linear_coefficient)
    force_fn = forcings.sum_forcings(force_fn, linear_force_fn)
  return force_fn


@gin.configurable
def no_forcing(grid: grids.Grid) -> ForcingFunction:
  return forcings.no_forcing(grid)
