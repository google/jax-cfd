# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forcing functions for Navier-Stokes equations."""

# TODO(jamieas): change the signature for all forcing functions so that they
# close over `grid`.

import functools
from typing import Callable, Union

import jax.numpy as jnp
from jax_cfd.base import equations
from jax_cfd.base import grids
from jax_cfd.base import spectral
from jax_cfd.base import validation_problems
import numpy as np

AlignedArray = grids.AlignedArray
Array = Union[np.ndarray, jnp.DeviceArray]
ForcingFunction = Callable[
    [equations.AlignedField, grids.Grid], equations.AlignedField]


def taylor_green_forcing(
    grid: grids.Grid, scale: float = 1, k: int = 2,
) -> ForcingFunction:
  """Constant driving forced in the form of Taylor-Green vorcities."""
  u, v = validation_problems.TaylorGreen(
      shape=grid.shape[:2], kx=k, ky=k).velocity()
  u, v = u * scale, v * scale
  if grid.ndim == 2:
    f = (u, v)
  elif grid.ndim == 3:
    u = grids.AlignedArray(u.data, u.offset + (1/2,))
    v = grids.AlignedArray(v.data, v.offset + (1/2,))
    w = grids.AlignedArray(jnp.zeros_like(u.data), (1/2, 1/2, 1))
    f = (u, v, w)
  else:
    raise NotImplementedError

  def forcing(v, grid):
    del v, grid
    return f
  return forcing


def kolmogorov_forcing(
    grid: grids.Grid,
    scale: float = 1,
    k: int = 2,
    swap_xy: bool = False,
) -> ForcingFunction:
  """Returns the Kolmogorov forcing function for turbulence in 2D."""
  offsets = grid.cell_faces

  if swap_xy:
    x = grid.mesh(offsets[1])[0]
    v = scale * grids.AlignedArray(jnp.sin(k * x), offsets[1])

    if grid.ndim == 2:
      u = grids.AlignedArray(jnp.zeros_like(v.data), (1, 1/2))
      f = (u, v)
    elif grid.ndim == 3:
      u = grids.AlignedArray(jnp.zeros_like(v.data), (1, 1/2, 1/2))
      w = grids.AlignedArray(jnp.zeros_like(u.data), (1/2, 1/2, 1))
      f = (u, v, w)
    else:
      raise NotImplementedError
  else:
    y = grid.mesh(offsets[0])[1]
    u = scale * grids.AlignedArray(jnp.sin(k * y), offsets[0])

    if grid.ndim == 2:
      v = grids.AlignedArray(jnp.zeros_like(u.data), (1/2, 1))
      f = (u, v)
    elif grid.ndim == 3:
      v = grids.AlignedArray(jnp.zeros_like(u.data), (1/2, 1, 1/2))
      w = grids.AlignedArray(jnp.zeros_like(u.data), (1/2, 1/2, 1))
      f = (u, v, w)
    else:
      raise NotImplementedError

  def forcing(v, grid):
    del v, grid
    return f
  return forcing


def filtered_forcing(
    spectral_density: Callable[[Array], Array]
) -> ForcingFunction:
  """Apply forcing as a function of angular frequency.

  Args:
    spectral_density: if `x_hat` is a Fourier component of the velocity with
      angular frequency `k` then the forcing applied to `x_hat` is
      `spectral_density(k)`.
  Returns:
    A forcing function that applies filtered forcing.
  """
  def forcing(v, grid):
    filter_ = grids.applied(
        functools.partial(spectral.filter, spectral_density, grid=grid))
    return tuple(filter_(u) for u in v)
  return forcing


def filtered_linear_forcing(
    lower_wavenumber: float,
    upper_wavenumber: float,
    coefficient: float
) -> ForcingFunction:
  """Apply linear forcing to low frequency components of the velocity field.

  Args:
    lower_wavenumber: the minimum wavenumber to which forcing should be
      applied.
    upper_wavenumber: the maximum wavenumber to which forcing should be
      applied.
    coefficient: the linear coefficient for forcing applied to components with
      wavenumber below `threshold`.
  Returns:
    A forcing function that applies filtered linear forcing.
  """
  def spectral_density(k):
    return jnp.where(((k >= 2 * jnp.pi * lower_wavenumber) &
                      (k <= 2 * jnp.pi * upper_wavenumber)),
                     coefficient,
                     0)
  return filtered_forcing(spectral_density)


def linear_forcing(coefficient: float) -> ForcingFunction:
  """Linear forcing, proportional to velocity."""
  def forcing(v, grid):
    del grid  # unused
    return tuple(coefficient * u for u in v)
  return forcing


def no_forcing(grid):
  """Zero-valued forcing field for unforced simulations."""
  offsets = grid.cell_faces

  def forcing(v, grid):
    del grid  # unused
    return (grids.AlignedArray(jnp.zeros_like(u.data), o)
            for u, o in zip(v, offsets))
  return forcing


def sum_forcings(*forcings: ForcingFunction) -> ForcingFunction:
  """Sum multiple forcing functions."""
  def forcing(v, grid):
    return equations.sum_fields(*[forcing(v, grid) for forcing in forcings])
  return forcing


FORCING_FUNCTIONS = dict(kolmogorov=kolmogorov_forcing,
                         taylor_green=taylor_green_forcing)


def simple_turbulence_forcing(
    grid: grids.Grid,
    constant_magnitude: float = 0,
    constant_wavenumber: int = 2,
    linear_coefficient: float = 0,
    forcing_type: str = 'kolmogorov',
) -> ForcingFunction:
  """Returns a forcing function for turbulence in 2D or 3D.

  2D turbulence needs a driving force injecting energy at intermediate
  length-scales, and a damping force at long length-scales to avoid all energy
  accumulating in giant vorticies. This can be achieved with
  `constant_magnitude > 0` and `linear_coefficient < 0`.

  3D turbulence only needs a driving force at the longest length-scale (damping
  happens at the smallest length-scales due to viscosity and/or numerical
  dispersion). This can be achieved with `constant_magnitude = 0` and
  `linear_coefficient > 0`.

  Args:
    grid: grid on which to simulate.
    constant_magnitude: magnitude for constant forcing with Taylor-Green
      vortices.
    constant_wavenumber: wavenumber for constant forcing with Taylor-Green
      vortices.
    linear_coefficient: forcing coefficient proportional to velocity, for
      either driving or damping based on the sign.
    forcing_type: String that specifies forcing. This must specify the name of
      function declared in FORCING_FUNCTIONS (taylor_green, etc.)

  Returns:
    Forcing function.
  """

  linear_force = linear_forcing(linear_coefficient)
  constant_force_fn = FORCING_FUNCTIONS.get(forcing_type)
  if constant_force_fn is None:
    raise ValueError('Unknown `forcing_type`. '
                     f'Expected one of {list(FORCING_FUNCTIONS.keys())}; '
                     f'got {forcing_type}.')
  constant_force = constant_force_fn(grid, constant_magnitude,
                                     constant_wavenumber)
  return sum_forcings(linear_force, constant_force)
