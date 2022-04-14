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
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax_cfd.base import equations
from jax_cfd.base import filter_utils
from jax_cfd.base import grids
from jax_cfd.base import validation_problems

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


def taylor_green_forcing(
    grid: grids.Grid, scale: float = 1, k: int = 2,
) -> ForcingFn:
  """Constant driving forced in the form of Taylor-Green vorcities."""
  u, v = validation_problems.TaylorGreen(
      shape=grid.shape[:2], kx=k, ky=k).velocity()
  # Put force on same offset, grid as velocity components
  if grid.ndim == 2:
    u = grids.GridArray(u.data * scale, u.offset, grid)
    v = grids.GridArray(v.data * scale, v.offset, grid)
    f = (u, v)
  elif grid.ndim == 3:
    # append z-dimension to u,v arrays
    u_data = jnp.broadcast_to(jnp.expand_dims(u.data * scale, -1), grid.shape)
    v_data = jnp.broadcast_to(jnp.expand_dims(v.data * scale, -1), grid.shape)
    u = grids.GridArray(u_data, (1, 0.5, 0.5), grid)
    v = grids.GridArray(v_data, (0.5, 1, 0.5), grid)
    w = grids.GridArray(jnp.zeros_like(u.data), (0.5, 0.5, 1), grid)
    f = (u, v, w)
  else:
    raise NotImplementedError

  def forcing(v):
    del v
    return f
  return forcing


def kolmogorov_forcing(
    grid: grids.Grid,
    scale: float = 1,
    k: int = 2,
    swap_xy: bool = False,
    offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> ForcingFn:
  """Returns the Kolmogorov forcing function for turbulence in 2D."""
  if offsets is None:
    offsets = grid.cell_faces

  if swap_xy:
    x = grid.mesh(offsets[1])[0]
    v = scale * grids.GridArray(jnp.sin(k * x), offsets[1], grid)

    if grid.ndim == 2:
      u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2), grid)
      f = (u, v)
    elif grid.ndim == 3:
      u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2, 1/2), grid)
      w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
      f = (u, v, w)
    else:
      raise NotImplementedError
  else:
    y = grid.mesh(offsets[0])[1]
    u = scale * grids.GridArray(jnp.sin(k * y), offsets[0], grid)

    if grid.ndim == 2:
      v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1), grid)
      f = (u, v)
    elif grid.ndim == 3:
      v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1, 1/2), grid)
      w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
      f = (u, v, w)
    else:
      raise NotImplementedError

  def forcing(v):
    del v
    return f
  return forcing


def linear_forcing(grid, coefficient: float) -> ForcingFn:
  """Linear forcing, proportional to velocity."""
  del grid

  def forcing(v):
    return tuple(coefficient * u.array for u in v)
  return forcing


def no_forcing(grid):
  """Zero-valued forcing field for unforced simulations."""
  del grid

  def forcing(v):
    return tuple(0 * u.array for u in v)
  return forcing


def sum_forcings(*forcings: ForcingFn) -> ForcingFn:
  """Sum multiple forcing functions."""
  def forcing(v):
    return equations.sum_fields(*[forcing(v) for forcing in forcings])
  return forcing


FORCING_FUNCTIONS = dict(kolmogorov=kolmogorov_forcing,
                         taylor_green=taylor_green_forcing)


def simple_turbulence_forcing(
    grid: grids.Grid,
    constant_magnitude: float = 0,
    constant_wavenumber: int = 2,
    linear_coefficient: float = 0,
    forcing_type: str = 'kolmogorov',
) -> ForcingFn:
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

  linear_force = linear_forcing(grid, linear_coefficient)
  constant_force_fn = FORCING_FUNCTIONS.get(forcing_type)
  if constant_force_fn is None:
    raise ValueError('Unknown `forcing_type`. '
                     f'Expected one of {list(FORCING_FUNCTIONS.keys())}; '
                     f'got {forcing_type}.')
  constant_force = constant_force_fn(grid, constant_magnitude,
                                     constant_wavenumber)
  return sum_forcings(linear_force, constant_force)


def filtered_forcing(
    spectral_density: Callable[[Array], Array],
    grid: grids.Grid,
) -> ForcingFn:
  """Apply forcing as a function of angular frequency.

  Args:
    spectral_density: if `x_hat` is a Fourier component of the velocity with
      angular frequency `k` then the forcing applied to `x_hat` is
      `spectral_density(k)`.
    grid: object representing spatial discretization.
  Returns:
    A forcing function that applies filtered forcing.
  """
  def forcing(v):
    filter_ = grids.applied(
        functools.partial(filter_utils.filter, spectral_density, grid=grid))
    return tuple(filter_(u.array) for u in v)
  return forcing


def filtered_linear_forcing(
    lower_wavenumber: float,
    upper_wavenumber: float,
    coefficient: float,
    grid: grids.Grid,
) -> ForcingFn:
  """Apply linear forcing to low frequency components of the velocity field.

  Args:
    lower_wavenumber: the minimum wavenumber to which forcing should be
      applied.
    upper_wavenumber: the maximum wavenumber to which forcing should be
      applied.
    coefficient: the linear coefficient for forcing applied to components with
      wavenumber below `threshold`.
    grid: object representing spatial discretization.
  Returns:
    A forcing function that applies filtered linear forcing.
  """
  def spectral_density(k):
    return jnp.where(((k >= 2 * jnp.pi * lower_wavenumber) &
                      (k <= 2 * jnp.pi * upper_wavenumber)),
                     coefficient,
                     0)
  return filtered_forcing(spectral_density, grid)
