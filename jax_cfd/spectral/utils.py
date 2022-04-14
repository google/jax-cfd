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

"""Helper functions for building pseudospectral methods."""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.spectral import types as spectral_types


def truncated_rfft(u: spectral_types.Array) -> spectral_types.Array:
  """Applies the 2/3 rule by truncating higher Fourier modes.

  Args:
    u: the real-space representation of the input signal

  Returns:
    Downsampled version of `u` in rfft-space.
  """
  uhat = jnp.fft.rfft(u)
  k, = uhat.shape
  final_size = int(2 / 3 * k) + 1
  return 2 / 3 * uhat[:final_size]


def padded_irfft(uhat: spectral_types.Array) -> spectral_types.Array:
  """Applies the 3/2 rule by padding with zeros.

  Args:
    uhat: the rfft representation of a signal

  Returns:
    An upsampled signal in real space which 3/2 times larger than the input
    signal `uhat`.
  """
  n, = uhat.shape
  final_shape = int(3 / 2 * n)
  smoothed = jnp.pad(uhat, (0, final_shape - n))
  assert smoothed.shape == (final_shape,), "incorrect padded shape"
  return 1.5 * jnp.fft.irfft(smoothed)


def circular_filter_2d(grid: grids.Grid) -> spectral_types.Array:
  """Circular filter which roughly matches the 2/3 rule but is smoother.

  Follows the technique described in Equation 1 of [1]. We use a different value
  for alpha as used by pyqg [2].

  Args:
    grid: the grid to filter over

  Returns:
    Filter mask

  Reference:
    [1] Arbic, Brian K., and Glenn R. Flierl. "Coherent vortices and kinetic
    energy ribbons in asymptotic, quasi two-dimensional f-plane turbulence."
    Physics of Fluids 15, no. 8 (2003): 2177-2189.
    https://doi.org/10.1063/1.1582183

    [2] Ryan Abernathey, rochanotes, Malte Jansen, Francis J. Poulin, Navid C.
    Constantinou, Dhruv Balwada, Anirban Sinha, Mike Bueti, James Penn,
    Christopher L. Pitt Wolfe, & Bia Villas Boas. (2019). pyqg/pyqg: v0.3.0
    (v0.3.0). Zenodo. https://doi.org/10.5281/zenodo.3551326.
    See:
    https://github.com/pyqg/pyqg/blob/02e8e713660d6b2043410f2fef6a186a7cb225a6/pyqg/model.py#L136
  """
  kx, ky = grid.rfft_mesh()
  max_k = ky[-1, -1]

  circle = jnp.sqrt(kx**2 + ky**2)
  cphi = 0.65 * max_k
  filterfac = 23.6
  filter_ = jnp.exp(-filterfac * (circle - cphi)**4.)
  filter_ = jnp.where(circle <= cphi, jnp.ones_like(filter_), filter_)
  return filter_


def brick_wall_filter_2d(grid: grids.Grid):
  """Implements the 2/3 rule."""
  n, _ = grid.shape
  filter_ = jnp.zeros((n, n // 2 + 1))
  filter_ = filter_.at[:int(2 / 3 * n) // 2, :int(2 / 3 * (n // 2 + 1))].set(1)
  filter_ = filter_.at[-int(2 / 3 * n) // 2:, :int(2 / 3 * (n // 2 + 1))].set(1)
  return filter_


def exponential_filter(signal, alpha=1e-6, order=2):
  """Apply a low-pass smoothing filter to remove noise from 2D signal."""
  # Based on:
  # 1. Gottlieb and Hesthaven (2001), "Spectral methods for hyperbolic problems"
  # https://doi.org/10.1016/S0377-0427(00)00510-0
  # 2. Also, see https://arxiv.org/pdf/math/0701337.pdf --- Eq. 5

  # TODO(dresdner) save a few ffts by factoring out the actual filter, sigma.
  alpha = -jnp.log(alpha)
  n, _ = signal.shape  # TODO(dresdner) check square / handle 1D case
  kx, ky = jnp.fft.fftfreq(n), jnp.fft.rfftfreq(n)
  kx, ky = jnp.meshgrid(kx, ky, indexing="ij")
  eta = jnp.sqrt(kx**2 + ky**2)
  sigma = jnp.exp(-alpha * eta**(2 * order))
  return jnp.fft.irfft2(sigma * jnp.fft.rfft2(signal))


def vorticity_to_velocity(
    grid: grids.Grid
) -> Callable[[spectral_types.Array], Tuple[spectral_types.Array,
                                            spectral_types.Array]]:
  """Constructs a function for converting vorticity to velocity, both in Fourier domain.

  Solves for the stream function and then uses the stream function to compute
  the velocity. This is the standard approach. A quick sketch can be found in
  [1].

  Args:
    grid: the grid underlying the vorticity field.

  Returns:
    A function that takes a vorticity (rfftn) and returns a velocity vector
    field.

  Reference:
    [1] Z. Yin, H.J.H. Clercx, D.C. Montgomery, An easily implemented task-based
    parallel scheme for the Fourier pseudospectral solver applied to 2D
    Navier–Stokes turbulence, Computers & Fluids, Volume 33, Issue 4, 2004,
    Pages 509-520, ISSN 0045-7930,
    https://doi.org/10.1016/j.compfluid.2003.06.003.
  """
  kx, ky = grid.rfft_mesh()
  two_pi_i = 2 * jnp.pi * 1j
  laplace = two_pi_i ** 2 * (abs(kx)**2 + abs(ky)**2)
  laplace = laplace.at[0, 0].set(1)

  def ret(vorticity_hat):
    psi_hat = -1 / laplace * vorticity_hat
    vxhat = two_pi_i * ky * psi_hat
    vyhat = -two_pi_i * kx * psi_hat
    return vxhat, vyhat

  return ret


def filter_step(step_fn: spectral_types.StepFn, filter_: spectral_types.Array):
  """Returns a filtered version of the step_fn."""
  def new_step_fn(state):
    return filter_ * step_fn(state)
  return new_step_fn


def spectral_curl_2d(mesh, velocity_hat):
  """Computes the 2D curl in the Fourier basis."""
  kx, ky = mesh
  uhat, vhat = velocity_hat
  return 2j * jnp.pi * (vhat * kx - uhat * ky)
