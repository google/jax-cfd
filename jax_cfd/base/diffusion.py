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

"""Module for functionality related to diffusion."""
from typing import Optional, Sequence, Tuple

import jax.scipy.sparse.linalg

from jax_cfd.base import array_utils
from jax_cfd.base import fast_diagonalization
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids

Array = grids.Array
AlignedArray = grids.AlignedArray
AlignedField = Tuple[AlignedArray, ...]

# pylint: disable=g-bare-generic


def diffuse(c: AlignedArray, nu: float, grid: grids.Grid) -> AlignedArray:
  """Returns the rate of change in a concentration `c` due to diffusion."""
  return nu * fd.laplacian(c, grid)


def stable_time_step(viscosity: float, grid: grids.Grid) -> float:
  """Calculate a stable time step size for explicit diffusion.

  The calculation is based on analysis of central-time-central-space (CTCS)
  schemes.

  Args:
    viscosity: visosity
    grid: a `Grid` object.

  Returns:
    The prescribed time interval.
  """
  if viscosity == 0:
    return float("inf")
  dx = min(grid.step)
  ndim = grid.ndim
  return dx ** 2 / (viscosity * 2 ** ndim)


def solve_cg(v: Sequence[AlignedArray],
             nu: float,
             dt: float,
             grid: grids.Grid,
             rtol: float = 1e-6,
             atol: float = 1e-6,
             maxiter: Optional[int] = None) -> AlignedField:
  """Conjugate gradient solve for diffusion."""

  def linear_op(u: AlignedArray) -> AlignedArray:
    return u - dt * nu * fd.laplacian(u, grid)

  def inv(b: AlignedArray, x0: AlignedArray) -> AlignedArray:
    x, _ = jax.scipy.sparse.linalg.cg(
        linear_op, b, x0=x0, tol=rtol, atol=atol, maxiter=maxiter)
    return x

  return tuple(inv(u, u) for u in v)


def solve_fast_diag(v: Sequence[AlignedArray],
                    nu: float,
                    dt: float,
                    grid: grids.Grid,
                    implementation: Optional[str] = None) -> AlignedField:
  """Solve for diffusion using the fast diagonalization approach."""
  if grid.device_layout is not None:
    raise NotImplementedError(
        "distributed fast diagonalization not implemented yet.")
  # We reuse eigenvectors from the Laplacian and transform the eigenvalues
  # because this is better conditioned than directly diagonalizing 1 - ν Δt ∇²
  # when ν Δt is small.
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))

  # Transform the eigenvalues to implement (1 - ν Δt ∇²)⁻¹ (ν Δt ∇²)
  def func(x):
    dt_nu_x = (dt * nu) * x
    return dt_nu_x / (1 - dt_nu_x)

  # Note: this assumes that each velocity field has the same shape and dtype.
  op = fast_diagonalization.transform(
      func, laplacians, v[0].dtype,
      hermitian=True, circulant=True, implementation=implementation)

  # Compute (1 - ν Δt ∇²)⁻¹ u as u + (1 - ν Δt ∇²)⁻¹ (ν Δt ∇²) u, for less error
  # when ν Δt is small.
  return tuple(u + grids.applied(op)(u) for u in v)
