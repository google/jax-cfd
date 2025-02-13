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

# TODO(pnorgaard) Implement bicgstab for non-symmetric operators

"""Module for functionality related to diffusion."""
from typing import Optional

import jax.scipy.sparse.linalg

from jax_cfd.base import array_utils
from jax_ib.base import boundaries
from jax_cfd.base import fast_diagonalization
from jax_ib.base import finite_differences as fd
from jax_ib.base import grids

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def diffuse(c: GridVariable, nu: float) -> GridArray:
  """Returns the rate of change in a concentration `c` due to diffusion."""
  return nu * fd.laplacian(c)


def stable_time_step(viscosity: float, grid: grids.Grid) -> float:
  """Calculate a stable time step size for explicit diffusion.

  The calculation is based on analysis of central-time-central-space (CTCS)
  schemes.

  Args:
    viscosity: kinematic visosity
    grid: a `Grid` object.

  Returns:
    The prescribed time interval.
  """
  if viscosity == 0:
    return float('inf')
  dx = min(grid.step)
  ndim = grid.ndim
  return dx ** 2 / (viscosity * 2 ** ndim)


def solve_cg(v: GridVariableVector,
             nu: float,
             dt: float,
             rtol: float = 1e-6,
             atol: float = 1e-6,
             maxiter: Optional[int] = None) -> GridVariableVector:
  """Conjugate gradient solve for diffusion."""
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_cg() expects periodic BC')

  def solve_component(u: GridVariable) -> GridArray:
    """Solves (1 - ν Δt ∇²) u_{t+1} = u_{tilda} for u_{t+1}."""

    def linear_op(u_new: GridArray) -> GridArray:
      """Linear operator for (1 - ν Δt ∇²) u_{t+1}."""
      u_new = grids.GridVariable(u_new, u.bc)  # get boundary condition from u
      return u_new.array - dt * nu * fd.laplacian(u_new)

    def cg(b: GridArray, x0: GridArray) -> GridArray:
      """Iteratively solves Lx = b. with initial guess x0."""
      x, _ = jax.scipy.sparse.linalg.cg(
          linear_op, b, x0=x0, tol=rtol, atol=atol, maxiter=maxiter)
      return x

    return cg(u.array, u.array)

  return tuple(grids.GridVariable(solve_component(u), u.bc) for u in v)


def solve_fast_diag(v: GridVariableVector,
                    nu: float,
                    dt: float,
                    implementation: Optional[str] = None) -> GridVariableVector:
  """Solve for diffusion using the fast diagonalization approach."""
  # We reuse eigenvectors from the Laplacian and transform the eigenvalues
  # because this is better conditioned than directly diagonalizing 1 - ν Δt ∇²
  # when ν Δt is small.
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic BC')
  grid = grids.consistent_grid(*v)
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
  return tuple(grids.GridVariable(u.array + grids.applied(op)(u.array), u.bc)
               for u in v)
