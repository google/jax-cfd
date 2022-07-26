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

"""Functions for computing and applying pressure."""

from typing import Callable, Optional

import jax.numpy as jnp
import jax.scipy.sparse.linalg

from jax_cfd.base import array_utils
from jax_cfd.base import boundaries
from jax_cfd.base import fast_diagonalization
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic


# TODO(pnorgaard) Implement bicgstab for non-symmetric operators


def solve_cg(
    v: GridVariableVector,
    q0: GridVariable,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    maxiter: Optional[int] = None) -> GridArray:
  """Conjugate gradient solve for the pressure such that continuity is enforced.

  Returns a pressure correction `q` such that `div(v - grad(q)) == 0`.

  The relationship between `q` and our actual pressure estimate is given by
  `p = q * density / dt`.

  Args:
    v: the velocity field.
    q0: an initial value, or "guess" for the pressure correction. A common
      choice is the correction from the previous time step. Also specifies the
      boundary conditions on `q`.
    rtol: relative tolerance for convergence.
    atol: absolute tolerance for convergence.
    maxiter: optional int, the maximum number of iterations to perform.

  Returns:
    A pressure correction `q` such that `div(v - grad(q))` is zero.
  """
  # TODO(jamieas): add functionality for non-uniform density.
  rhs = fd.divergence(v)

  def laplacian_with_bcs(array: GridArray) -> GridArray:
    variable = grids.GridVariable(array, q0.bc)
    return fd.laplacian(variable)

  q, _ = jax.scipy.sparse.linalg.cg(
      laplacian_with_bcs,
      rhs,
      x0=q0.array,
      tol=rtol,
      atol=atol,
      maxiter=maxiter)
  return q


def solve_fast_diag(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None) -> GridArray:
  """Solve for pressure using the fast diagonalization approach."""
  del q0  # unused
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic velocity BC')
  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))
  pinv = fast_diagonalization.psuedoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True, implementation=implementation)
  return grids.applied(pinv)(rhs)


def solve_fast_diag_channel_flow(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = 'matmul') -> GridArray:
  """Solve for channel flow pressure using fast diagonalization."""
  del q0  # unused
  ndim = len(v)
  for u in v:
    if u.bc != boundaries.channel_flow_boundary_conditions(ndim):
      raise ValueError('Only for use with channel flow boundary conditions.')
  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = [
      array_utils.laplacian_matrix(grid.shape[0], grid.step[0]),
      array_utils.laplacian_matrix_neumann(grid.shape[1], grid.step[1]),
  ]
  for d in range(2, ndim):
    laplacians += [array_utils.laplacian_matrix(grid.shape[d], grid.step[d])]
  pinv = fast_diagonalization.psuedoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=False, implementation=implementation)
  return grids.applied(pinv)(rhs)


def projection(
    v: GridVariableVector,
    solve: Callable = solve_fast_diag,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free."""
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = grids.GridVariable(q0, pressure_bc)

  q = solve(v, q0)
  q = grids.GridVariable(q, pressure_bc)
  q_grad = fd.forward_difference(q)
  if boundaries.has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
  else:
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc).impose_bc()
        for u, q_g in zip(v, q_grad))
  return v_projected
