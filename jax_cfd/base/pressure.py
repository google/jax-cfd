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


def _rhs_transform(
    u: grids.GridArray,
    bc: boundaries.BoundaryConditions,
) -> Array:
  """Transform the RHS of pressure projection equation for stability.

  In case of poisson equation, the kernel is subtracted from RHS for stability.

  Args:
    u: a GridArray that solves ∇²x = u.
    bc: specifies boundary of x.

  Returns:
    u' s.t. u = u' + kernel of the laplacian.
  """
  u_data = u.data
  for axis in range(u.grid.ndim):
    if bc.types[axis][0] == boundaries.BCType.NEUMANN and bc.types[axis][
        1] == boundaries.BCType.NEUMANN:
      # if all sides are neumann, poisson solution has a kernel of constant
      # functions. We substact the mean to ensure consistency.
      u_data = u_data - jnp.mean(u_data)
  return u_data


def solve_cg(
    v: GridVariableVector,
    q0: GridVariable,
    pressure_bc: Optional[boundaries.ConstantBoundaryConditions] = None,
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
    pressure_bc: the boundary condition to assign to pressure. If None,
      boundary condition is infered from velocity.
    rtol: relative tolerance for convergence.
    atol: absolute tolerance for convergence.
    maxiter: optional int, the maximum number of iterations to perform.

  Returns:
    A pressure correction `q` such that `div(v - grad(q))` is zero.
  """
  # TODO(jamieas): add functionality for non-uniform density.
  rhs = fd.divergence(v)

  if pressure_bc is None:
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  def laplacian_with_bcs(array: GridArray) -> GridArray:
    variable = pressure_bc.impose_bc(array)
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
    q0: Optional[grids.GridArray] = None,
    pressure_bc: Optional[boundaries.ConstantBoundaryConditions] = None,
    implementation: Optional[str] = None,
) -> grids.GridArray:
  """Solve for pressure using the fast diagonalization approach.

  To support backward compatibility, if the pressure_bc are not provided and
  velocity has all periodic boundaries, pressure_bc are assigned to be periodic.

  Args:
    v: a tuple of velocity values for each direction.
    q0: the starting guess for the pressure.
    pressure_bc: the boundary condition to assign to pressure. If None,
      boundary condition is infered from velocity.
    implementation: how to implement fast diagonalization.
      For non-periodic BCs will automatically be matmul.


  Returns:
    A solution to the PPE equation.
  """
  del q0  # unused
  if pressure_bc is None:
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  if boundaries.has_all_periodic_boundary_conditions(*v):
    circulant = True
  else:
    circulant = False
    # only matmul implementation supports non-circulant matrices
    implementation = 'matmul'
  rhs = fd.divergence(v)
  laplacians = array_utils.laplacian_matrix_w_boundaries(
      rhs.grid, rhs.offset, pressure_bc)
  rhs_transformed = _rhs_transform(rhs, pressure_bc)
  pinv = fast_diagonalization.psuedoinverse(
      laplacians,
      rhs_transformed.dtype,
      hermitian=True,
      circulant=circulant,
      implementation=implementation)
  return grids.GridArray(pinv(rhs_transformed), rhs.offset, rhs.grid)


def solve_fast_diag_channel_flow(
    v: GridVariableVector,
    q0: Optional[grids.GridArray] = None,
    pressure_bc: Optional[boundaries.ConstantBoundaryConditions] = None,
) -> grids.GridArray:
  """Applies solve_fast_diag for channel flow.

  Args:
    v: a tuple of velocity values for each direction.
    q0: the starting guess for the pressure.
    pressure_bc: the boundary condition to assign to pressure. If None,
      boundary condition is infered from velocity.

  Returns:
    A solutiion to the PPE equation.
  """
  if pressure_bc is None:
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  return solve_fast_diag(v, q0, pressure_bc, implementation='matmul')


def projection(
    v: GridVariableVector,
    solve: Callable = solve_fast_diag,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free."""
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = pressure_bc.impose_bc(q0)

  q = solve(v, q0, pressure_bc)
  q = pressure_bc.impose_bc(q)
  q_grad = fd.forward_difference(q)
  v_projected = tuple(
      u.bc.impose_bc(u.array - q_g) for u, q_g in zip(v, q_grad))

  return v_projected
