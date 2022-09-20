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
from typing import Optional, Tuple

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


def _subtract_linear_part_dirichlet(
    c_data: Array,
    grid: grids.Grid,
    axis: int,
    offset: Tuple[float, float],
    bc_values: Tuple[float, float],
) -> Array:
  """Transforms c_data such that c_data satisfies dirichlet boundary.

  The function subtracts a linear function from c_data s.t. the returned
  array has homogeneous dirichlet boundaries. Note that this assumes c_data has
  constant dirichlet boundary values.

  Args:
    c_data: right-hand-side of diffusion equation.
    grid: grid object
    axis: axis along which to impose boundary transformation
    offset: offset of the right-hand-side
    bc_values: boundary values along axis

  Returns:
    transformed right-hand-side
  """

  def _update_rhs_along_axis(arr_1d, linear_part):
    arr_1d = arr_1d - linear_part
    return arr_1d

  lower_value, upper_value = bc_values
  y = grid.mesh(offset)[axis][0]
  one_d_grid = grids.Grid((grid.shape[axis],), domain=(grid.domain[axis],))
  y_boundary = boundaries.dirichlet_boundary_conditions(ndim=1)
  y = y_boundary.trim_boundary(grids.GridArray(y, (offset[axis],),
                                               one_d_grid)).data
  domain_length = (grid.domain[axis][1] - grid.domain[axis][0])
  domain_start = grid.domain[axis][0]
  linear_part = lower_value + (upper_value - lower_value) * (
      y - domain_start) / domain_length
  c_data = jnp.apply_along_axis(
      _update_rhs_along_axis, axis, c_data, linear_part)
  return c_data


def _rhs_transform(
    u: grids.GridArray,
    bc: boundaries.BoundaryConditions,
) -> Array:
  """Transforms the RHS of diffusion equation.

  In case of constant dirichlet boundary conditions for heat equation
  the linear term is subtracted. See diffusion.solve_fast_diag.

  Args:
    u: a GridArray that solves ∇²x = ∇²u for x.
    bc: specifies boundary of u.

  Returns:
    u' s.t. u = u' + w where u' has 0 dirichlet bc and w is linear.
  """
  if not isinstance(bc, boundaries.ConstantBoundaryConditions):
    raise NotImplementedError(
        f'transformation cannot be done for this {bc}.')
  u_data = u.data
  for axis in range(u.grid.ndim):
    for i, _ in enumerate(['lower', 'upper']):  # lower and upper boundary
      if bc.types[axis][i] == boundaries.BCType.DIRICHLET:
        bc_values = [0., 0.]
        bc_values[i] = bc.bc_values[axis][i]
        u_data = _subtract_linear_part_dirichlet(u_data, u.grid, axis, u.offset,
                                                 bc_values)
      elif bc.types[axis][i] == boundaries.BCType.NEUMANN:
        if any(bc.bc_values[axis]):
          raise NotImplementedError(
              'transformation is not implemented for inhomogeneous Neumann bc.')
  return u_data


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


def solve_fast_diag(
    v: GridVariableVector,
    nu: float,
    dt: float,
    implementation: Optional[str] = None,
) -> GridVariableVector:
  """Solve for diffusion using the fast diagonalization approach."""
  # We reuse eigenvectors from the Laplacian and transform the eigenvalues
  # because this is better conditioned than directly diagonalizing 1 - ν Δt ∇²
  # when ν Δt is small.
  def func(x):
    dt_nu_x = (dt * nu) * x
    return dt_nu_x / (1 - dt_nu_x)

  # Compute (1 - ν Δt ∇²)⁻¹ u as u + (1 - ν Δt ∇²)⁻¹ (ν Δt ∇²) u, for less
  # error when ν Δt is small.
  # If dirichlet bc are supplied: only works for dirichlet bc that are linear
  # functions on the boundary. Then u = u' + w where u' has 0 dirichlet bc and
  # w is linear. Then u + (1 - ν Δt ∇²)⁻¹ (ν Δt ∇²) u = u +
  # (1 - ν Δt ∇²)⁻¹(ν Δt ∇²)u'. The function _rhs_transform subtracts
  # the linear part s.t. fast_diagonalization solves
  # u + (1 - ν Δt ∇²)⁻¹ (ν Δt ∇²) u'.
  v_diffused = list()
  if boundaries.has_all_periodic_boundary_conditions(*v):
    circulant = True
  else:
    circulant = False
    # only matmul implementation supports non-circulant matrices
    implementation = 'matmul'
  for u in v:
    laplacians = array_utils.laplacian_matrix_w_boundaries(
        u.grid, u.offset, u.bc)
    op = fast_diagonalization.transform(
        func,
        laplacians,
        v[0].dtype,
        hermitian=True,
        circulant=circulant,
        implementation=implementation)
    u_interior = u.bc.trim_boundary(u.array)
    u_interior_transformed = _rhs_transform(u_interior, u.bc)
    u_dt_diffused = grids.GridArray(
        op(u_interior_transformed), u_interior.offset, u_interior.grid)
    u_diffused = u_interior + u_dt_diffused
    u_diffused = u.bc.pad_and_impose_bc(u_diffused, offset_to_pad_to=u.offset)
    v_diffused.append(u_diffused)
  return tuple(v_diffused)
