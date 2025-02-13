

from typing import Callable, Optional
import scipy.linalg
import numpy as np
from jax_ib.base import array_utils
from jax_cfd.base import fast_diagonalization
import jax.numpy as jnp
from jax_cfd.base import pressure
from jax_ib.base import grids
from jax_ib.base import boundaries
from jax_ib.base import finite_differences as fd
from jax_ib.base import particle_class

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


def laplacian_matrix_neumann(size: int, step: float) -> np.ndarray:
  """Create 1D Laplacian operator matrix, with homogeneous Neumann BC."""
  column = np.zeros(size)
  column[0] = -2 / step ** 2
  column[1] = 1 / step ** 2
  matrix = scipy.linalg.toeplitz(column)
  matrix[0, 0] = matrix[-1, -1] = -1 / step**2
  return matrix


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
  

def projection_and_update_pressure(
    All_variables: particle_class.All_Variables,
    solve: Callable = pressure.solve_fast_diag,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free."""
  v = All_variables.velocity
  old_pressure = All_variables.pressure
  particles = All_variables.particles
  Drag =  All_variables.Drag 
  Step_count = All_variables.Step_count
  MD_var = All_variables.MD_var
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = grids.GridVariable(q0, pressure_bc)

  qsol = solve(v, q0)
  q = grids.GridVariable(qsol, pressure_bc)
    
  New_pressure_Array =  grids.GridArray(qsol.data + old_pressure.data,qsol.offset,qsol.grid)  
  New_pressure = grids.GridVariable(New_pressure_Array,pressure_bc) 

  q_grad = fd.forward_difference(q)
  if boundaries.has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
    new_variable = particle_class.All_Variables(particles,v_projected,New_pressure,Drag,Step_count,MD_var)
  else:
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc).impose_bc()
        for u, q_g in zip(v, q_grad))
    new_variable = particle_class.All_Variables(particles,v_projected,New_pressure,Drag,Step_count,MD_var)
  return new_variable


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
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True, implementation=implementation)
  return grids.applied(pinv)(rhs)


def solve_fast_diag_moving_wall(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = 'matmul') -> GridArray:
  """Solve for channel flow pressure using fast diagonalization."""
  del q0  # unused
  ndim = len(v)

  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = [
      array_utils.laplacian_matrix(grid.shape[0], grid.step[0]),
      array_utils.laplacian_matrix_neumann(grid.shape[1], grid.step[1]),
  ]
  for d in range(2, ndim):
    laplacians += [array_utils.laplacian_matrix(grid.shape[d], grid.step[d])]
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=False, implementation=implementation)
  return grids.applied(pinv)(rhs)
  
  
  
def solve_fast_diag_Far_Field(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None) -> GridArray:
  """Solve for pressure using the fast diagonalization approach."""
  del q0  # unused

  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  rhs_transformed = _rhs_transform(rhs, pressure_bc)
  #laplacians = [
  #          laplacian_matrix_neumann(grid.shape[0], grid.step[0]),
  #          laplacian_matrix_neumann(grid.shape[1], grid.step[1]),
  #]
  laplacians = array_utils.laplacian_matrix_w_boundaries(
      rhs.grid, rhs.offset, pressure_bc)
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs_transformed.dtype,
      hermitian=True, circulant=False, implementation='matmul')
  return grids.applied(pinv)(rhs)

def calc_P(
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

  return q
