"""Models for pressure solvers.

All modules are functions that return `pressure_solve` method that has the same
signature as baseline methods e.g. `pressure.solve_fast_diag`.
"""
import functools
from typing import Callable, Optional

import gin

from jax_cfd.base import grids
from jax_cfd.base import pressure


GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
PressureSolveFn = Callable[
    [GridVariableVector, Optional[GridVariable]], GridArray]
PressureModule = Callable[..., PressureSolveFn]


@gin.register
def fast_diagonalization(grid, dt, physics_specs):
  del grid, dt, physics_specs  # unused.
  return pressure.solve_fast_diag


@gin.register
def conjugate_gradient(grid, dt, physics_specs, atol=1e-5, maxiter=32):
  del grid, dt, physics_specs  # unused.
  return functools.partial(pressure.solve_cg, atol=atol, maxiter=maxiter)
