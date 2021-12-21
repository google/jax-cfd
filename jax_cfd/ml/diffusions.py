"""Models for diffusion components.

All modules are functions that return `DiffuseFn` or `DiffusionSolveFn` method.
The two types of diffusion modules should be used with corresponding explicit
and implicit navier-stokes solvers.

An example explicit diffusion module:

```python
def diffusion_module(dt, module_params, **kwargs):
  pre_compute_values = f(dt, module_params)
  def diffuse(c: grids.GridVariable, nu: float, grid: grids.Grid, dt: float):
    # compute time derivative due to diffusion.
    return dc_dt

  return diffuse
```
"""
import functools
from typing import Callable, Optional
import gin
import haiku as hk
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.base import subgrid_models

from jax_cfd.ml import viscosities


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
DiffuseFn = Callable[[GridVariable, float], GridArray]
DiffusionSolveFn = Callable[[GridVariableVector, float, float],
                            GridVariableVector]
DiffuseModule = Callable[..., DiffuseFn]
DiffusionSolveModule = Callable[..., DiffusionSolveFn]
ViscosityModule = viscosities.ViscosityModule


# TODO(shoyer): stop deleting unrecognized **kwargs. This is really error-prone!


@gin.register(denylist=("grid", "dt", "physics_specs"))
def diffuse(grid, dt, physics_specs) -> DiffuseFn:
  del grid, dt, physics_specs  # unused.
  return diffusion.diffuse


@gin.register(denylist=("grid", "dt", "physics_specs"))
def solve_fast_diag(
    grid,
    dt,
    physics_specs,
    implementation=None
) -> DiffusionSolveFn:
  del grid, dt, physics_specs  # unused.
  return functools.partial(
      diffusion.solve_fast_diag, implementation=implementation)


@gin.register(denylist=("grid", "dt", "physics_specs"))
def solve_cg(
    grid,
    dt,
    physics_specs,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    maxiter: Optional[int] = 64,
) -> DiffusionSolveFn:
  """Returns conjugate gradient solve method."""
  del grid, dt, physics_specs  # unused.
  return functools.partial(
      diffusion.solve_cg, atol=atol, rtol=rtol, maxiter=maxiter)


@gin.register(denylist=("grid", "dt", "physics_specs"))
def implicit_evm_solve_with_diffusion(
    grid,
    dt,
    physics_specs,
    viscosity_module: ViscosityModule = viscosities.eddy_viscosity_model,
    atol: float = 1e-5,
    maxiter: Optional[int] = 64,
) -> DiffusionSolveFn:
  """Returns solve_diffusion method that also includes a viscosity model."""
  evm_model = viscosity_module(grid, dt, physics_specs)
  cg_kwargs = dict(atol=atol, maxiter=maxiter)
  diffusion_solve = functools.partial(
      subgrid_models.implicit_evm_solve_with_diffusion,
      configured_evm_model=evm_model,
      cg_kwargs=cg_kwargs)
  return hk.to_module(diffusion_solve)(name="diffusion_solve")
