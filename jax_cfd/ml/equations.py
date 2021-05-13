"""Implementations of equation modules."""

from typing import Any, Callable, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import equations
from jax_cfd.base import grids

from jax_cfd.ml import advections
from jax_cfd.ml import diffusions
from jax_cfd.ml import forcings
from jax_cfd.ml import networks  # pylint: disable=unused-import
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import pressures
from jax_cfd.ml import time_integrators

ConvectionModule = advections.ConvectionModule
DiffuseModule = diffusions.DiffuseModule
DiffusionSolveModule = diffusions.DiffusionSolveModule
ForcingModule = forcings.ForcingModule
PressureModule = pressures.PressureModule
# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic


# TODO(dkochkov) move diffusion to modular_navier_stokes after b/160947162.
@gin.configurable(denylist=("grid", "dt", "physics_specs"))
def semi_implicit_navier_stokes(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    diffusion_module: DiffuseModule = diffusions.diffuse,
    **kwargs,
):
  """Semi-implicit navier stokes solver compatible with explicit diffusion."""
  diffusion = diffusion_module(grid, dt, physics_specs)
  step_fn = equations.semi_implicit_navier_stokes(
      diffuse=diffusion, grid=grid, dt=dt, **kwargs)
  return hk.to_module(step_fn)()


@gin.configurable(denylist=("grid", "dt", "physics_specs"))
def implicit_diffusion_navier_stokes(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    diffusion_module: DiffusionSolveModule = diffusions.solve_fast_diag,
    **kwargs
):
  """Implicit navier stokes solver compatible with implicit diffusion."""
  diffusion = diffusion_module(grid, dt, physics_specs)
  step_fn = equations.implicit_diffusion_navier_stokes(
      diffusion_solve=diffusion, grid=grid, dt=dt, **kwargs)
  return hk.to_module(step_fn)()


@gin.configurable(denylist=("grid", "dt", "physics_specs"))
def modular_navier_stokes_model(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    equation_solver=implicit_diffusion_navier_stokes,
    convection_module: ConvectionModule = advections.self_advection,
    pressure_module: PressureModule = pressures.fast_diagonalization,
    acceleration_modules=(),
):
  """Returns an incompressible Navier-Stokes time step model.

  This model is derived from standard components of numerical solvers that could
  be replaced with learned components. Note that diffusion module is specified
  in the equation_solver due to differences in implicit/explicit schemes.

  Args:
    grid: grid on which the Navier-Stokes equation is discretized.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation module.
    equation_solver: solver to call to create a time-stepping function.
    convection_module: module to use to simulate convection.
    pressure_module: module to use to perform pressure projection.
    acceleration_modules: additional explicit terms to be adde to the equation
      before the pressure projection step.

  Returns:
    A function that performs `steps` steps of the Navier-Stokes time dynamics.
  """
  active_forcing_fn = physics_specs.forcing_module(grid)

  def navier_stokes_step_fn(state):
    """Advances Navier-Stokes state forward in time."""
    v = state
    convection = convection_module(grid, dt, physics_specs, v=v)
    accelerations = [
        acceleration_module(grid, dt, physics_specs, v=v)
        for acceleration_module in acceleration_modules
    ]
    forcing = forcings.sum_forcings(active_forcing_fn, *accelerations)
    pressure_solve_fn = pressure_module(grid, dt, physics_specs)
    step_fn = equation_solver(
        grid=grid,
        dt=dt,
        physics_specs=physics_specs,
        density=physics_specs.density,
        viscosity=physics_specs.viscosity,
        pressure_solve=pressure_solve_fn,
        convect=convection,
        forcing=forcing)
    return step_fn(v)

  return hk.to_module(navier_stokes_step_fn)()


@gin.configurable
def time_derivative_network_model(
    grid: grids.Grid,
    dt: float,
    physics_specs: Any,
    derivative_modules: Tuple[Callable, ...],
    time_integrator=time_integrators.euler_integrator,
):
  """Returns a ML model that performs time stepping by time integration.

  Note: the model state is assumed to be a stack of observable values
  along the last axis.

  Args:
    grid: grid specifying spatial discretization of the physical system.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation module.
    derivative_modules: tuple of modules that are used sequentially to compute
      unforced time derivative of the input state, which is then integrated.
    time_integrator: time integration scheme to use.

  Returns:
    `step_fn` that advances the input state forward in time by `dt`.
  """
  active_forcing_fn = physics_specs.forcing_module(grid)

  def step_fn(state):
    """Advances `state` forward in time by `dt`."""
    modules = [module(grid, dt, physics_specs) for module in derivative_modules]

    def time_derivative_fn(x):
      v = array_utils.split_axis(x, axis=-1)
      v = (grids.AlignedArray(u, o) for u, o in zip(v, grid.cell_faces))
      forcing_scalars = jnp.stack(
          [a.data for a in active_forcing_fn(v, None)], axis=-1)
      # TODO(dkochkov) consider conditioning on the forcing terms.
      for module_fn in modules:
        x = module_fn(x)
      return x + forcing_scalars

    time_derivative_module = hk.to_module(time_derivative_fn)()
    out, _ = time_integrator(time_derivative_module, state, dt, 1)
    return out

  return hk.to_module(step_fn)()


@gin.configurable
def learned_corrector(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    base_solver_module: Callable,
    corrector_module: Callable,
):
  """Returns a model that uses base solver with ML correction step."""
  # Idea similar to solver in the loop in https://arxiv.org/abs/2007.00016 and
  # learned corrector in https://arxiv.org/pdf/2102.01010.pdf.
  base_solver = base_solver_module(grid, dt, physics_specs)
  corrector = corrector_module(grid, dt, physics_specs)

  def step_fn(state):
    next_state = base_solver(state)
    corrections = corrector(next_state)
    return jax.tree_multimap(lambda x, y: x + y, next_state, corrections)

  return hk.to_module(step_fn)()


@gin.configurable
def learned_corrector_v2(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    base_solver_module: Callable,
    corrector_module: Callable,
):
  """Like learned_corrector, but based on the input rather than output state."""
  base_solver = base_solver_module(grid, dt, physics_specs)
  corrector = corrector_module(grid, dt, physics_specs)

  def step_fn(state):
    next_state = base_solver(state)
    corrections = corrector(state)
    return jax.tree_multimap(lambda x, y: x + dt * y, next_state, corrections)

  return hk.to_module(step_fn)()
