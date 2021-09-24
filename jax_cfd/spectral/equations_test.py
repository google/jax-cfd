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
"""Tests for spectral equations."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import jax_cfd.base as cfd
from jax_cfd.base import finite_differences
from jax_cfd.base import grids
from jax_cfd.base import test_util
from jax_cfd.spectral import equations as spectral_equations
from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping

ALL_TIME_STEPPERS = [
    time_stepping.backward_forward_euler,
    time_stepping.crank_nicolson_rk2,
    time_stepping.crank_nicolson_rk3,
    time_stepping.crank_nicolson_rk4,
]

ALL_TIME_STEPPERS = [
    dict(testcase_name='_' + s.__name__, time_stepper=s)
    for s in ALL_TIME_STEPPERS
]


class EquationsTest2D(test_util.TestCase):

  @parameterized.named_parameters(ALL_TIME_STEPPERS)
  def test_forced_turbulence(self, time_stepper):
    # make sure it runs for 100 steps without failing
    grid = grids.Grid((128, 128), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(42), grid, 7, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    viscosity = 1e-3
    dt = 1e-5

    step_fn = time_stepper(
        spectral_equations.NavierStokes2D(
            viscosity,
            grid,
            forcing_fn=spectral_forcings.kolmogorov_forcing_fn,
            drag=0.1), dt)

    trajectory_fn = cfd.funcutils.trajectory(step_fn, 100)
    _, trajectory = trajectory_fn(vorticity_hat0)
    self.assertTrue(jnp.all(~jnp.isnan(trajectory)))

  def test_viscosity(self):
    grid = grids.Grid((128, 128), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(42), grid, 7, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    norms = []
    for viscosity in [1e-3, 1e-1, 1]:
      dt = cfd.equations.stable_time_step(
          7, .5, viscosity, grid, implicit_diffusion=True)
      step_fn = time_stepping.crank_nicolson_rk4(
          spectral_equations.NavierStokes2D(
              viscosity,
              grid,
              forcing_fn=spectral_forcings.kolmogorov_forcing_fn,
              drag=0.1), dt)

      trajectory_fn = cfd.funcutils.trajectory(step_fn, 100)
      _, trajectory = trajectory_fn(vorticity_hat0)

      norms.append(jnp.linalg.norm(trajectory))

    # higher viscosity means that you get to zero faster.
    self.assertLess(norms[2], norms[1])
    self.assertLess(norms[1], norms[0])

  @parameterized.named_parameters(
      dict(
          testcase_name='_TaylorGreen_SemiImplicitNavierStokes',
          problem=cfd.validation_problems.TaylorGreen(
              shape=(1024, 1024), density=1., viscosity=1e-3),
          equation=spectral_equations.NavierStokes2D,
          time_stepper=time_stepping.crank_nicolson_rk4,
          max_courant_number=.5,
          time=.11,
          atol=1e-3),)
  def test_accuracy(self, problem, equation, time_stepper, max_courant_number,
                    time, atol):
    # This closely emulates a test in jax cfd:
    # https://source.corp.google.com/piper///depot/google3/third_party/py/jax_cfd/base/validation_test.py;l=113
    vorticity = finite_differences.curl_2d(problem.velocity(0.)).data

    dt = cfd.equations.stable_time_step(
        7,
        max_courant_number,
        problem.viscosity,
        problem.grid,
        implicit_diffusion=True)
    steps = int(jnp.ceil(time / dt))
    step_fn = time_stepper(
        equation(
            viscosity=problem.viscosity,
            grid=problem.grid,
            forcing_fn=None,
            drag=0), dt)

    _, vorticity_computed = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, steps), 1)(
            jnp.fft.rfftn(vorticity))

    vorticity_analytic = finite_differences.curl_2d(problem.velocity(time)).data

    self.assertAllClose(
        jnp.fft.irfftn(vorticity_computed[0]), vorticity_analytic, atol=atol)

  @parameterized.named_parameters(
      dict(
          testcase_name='_decaying_turbulence',
          viscosity=1e-2,
          cfl_safety_factor=.1,
          max_velocity=2.0,
          peak_wavenumber=4,
          seed=0,
          density=1.0,
          n_steps=500,
          grid_size=512,
          atol=0.062))
  def test_compare_to_finite_difference_method(self, viscosity,
                                               cfl_safety_factor, max_velocity,
                                               peak_wavenumber, seed, density,
                                               n_steps, grid_size, atol):
    """Compare spectral to jax_cfd's finite volume base method."""

    grid = cfd.grids.Grid((grid_size, grid_size),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    # Construct a random initial velocity.
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid, max_velocity)

    # Choose a time step.
    dt = cfd.equations.stable_time_step(max_velocity, cfl_safety_factor,
                                        viscosity, grid)

    # use `repeated` since we only compare the final state
    fvm_rollout_fn = jax.jit(
        cfd.funcutils.repeated(
            cfd.equations.semi_implicit_navier_stokes(
                density=density,
                viscosity=viscosity,
                dt=dt,
                grid=grid,
                forcing=None),
            steps=n_steps))

    final_state_fvm = cfd.finite_differences.curl_2d(fvm_rollout_fn(v0)).data

    spectral_rollout_fn = jax.jit(
        cfd.funcutils.repeated(
            time_stepping.crank_nicolson_rk4(
                spectral_equations.NavierStokes2D(viscosity, grid, 0), dt),
            steps=n_steps))

    final_state_spectral = jnp.fft.irfftn(
        spectral_rollout_fn(
            jnp.fft.rfftn(cfd.finite_differences.curl_2d(v0).data)))

    self.assertAllClose(final_state_fvm, final_state_spectral, atol=atol)

if __name__ == '__main__':
  absltest.main()
