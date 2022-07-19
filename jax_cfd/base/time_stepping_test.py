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

"""Tests for time_stepping."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.config import config
import jax.numpy as jnp
from jax_cfd.base import funcutils
from jax_cfd.base import time_stepping
import numpy as np


def harmonic_oscillator(x0, t):
  theta = jnp.arctan(x0[0] / x0[1])
  r = jnp.linalg.norm(x0, ord=2, axis=0)
  return r * jnp.stack([jnp.sin(t + theta), jnp.cos(t + theta)])


ALL_TEST_PROBLEMS = [
    dict(testcase_name='_harmonic_oscillator_explicit',
         explicit_terms=lambda x: jnp.stack([x[1], -x[0]]),
         pressure_projection=lambda x: x,
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.ones(2),
         closed_form=harmonic_oscillator,
         tolerances=[1e-2, 3e-5, 3e-5, 4e-7]),
]


ALL_TIME_STEPPERS = [
    time_stepping.forward_euler,
    time_stepping.midpoint_rk2,
    time_stepping.heun_rk2,
    time_stepping.classic_rk4,
]


class TimeSteppingTest(parameterized.TestCase):

  @parameterized.named_parameters(ALL_TEST_PROBLEMS)
  def test_integration(
      self,
      explicit_terms,
      pressure_projection,
      dt,
      inner_steps,
      outer_steps,
      initial_state,
      closed_form,
      tolerances,
  ):
    # Compute closed-form solution.
    time = dt * inner_steps * (1 + np.arange(outer_steps))
    expected = jax.vmap(closed_form, in_axes=(None, 0))(
        initial_state, time)

    # Compute trajectory using time-stepper.
    for atol, time_stepper in zip(tolerances, ALL_TIME_STEPPERS):
      with self.subTest(time_stepper.__name__):
        equation = time_stepping.ExplicitNavierStokesODE(
            explicit_terms, pressure_projection)
        step_fn = time_stepper(equation, dt)
        integrator = funcutils.trajectory(
            funcutils.repeated(step_fn, inner_steps), outer_steps)
        _, actual = integrator(initial_state)
        np.testing.assert_allclose(expected, actual, atol=atol, rtol=0)


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
