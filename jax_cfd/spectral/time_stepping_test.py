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
from jax import tree_util
from jax.config import config
import jax.numpy as jnp
from jax_cfd.base import funcutils
from jax_cfd.spectral import time_stepping
import numpy as np


def harmonic_oscillator(x0, t):
  theta = jnp.arctan(x0[0] / x0[1])
  r = jnp.linalg.norm(x0, ord=2, axis=0)
  return r * jnp.stack([jnp.sin(t + theta), jnp.cos(t + theta)])


class CustomODE(time_stepping.ImplicitExplicitODE):

  def __init__(self, explicit_terms, implicit_terms, implicit_solve):
    self.explicit_terms = explicit_terms
    self.implicit_terms = implicit_terms
    self.implicit_solve = implicit_solve


ALL_TEST_PROBLEMS = [
    # x(t) = np.ones(10)
    dict(testcase_name='_zero_derivative',
         explicit_terms=lambda x: 0 * x,
         implicit_terms=lambda x: 0 * x,
         implicit_solve=lambda x, eta: x,
         dt=1e-2,
         inner_steps=10,
         outer_steps=5,
         initial_state=np.ones(10),
         closed_form=lambda x0, t: x0,
         tolerances=[1e-12] * 5),
    # x(t) = 5 * t * np.ones(3)
    dict(testcase_name='_constant_derivative',
         explicit_terms=lambda x: 5 * jnp.ones_like(x),
         implicit_terms=lambda x: 0 * x,
         implicit_solve=lambda x, eta: x,
         dt=1e-2,
         inner_steps=10,
         outer_steps=5,
         initial_state=np.ones(3),
         closed_form=lambda x0, t: x0 + 5 * t,
         tolerances=[1e-12] * 5),
    # x(t) = np.arange(3) * np.exp(t)
    # Uses explicit terms only.
    dict(testcase_name='_linear_derivative_explicit',
         explicit_terms=lambda x: x,
         implicit_terms=lambda x: 0 * x,
         implicit_solve=lambda x, eta: x,
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.arange(3.0),
         closed_form=lambda x0, t: np.arange(3) * jnp.exp(t),
         tolerances=[5e-2, 1e-4, 1e-6, 1e-9, 1e-6]),
    # x(t) = np.arange(3) * np.exp(t)
    # Uses implicit terms only.
    dict(testcase_name='_linear_derivative_implicit',
         explicit_terms=lambda x: 0 * x,
         implicit_terms=lambda x: x,
         implicit_solve=lambda x, eta: x / (1 - eta),
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.arange(3.0),
         closed_form=lambda x0, t: np.arange(3) * jnp.exp(t),
         tolerances=[5e-2, 5e-5, 1e-5, 1e-5, 3e-5]),
    # x(t) = np.arange(3) * np.exp(t)
    # Splits the equation into an implicit and explicit term.
    dict(testcase_name='_linear_derivative_semi_implicit',
         explicit_terms=lambda x: x / 2,
         implicit_terms=lambda x: x / 2,
         implicit_solve=lambda x, eta: x / (1 - eta / 2),
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.arange(3) * np.exp(0),
         closed_form=lambda x0, t: np.arange(3.0) * jnp.exp(t),
         tolerances=[1e-4, 2e-5, 2e-6, 1e-6, 2e-5]),
    dict(testcase_name='_harmonic_oscillator_explicit',
         explicit_terms=lambda x: jnp.stack([x[1], -x[0]]),
         implicit_terms=jnp.zeros_like,
         implicit_solve=lambda x, eta: x,
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.ones(2),
         closed_form=harmonic_oscillator,
         tolerances=[1e-2, 3e-5, 6e-8, 5e-11, 6e-8]),
    dict(testcase_name='_harmonic_oscillator_implicit',
         explicit_terms=jnp.zeros_like,
         implicit_terms=lambda x: jnp.stack([x[1], -x[0]]),
         implicit_solve=lambda x, eta: jnp.stack(  # pylint: disable=g-long-lambda
             [x[0] + eta * x[1], x[1] - eta * x[0]]) / (1 + eta ** 2),
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.ones(2),
         closed_form=harmonic_oscillator,
         tolerances=[1e-2, 2e-5, 2e-6, 1e-6, 6e-6]),
]


ALL_TIME_STEPPERS = [
    time_stepping.backward_forward_euler,
    time_stepping.crank_nicolson_rk2,
    time_stepping.crank_nicolson_rk3,
    time_stepping.crank_nicolson_rk4,
    time_stepping.imex_rk_sil3,
]


class TimeSteppingTest(parameterized.TestCase):

  @parameterized.named_parameters(ALL_TEST_PROBLEMS)
  def test_implicit_solve(
      self,
      explicit_terms,
      implicit_terms,
      implicit_solve,
      dt,
      inner_steps,
      outer_steps,
      initial_state,
      closed_form,
      tolerances,
  ):
    """Tests that time integration is accurate for a range of test cases."""
    del dt, explicit_terms, inner_steps, outer_steps, closed_form  # unused
    del tolerances  # unused

    # Verifies that `implicit_solve` solves (y - eta * F(y)) = x
    # This does not test the integrator, but rather verifies that the test
    # case is valid.
    eta = 0.3
    solved_state = implicit_solve(initial_state, eta)
    reconstructed_state = solved_state - eta * implicit_terms(solved_state)
    np.testing.assert_allclose(reconstructed_state, initial_state)

  @parameterized.named_parameters(ALL_TEST_PROBLEMS)
  def test_integration(
      self,
      explicit_terms,
      implicit_terms,
      implicit_solve,
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
        equation = CustomODE(explicit_terms, implicit_terms, implicit_solve)
        semi_implicit_step = time_stepper(equation, dt)
        integrator = funcutils.trajectory(
            funcutils.repeated(semi_implicit_step, inner_steps), outer_steps)
        _, actual = integrator(initial_state)
        np.testing.assert_allclose(expected, actual, atol=atol, rtol=0)

  def test_pytree_state(self):
    equation = CustomODE(
        explicit_terms=lambda x: tree_util.tree_map(jnp.zeros_like, x),
        implicit_terms=lambda x: tree_util.tree_map(jnp.zeros_like, x),
        implicit_solve=lambda x, eta: x,
    )
    u0 = {'x': 1.0, 'y': 1.0}
    for time_stepper in ALL_TIME_STEPPERS:
      with self.subTest(time_stepper.__name__):
        u1 = time_stepper(equation, 1.0)(u0)
        self.assertEqual(u0, u1)


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
