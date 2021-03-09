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

"""Validation tests for JAX-CFD."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
from jax_cfd.base import advection
from jax_cfd.base import diffusion
from jax_cfd.base import equations
from jax_cfd.base import funcutils
from jax_cfd.base import test_util
from jax_cfd.base import validation_problems


class ValidationTests(test_util.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_TaylorGreen_SemiImplicitNavierStokes',
          problem=validation_problems.TaylorGreen(
              shape=(1024, 1024), density=1., viscosity=1e-3),
          solver=functools.partial(
              equations.semi_implicit_navier_stokes,
              convect=advection.convect_linear),
          implicit_diffusion=False,
          max_courant_number=.1,
          time=10.,
          atol=1e-5),
      dict(
          testcase_name='_TaylorGreen_ImplicitDiffusionNavierStokes_matmul',
          problem=validation_problems.TaylorGreen(
              shape=(1024, 1024), density=1., viscosity=1e-3),
          solver=functools.partial(
              equations.implicit_diffusion_navier_stokes,
              convect=advection.convect_linear,
              diffusion_solve=functools.partial(
                  diffusion.solve_fast_diag, implementation='matmul'),
          ),
          implicit_diffusion=True,
          max_courant_number=.1,
          time=10.,
          atol=3e-5),
      dict(
          testcase_name='_TaylorGreen_ImplicitDiffusionNavierStokes_fft',
          problem=validation_problems.TaylorGreen(
              shape=(1024, 1024), density=1., viscosity=1e-3),
          solver=functools.partial(
              equations.implicit_diffusion_navier_stokes,
              convect=advection.convect_linear,
              diffusion_solve=functools.partial(
                  diffusion.solve_fast_diag, implementation='fft'),
          ),
          implicit_diffusion=True,
          max_courant_number=.1,
          time=10.,
          atol=4e-5),
      dict(
          testcase_name='_TaylorGreen_ImplicitDiffusionNavierStokes_rfft',
          problem=validation_problems.TaylorGreen(
              shape=(1024, 1024), density=1., viscosity=1e-3),
          solver=functools.partial(
              equations.implicit_diffusion_navier_stokes,
              convect=advection.convect_linear,
              diffusion_solve=functools.partial(
                  diffusion.solve_fast_diag, implementation='rfft'),
          ),
          implicit_diffusion=True,
          max_courant_number=.1,
          time=10.,
          atol=4e-5),
      dict(
          testcase_name='_TaylorGreen_ImplicitDiffusionNavierStokes_cg',
          problem=validation_problems.TaylorGreen(
              shape=(1024, 1024), density=1., viscosity=1e-3),
          solver=functools.partial(
              equations.implicit_diffusion_navier_stokes,
              convect=advection.convect_linear,
              diffusion_solve=functools.partial(
                  diffusion.solve_cg, atol=1e-6, maxiter=512)),
          implicit_diffusion=True,
          max_courant_number=.1,
          time=10.,
          atol=3e-5),
      dict(
          testcase_name='_TaylorGreen_ImplicitDiffusionNavierStokes_viscous',
          problem=validation_problems.TaylorGreen(
              shape=(1024, 1024), density=1., viscosity=0.5),
          solver=functools.partial(
              equations.implicit_diffusion_navier_stokes,
              convect=advection.convect_linear,
          ),
          implicit_diffusion=True,
          max_courant_number=.5,
          time=1.0,
          atol=6e-4),
  )
  def test_accuracy(self, problem, solver, implicit_diffusion,
                    max_courant_number, time, atol):
    """Test the accuracy of `solver` on the given `problem`.

    Args:
      problem: an instance of `validation_problems.Problem`.
      solver: a callable that takes `density`, `viscosity`, `dt`, `grid`, and
        `steps`. It returns a callable that takes `velocity`,
        `pressure_correction` and `force` and returns updated versions of these
        values at the next time step.
      implicit_diffusion: whether or not the solver models diffusion implicitly.
      max_courant_number: a float used to choose the size of the time step `dt`
        according to the Courant-Friedrichs-Lewy condition. See
        https://en.wikipedia.org/wiki/Courant-Friedrichs-Lewy_condition.
      time: the amount of time to run the simulation for.
      atol: absolute error tolerance per entry.
    """
    v = problem.velocity(0.)
    dt = equations.dynamic_time_step(
        v, max_courant_number, problem.viscosity, problem.grid,
        implicit_diffusion)
    steps = int(jnp.ceil(time / dt))
    navier_stokes = solver(density=problem.density,
                           viscosity=problem.viscosity,
                           dt=dt,
                           grid=problem.grid)
    v_computed = funcutils.repeated(navier_stokes, steps)(v)
    v_analytic = problem.velocity(time)
    for u_c, u_a in zip(v_computed, v_analytic):
      self.assertAllClose(u_c, u_a, atol=atol)


if __name__ == '__main__':
  absltest.main()
