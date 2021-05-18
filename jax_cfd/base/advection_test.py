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

"""Tests for jax_cfd.advection."""

import functools
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax_cfd.base import advection as adv
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np


def _gaussian_concentration(grid):
  offset = tuple(-int(jnp.ceil(s / 2.)) for s in grid.shape)
  return grids.AlignedArray(
      jnp.exp(-sum(jnp.square(m) * 30. for m in grid.mesh(offset=offset))),
      (0.5,) * len(grid.shape))


def _square_concentration(grid):
  select_square = lambda x: jnp.where(jnp.logical_and(x > 0.4, x < 0.6), 1., 0.)
  return grids.AlignedArray(
      jnp.array([select_square(m) for m in grid.mesh()]).prod(0),
      (0.5,) * len(grid.shape))


def _unit_velocity(grid, velocity_sign=1.):
  dim = len(grid.shape)
  offsets = (jnp.eye(dim) + jnp.ones([dim, dim])) / 2.
  return tuple(
      grids.AlignedArray(velocity_sign * jnp.ones(grid.shape) if ax == 0
                         else jnp.zeros(grid.shape), tuple(offset))
      for ax, offset in enumerate(offsets))


def _cos_velocity(grid):
  dim = len(grid.shape)
  offsets = (jnp.eye(dim) + jnp.ones([dim, dim])) / 2.
  mesh = grid.mesh()
  v = tuple(grids.AlignedArray(jnp.cos(mesh[i] * 2. * np.pi), tuple(offset))
            for i, offset in enumerate(offsets))
  return v


def _total_variation(array, grid, motion_axis):
  next_values = grid.shift(array, 1, motion_axis)
  variation = jnp.sum(jnp.abs(next_values.data - array.data))
  return variation


def _euler_step(advection_method):
  def step(c, v, grid, dt):
    return c + dt * advection_method(c, v, grid, dt)
  return step


class AdvectionTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='linear_1D',
           shape=(101,),
           method=_euler_step(adv.advect_linear),
           num_steps=1000,
           cfl_number=0.01,
           atol=5e-2),
      dict(testcase_name='linear_3D',
           shape=(101, 101, 101),
           method=_euler_step(adv.advect_linear),
           num_steps=1000,
           cfl_number=0.01,
           atol=5e-2),
      dict(testcase_name='upwind_1D',
           shape=(101,),
           method=_euler_step(adv.advect_upwind),
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
      dict(testcase_name='upwind_3D',
           shape=(101, 5, 5),
           method=_euler_step(adv.advect_upwind),
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
      dict(testcase_name='van_leer_1D',
           shape=(101,),
           method=_euler_step(adv.advect_van_leer),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='van_leer_1D_negative_v',
           shape=(101,),
           method=_euler_step(adv.advect_van_leer),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2,
           v_sign=-1.),
      dict(testcase_name='van_leer_3D',
           shape=(101, 5, 5),
           method=_euler_step(adv.advect_van_leer),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='van_leer_using_limiters_1D',
           shape=(101,),
           method=_euler_step(adv.advect_van_leer_using_limiters),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='van_leer_using_limiters_3D',
           shape=(101, 5, 5),
           method=_euler_step(adv.advect_van_leer_using_limiters),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='semilagrangian_1D',
           shape=(101,),
           method=adv.advect_step_semilagrangian,
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
      dict(testcase_name='semilagrangian_3D',
           shape=(101, 5, 5),
           method=adv.advect_step_semilagrangian,
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
  )
  def test_advection_analytical(
      self, shape, method, num_steps, cfl_number, atol, v_sign=1):
    """Tests advection of a Gaussian concentration on a periodic grid."""
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)

    v = _unit_velocity(grid, v_sign)
    c = _gaussian_concentration(grid)

    dt = cfl_number * min(step)
    advect = functools.partial(method, v=v, grid=grid, dt=dt)
    evolve = jax.jit(funcutils.repeated(advect, num_steps))
    ct = evolve(c)

    expected_shift = int(round(-cfl_number * num_steps * v_sign))
    expected = grid.shift(c, expected_shift, axis=0).data
    self.assertAllClose(expected, ct.data, atol=atol)

  @parameterized.named_parameters(
      dict(testcase_name='linear_1D',
           shape=(101,),
           method=_euler_step(adv.advect_linear),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='linear_3D',
           shape=(101, 101, 101),
           method=_euler_step(adv.advect_linear),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='upwind_1D',
           shape=(101,),
           method=_euler_step(adv.advect_upwind),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='upwind_3D',
           shape=(101, 101, 101),
           method=_euler_step(adv.advect_upwind),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_1D',
           shape=(101,),
           method=_euler_step(adv.advect_van_leer),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_1D_negative_v',
           shape=(101,),
           method=_euler_step(adv.advect_van_leer),
           atol=1e-2,
           rtol=1e-2,
           v_sign=-1.),
      dict(testcase_name='van_leer_3D',
           shape=(101, 101, 101),
           method=_euler_step(adv.advect_van_leer),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_using_limiters_1D',
           shape=(101,),
           method=_euler_step(adv.advect_van_leer_using_limiters),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_using_limiters_3D',
           shape=(101, 101, 101),
           method=_euler_step(adv.advect_van_leer_using_limiters),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='semilagrangian_1D',
           shape=(101,),
           method=adv.advect_step_semilagrangian,
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='semilagrangian_3D',
           shape=(101, 101, 101),
           method=adv.advect_step_semilagrangian,
           atol=1e-2,
           rtol=1e-2),
  )
  def test_advection_gradients(
      self, shape, method, atol, rtol, cfl_number=0.5, v_sign=1,
  ):
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)

    v = _unit_velocity(grid, v_sign)
    c_init = _gaussian_concentration(grid)

    dt = cfl_number * min(step)
    advect = jax.remat(functools.partial(method, v=v, grid=grid, dt=dt))
    evolve = jax.jit(funcutils.repeated(advect, steps=10))

    def objective(c_init):
      return 0.5 * jnp.sum(evolve(c_init).data ** 2)

    gradient = jax.jit(jax.grad(objective))(c_init)
    self.assertAllClose(c_init, gradient, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      dict(testcase_name='_linear_1D',
           shape=(101,),
           advection_method=adv.advect_linear,
           convection_method=adv.convect_linear),
      dict(testcase_name='_linear_3D',
           shape=(101, 101, 101),
           advection_method=adv.advect_linear,
           convection_method=adv.convect_linear)
  )
  def test_convection_vs_advection(
      self, shape, advection_method, convection_method,
  ):
    """Exercises self-advection, check equality with advection on components."""
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)
    v = _cos_velocity(grid)
    self_advected = convection_method(v, grid)
    for u, du in zip(v, self_advected):
      advected_component = advection_method(u, v, grid)

      self.assertAllClose(advected_component, du)

  @parameterized.named_parameters(
      dict(testcase_name='upwind_1D',
           shape=(101,),
           method=_euler_step(adv.advect_upwind)),
      dict(testcase_name='van_leer_1D',
           shape=(101,),
           method=_euler_step(adv.advect_van_leer)),
      dict(testcase_name='semilagrangian_1D',
           shape=(101,),
           method=adv.advect_step_semilagrangian),
  )
  def test_tvd_property(self, shape, method):
    atol = 1e-6
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)

    v = _unit_velocity(grid)
    c = _square_concentration(grid)

    dt = min(step) / 100.
    num_steps = 300
    ct = c

    advect = jax.jit(functools.partial(method, v=v, grid=grid, dt=dt))

    initial_total_variation = _total_variation(c, grid, 0) + atol
    for _ in range(num_steps):
      ct = advect(ct)
      current_total_variation = _total_variation(ct, grid, 0)
      self.assertLessEqual(current_total_variation, initial_total_variation)

  @parameterized.named_parameters(
      dict(testcase_name='van_leers_equivalence_1d',
           shape=(101,), velocity_sign=1.),
      dict(testcase_name='van_leers_equivalence_3d',
           shape=(101, 101, 101), velocity_sign=1.),
      dict(testcase_name='van_leers_equivalence_1d_negative_v',
           shape=(101,), velocity_sign=-1.),
      dict(testcase_name='van_leers_equivalence_3d_negative_v',
           shape=(101, 101, 101), velocity_sign=-1.),
  )
  def test_van_leer_same_as_van_leer_using_limiters(self, shape, velocity_sign):
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)

    v = _unit_velocity(grid, velocity_sign)
    c = _square_concentration(grid)

    dt = min(step) / 100.
    num_steps = 100
    advect_vl = jax.jit(
        functools.partial(_euler_step(adv.advect_van_leer),
                          v=v, grid=grid, dt=dt))
    advect_vl_using_limiter = jax.jit(
        functools.partial(_euler_step(adv.advect_van_leer_using_limiters),
                          v=v, grid=grid, dt=dt))
    c_vl = c
    c_vl_using_limiter = c
    for _ in range(num_steps):
      c_vl = advect_vl(c_vl)
      c_vl_using_limiter = advect_vl_using_limiter(c_vl_using_limiter)
    self.assertAllClose(c_vl, c_vl_using_limiter, atol=1e-5)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
