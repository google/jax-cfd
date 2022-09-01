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
from jax_cfd.base import advection
from jax_cfd.base import boundaries
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np


def _gaussian_concentration(grid):
  offset = tuple(-int(jnp.ceil(s / 2.)) for s in grid.shape)
  return grids.GridArray(
      jnp.exp(-sum(jnp.square(m) * 30. for m in grid.mesh(offset=offset))),
      (0.5,) * len(grid.shape), grid)


def _square_concentration(grid):
  select_square = lambda x: jnp.where(jnp.logical_and(x > 0.4, x < 0.6), 1., 0.)
  return grids.GridArray(
      jnp.array([select_square(m) for m in grid.mesh()]).prod(0),
      (0.5,) * len(grid.shape), grid)


def _unit_velocity(grid, velocity_sign=1.):
  ndim = grid.ndim
  offsets = (np.eye(ndim) + np.ones([ndim, ndim])) / 2.
  return tuple(
      grids.GridArray(velocity_sign * jnp.ones(grid.shape) if ax == 0
                      else jnp.zeros(grid.shape), tuple(offset), grid)
      for ax, offset in enumerate(offsets))


def _cos_velocity(grid):
  ndim = grid.ndim
  offsets = (np.eye(ndim) + np.ones([ndim, ndim])) / 2.
  mesh = grid.mesh()
  v = tuple(grids.GridArray(jnp.cos(mesh[i] * 2. * np.pi), tuple(offset), grid)
            for i, offset in enumerate(offsets))
  return v


def _velocity_implicit(grid, offset, u, t):
  """Returns solution of a Burgers equation at time `t`."""
  x = grid.mesh((offset,))[0]
  return grids.GridArray(jnp.sin(x - u * t), (offset,), grid)


def _total_variation(array, motion_axis):
  next_values = array.shift(1, motion_axis)
  variation = jnp.sum(jnp.abs(next_values.data - array.data))
  return variation


def _euler_step(advection_method):
  def step(c, v, dt):
    c_new = c.array + dt * advection_method(c, v, dt)
    return c.bc.impose_bc(c_new)
  return step


class AdvectionTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='linear_1D',
           shape=(101,),
           method=_euler_step(advection.advect_linear),
           num_steps=1000,
           cfl_number=0.01,
           atol=5e-2),
      dict(testcase_name='linear_3D',
           shape=(101, 101, 101),
           method=_euler_step(advection.advect_linear),
           num_steps=1000,
           cfl_number=0.01,
           atol=5e-2),
      dict(testcase_name='upwind_1D',
           shape=(101,),
           method=_euler_step(advection.advect_upwind),
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
      dict(testcase_name='upwind_3D',
           shape=(101, 5, 5),
           method=_euler_step(advection.advect_upwind),
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
      dict(testcase_name='van_leer_1D',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='van_leer_1D_negative_v',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2,
           v_sign=-1.),
      dict(testcase_name='van_leer_3D',
           shape=(101, 5, 5),
           method=_euler_step(advection.advect_van_leer),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='van_leer_using_limiters_1D',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer_using_limiters),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='van_leer_using_limiters_3D',
           shape=(101, 5, 5),
           method=_euler_step(advection.advect_van_leer_using_limiters),
           num_steps=100,
           cfl_number=0.5,
           atol=2e-2),
      dict(testcase_name='semilagrangian_1D',
           shape=(101,),
           method=advection.advect_step_semilagrangian,
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
      dict(testcase_name='semilagrangian_3D',
           shape=(101, 5, 5),
           method=advection.advect_step_semilagrangian,
           num_steps=100,
           cfl_number=0.5,
           atol=7e-2),
  )
  def test_advection_analytical(
      self, shape, method, num_steps, cfl_number, atol, v_sign=1):
    """Tests advection of a Gaussian concentration on a periodic grid."""
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = tuple(grids.GridVariable(u, bc) for u in _unit_velocity(grid, v_sign))
    c = grids.GridVariable(_gaussian_concentration(grid), bc)
    dt = cfl_number * min(step)
    advect = functools.partial(method, v=v, dt=dt)
    evolve = jax.jit(funcutils.repeated(advect, num_steps))
    ct = evolve(c)

    expected_shift = int(round(-cfl_number * num_steps * v_sign))
    expected = c.shift(expected_shift, axis=0).data
    self.assertAllClose(expected, ct.data, atol=atol)

  @parameterized.named_parameters(
      dict(
          testcase_name='dirichlet_1d_100', shape=(100,), atol=0.001,
          offset=.5),
      dict(
          testcase_name='dirichlet_1d_200',
          shape=(200,),
          atol=0.00025,
          offset=.5),
      dict(
          testcase_name='dirichlet_1d_400',
          shape=(400,),
          atol=0.00007,
          offset=.5),
      dict(
          testcase_name='dirichlet_1d_100_cell_edge_0',
          shape=(100,),
          atol=0.002,
          offset=0.),
      dict(
          testcase_name='dirichlet_1d_200_cell_edge_0',
          shape=(200,),
          atol=0.0005,
          offset=0.),
      dict(
          testcase_name='dirichlet_1d_400_cell_edge_0',
          shape=(400,),
          atol=0.000125,
          offset=0.),
      dict(
          testcase_name='dirichlet_1d_100_cell_edge_1',
          shape=(100,),
          atol=0.002,
          offset=1.),
      dict(
          testcase_name='dirichlet_1d_200_cell_edge_1',
          shape=(200,),
          atol=0.0005,
          offset=1.),
      dict(
          testcase_name='dirichlet_1d_400_cell_edge_1',
          shape=(400,),
          atol=0.000125,
          offset=1.),
  )
  def test_burgers_analytical_dirichlet_convergence(
      self,
      shape,
      atol,
      offset,
  ):
    num_steps = 1000
    cfl_number = 0.01
    step = 2 * jnp.pi / 1000
    grid = grids.Grid(shape, domain=([0., 2 * jnp.pi],))
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
    v = (bc.impose_bc(_velocity_implicit(grid, offset, 0, 0)),)
    dt = cfl_number * step

    def _advect(v):
      dv_dt = advection.advect_van_leer(c=v[0], v=v, dt=dt) / 2
      return (bc.impose_bc(v[0].array + dt * dv_dt),)

    evolve = jax.jit(funcutils.repeated(_advect, num_steps))
    ct = evolve(v)

    expected = bc.impose_bc(
        _velocity_implicit(grid, offset, ct[0].data, dt * num_steps)).data
    self.assertAllClose(expected, ct[0].data, atol=atol)

  @parameterized.named_parameters(
      dict(testcase_name='linear_1D',
           shape=(101,),
           method=_euler_step(advection.advect_linear),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='linear_3D',
           shape=(101, 101, 101),
           method=_euler_step(advection.advect_linear),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='upwind_1D',
           shape=(101,),
           method=_euler_step(advection.advect_upwind),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='upwind_3D',
           shape=(101, 101, 101),
           method=_euler_step(advection.advect_upwind),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_1D',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_1D_negative_v',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer),
           atol=1e-2,
           rtol=1e-2,
           v_sign=-1.),
      dict(testcase_name='van_leer_3D',
           shape=(101, 101, 101),
           method=_euler_step(advection.advect_van_leer),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_using_limiters_1D',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer_using_limiters),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_using_limiters_3D',
           shape=(101, 101, 101),
           method=_euler_step(advection.advect_van_leer_using_limiters),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='semilagrangian_1D',
           shape=(101,),
           method=advection.advect_step_semilagrangian,
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='semilagrangian_3D',
           shape=(101, 101, 101),
           method=advection.advect_step_semilagrangian,
           atol=1e-2,
           rtol=1e-2),
  )
  def test_advection_gradients(
      self, shape, method, atol, rtol, cfl_number=0.5, v_sign=1,
  ):
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = tuple(grids.GridVariable(u, bc) for u in _unit_velocity(grid, v_sign))
    c = grids.GridVariable(_gaussian_concentration(grid), bc)
    dt = cfl_number * min(step)
    advect = jax.remat(functools.partial(method, v=v, dt=dt))
    evolve = jax.jit(funcutils.repeated(advect, steps=10))

    def objective(c):
      return 0.5 * jnp.sum(evolve(c).data ** 2)

    gradient = jax.jit(jax.grad(objective))(c)
    self.assertAllClose(c, gradient, atol=atol, rtol=rtol)


  @parameterized.named_parameters(
      dict(testcase_name='van_leer_1D',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer),
           atol=1e-2,
           rtol=1e-2),
      dict(testcase_name='van_leer_1D_negative_v',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer),
           atol=1e-2,
           rtol=1e-2,
           v_sign=-1.),
      dict(testcase_name='van_leer_3D',
           shape=(101, 101, 101),
           method=_euler_step(advection.advect_van_leer),
           atol=1e-2,
           rtol=1e-2),
  )
  def test_advection_gradients_division_by_zero(
      self, shape, method, atol, rtol, cfl_number=0.5, v_sign=1,
  ):
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = tuple(grids.GridVariable(u, bc) for u in _unit_velocity(grid, v_sign))
    c = grids.GridVariable(_unit_velocity(grid)[0], bc)
    dt = cfl_number * min(step)
    advect = jax.remat(functools.partial(method, v=v, dt=dt))
    evolve = jax.jit(funcutils.repeated(advect, steps=10))

    def objective(c):
      return 0.5 * jnp.sum(evolve(c).data ** 2)

    gradient = jax.jit(jax.grad(objective))(c)
    self.assertAllClose(c, gradient, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      dict(testcase_name='_linear_1D',
           shape=(101,),
           advection_method=advection.advect_linear,
           convection_method=advection.convect_linear),
      dict(testcase_name='_linear_3D',
           shape=(101, 101, 101),
           advection_method=advection.advect_linear,
           convection_method=advection.convect_linear)
  )
  def test_convection_vs_advection(
      self, shape, advection_method, convection_method,
  ):
    """Exercises self-advection, check equality with advection on components."""
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = tuple(grids.GridVariable(u, bc) for u in _cos_velocity(grid))
    self_advected = convection_method(v)
    for u, du in zip(v, self_advected):
      advected_component = advection_method(u, v)

      self.assertAllClose(advected_component, du)

  @parameterized.named_parameters(
      dict(testcase_name='upwind_1D',
           shape=(101,),
           method=_euler_step(advection.advect_upwind)),
      dict(testcase_name='van_leer_1D',
           shape=(101,),
           method=_euler_step(advection.advect_van_leer)),
      dict(testcase_name='semilagrangian_1D',
           shape=(101,),
           method=advection.advect_step_semilagrangian),
  )
  def test_tvd_property(self, shape, method):
    atol = 1e-6
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = tuple(grids.GridVariable(u, bc) for u in _unit_velocity(grid))
    c = grids.GridVariable(_square_concentration(grid), bc)
    dt = min(step) / 100.
    num_steps = 300
    ct = c

    advect = jax.jit(functools.partial(method, v=v, dt=dt))

    initial_total_variation = _total_variation(c, 0) + atol
    for _ in range(num_steps):
      ct = advect(ct)
      current_total_variation = _total_variation(ct, 0)
      self.assertLessEqual(current_total_variation, initial_total_variation)

  @parameterized.named_parameters(
      dict(
          testcase_name='van_leer_1D',
          shape=(101,),
          method=_euler_step(advection.advect_van_leer)),
  )
  def test_mass_conservation(self, shape, method):
    offset = 0.5
    cfl_number = 0.1
    dt = cfl_number / shape[0]
    num_steps = 1000

    grid = grids.Grid(shape, domain=([-1., 1.],))
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
    c_bc = boundaries.dirichlet_boundary_conditions(grid.ndim, ((-1., 1.),))

    def u(grid, offset):
      x = grid.mesh((offset,))[0]
      return grids.GridArray(-jnp.sin(jnp.pi * x), (offset,), grid)

    def c0(grid, offset):
      x = grid.mesh((offset,))[0]
      return grids.GridArray(x, (offset,), grid)

    v = (bc.impose_bc(u(grid, 1.)),)
    c = c_bc.impose_bc(c0(grid, offset))

    ct = c

    advect = jax.jit(functools.partial(method, v=v, dt=dt))

    initial_mass = np.sum(c.data)
    for _ in range(num_steps):
      ct = advect(ct)
      current_total_mass = np.sum(ct.data)
      self.assertAllClose(current_total_mass, initial_mass, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='van_leers_equivalence_1d',
           shape=(101,), v_sign=1.),
      dict(testcase_name='van_leers_equivalence_3d',
           shape=(101, 101, 101), v_sign=1.),
      dict(testcase_name='van_leers_equivalence_1d_negative_v',
           shape=(101,), v_sign=-1.),
      dict(testcase_name='van_leers_equivalence_3d_negative_v',
           shape=(101, 101, 101), v_sign=-1.),
  )
  def test_van_leer_same_as_van_leer_using_limiters(self, shape, v_sign):
    step = tuple(1. / s for s in shape)
    grid = grids.Grid(shape, step)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = tuple(grids.GridVariable(u, bc) for u in _unit_velocity(grid, v_sign))
    c = grids.GridVariable(_gaussian_concentration(grid), bc)
    dt = min(step) / 100.
    num_steps = 100
    advect_vl = jax.jit(
        functools.partial(_euler_step(advection.advect_van_leer), v=v, dt=dt))
    advect_vl_using_limiter = jax.jit(
        functools.partial(
            _euler_step(advection.advect_van_leer_using_limiters), v=v, dt=dt))
    c_vl = c
    c_vl_using_limiter = c
    for _ in range(num_steps):
      c_vl = advect_vl(c_vl)
      c_vl_using_limiter = advect_vl_using_limiter(c_vl_using_limiter)
    self.assertAllClose(c_vl, c_vl_using_limiter, atol=1e-5)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
