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
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import test_util
from jax_cfd.collocated import advection
import numpy as np


def _cos_velocity(grid):
  offset = grid.cell_center
  mesh = grid.mesh(offset)
  mesh_size = jnp.array(grid.shape) * jnp.array(grid.step)
  v = tuple(grids.GridArray(jnp.cos(2. * np.pi * x / s), offset, grid)
            for x, s in zip(mesh, mesh_size))
  return v


def _euler_step(advection_method):

  def step(c, v, dt):
    c_new = c.array + dt * advection_method(c, v, dt)
    return c.bc.impose_bc(c_new)

  return step


class AdvectionTest(test_util.TestCase):

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
      dict(
          testcase_name='dichlet_advect',
          shape=(101,),
          method=_euler_step(advection.advect_linear)),)
  def test_mass_conservation_dirichlet(self, shape, method):
    cfl_number = 0.1
    dt = cfl_number / shape[0]
    num_steps = 1000

    grid = grids.Grid(shape, domain=([-1., 1.],))
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
    c_bc = boundaries.dirichlet_boundary_conditions(grid.ndim, ((-1., 1.),))

    def u(grid):
      x = grid.mesh((0.5,))[0]
      return grids.GridArray(-jnp.sin(jnp.pi * x), (0.5,), grid)

    def c0(grid):
      x = grid.mesh((0.5,))[0]
      return grids.GridArray(x, (0.5,), grid)

    v = (bc.impose_bc(u(grid)),)
    c = c_bc.impose_bc(c0(grid))

    ct = c

    advect = jax.jit(functools.partial(method, v=v, dt=dt))

    initial_mass = np.sum(c.data)
    for _ in range(num_steps):
      ct = advect(ct)
      current_total_mass = np.sum(ct.data)
      self.assertAllClose(current_total_mass, initial_mass, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='linear_1d_neumann',
          shape=(1000,),
          method=advection.advect_linear),)
  def test_neumann_bc_one_step(self, shape, method):
    grid = grids.Grid(shape, domain=([-1., 1.],))
    bc = boundaries.neumann_boundary_conditions(grid.ndim)
    c_bc = boundaries.neumann_boundary_conditions(grid.ndim)

    def u(grid):
      x = grid.mesh((0.5,))[0]
      return grids.GridArray(jnp.cos(jnp.pi * x), (0.5,), grid)

    def c0(grid):
      x = grid.mesh((0.5,))[0]
      return grids.GridArray(jnp.cos(jnp.pi * x), (0.5,), grid)

    def dcdt(grid):
      x = grid.mesh((0.5,))[0]
      return grids.GridArray(jnp.pi * jnp.sin(2 * jnp.pi * x), (0.5,), grid)

    v = (bc.impose_bc(u(grid)),)
    c = c_bc.impose_bc(c0(grid))

    advect = jax.jit(functools.partial(method, v=v))

    ct = advect(c)
    self.assertAllClose(ct, dcdt(grid), atol=1e-4)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
