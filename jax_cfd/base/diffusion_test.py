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

"""Tests for jax_cfd.diffusion."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import diffusion
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import test_util
import  numpy as np


class DiffusionTest(test_util.TestCase):
  """Some simple sanity tests for diffusion on constant fields."""

  def diffusion_setup(self, bc, offset):
    nu = 1.0
    dt = 0.5
    rs = np.random.RandomState(0)
    b = rs.randn(4, 4).astype(np.float32)
    grid = grids.Grid((4, 4), domain=((0, 4), (0, 4)))  # has step = 1.0
    b = bc.impose_bc(grids.GridArray(b, offset, grid))
    x = diffusion.solve_fast_diag((b,), nu, dt)[0]
    # laplacian is defined only on the interior
    x_interior = grids.GridVariable(bc.trim_boundary(x.array), bc)
    x = x_interior.array - nu * dt * fd.laplacian(x_interior)
    return x, b

  def test_explicit_diffusion(self):
    nu = 1.
    shape = (101, 101, 101)
    offset = (0.5, 0.5, 0.5)
    step = (1., 1., 1.)
    grid = grids.Grid(shape, step)

    c = grids.GridVariable(
        array=grids.GridArray(jnp.ones(shape), offset, grid),
        bc=boundaries.periodic_boundary_conditions(grid.ndim))
    diffused = diffusion.diffuse(c, nu)
    expected = grids.GridArray(jnp.zeros_like(diffused.data), offset, grid)
    self.assertAllClose(expected, diffused)

  @parameterized.parameters(
      dict(solve=diffusion.solve_cg, atol=1e-6),
      dict(solve=diffusion.solve_fast_diag, atol=1e-6),
  )
  def test_implicit_diffusion(self, solve, atol):
    nu = 1.
    dt = 0.1
    shape = (100, 100)
    grid = grids.Grid(shape, step=1)
    periodic_bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = (
        grids.GridVariable(
            grids.GridArray(jnp.ones(shape), (1, 0.5), grid), periodic_bc),
        grids.GridVariable(
            grids.GridArray(jnp.ones(shape), (0.5, 1), grid), periodic_bc),
    )
    actual = solve(v, nu, dt)
    expected = v
    self.assertAllClose(expected[0], actual[0], atol=atol)
    self.assertAllClose(expected[1], actual[1], atol=atol)

  @parameterized.parameters(((1.0, 0.5), 0.0), ((1.0, 1.0), 0.0),
                            ((1.0, 0.0), 0.0), ((1.0, 0.0), 0.0),
                            ((1.0, 0.5), 1.0), ((1.0, 1.0), 1.0),
                            ((1.0, 0.0), 1.0))
  def test_diffusion_2d_periodic_and_dirichlet(self, offset, value_lo):
    bc = boundaries.periodic_and_dirichlet_boundary_conditions((value_lo, 0.0))
    x, b = self.diffusion_setup(bc, offset)
    self.assertAllClose(
        x.data,
        bc.trim_boundary(b).data,
        atol=1e-5)
    self.assertArrayEqual(x.grid, b.grid)

  @parameterized.parameters(((1.0, 0.5),), ((0.5, 0.5),))
  def test_diffusion_2d_periodic_and_neumann(self, offset):
    bc = boundaries.periodic_and_neumann_boundary_conditions()
    x, b = self.diffusion_setup(bc, offset)
    self.assertAllClose(
        x.data,
        bc.trim_boundary(b).data,
        atol=1e-5)
    self.assertArrayEqual(x.grid, b.grid)

  @parameterized.parameters(((0.5, 0.5),))
  def test_diffusion_2d_fully_neumann(self, offset):
    bc = boundaries.neumann_boundary_conditions(2)
    x, b = self.diffusion_setup(bc, offset)
    self.assertAllClose(
        x.data,
        bc.trim_boundary(b).data,
        atol=1e-5)
    self.assertArrayEqual(x.grid, b.grid)

  @parameterized.parameters(((1.0, 0.5),), ((1.0, 1.0),), ((1.0, 0.0),))
  def test_diffusion_2d_fully_dirichlet(self, offset):
    bc = boundaries.dirichlet_boundary_conditions(2)
    x, b = self.diffusion_setup(bc, offset)
    self.assertAllClose(
        x.data,
        bc.trim_boundary(b).data,
        atol=1e-5)
    self.assertArrayEqual(x.grid, b.grid)

if __name__ == '__main__':
  absltest.main()
