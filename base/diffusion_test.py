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
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.base import test_util


class DiffusionTest(test_util.TestCase):
  """Some simple sanity tests for diffusion on constant fields."""

  def test_explicit_diffusion(self):
    nu = 1.
    shape = (101, 101, 101)
    offset = (0, 0, 0)
    step = (1., 1., 1.)
    grid = grids.Grid(shape, step)

    c = grids.AlignedArray(jnp.ones(shape), offset)
    diffused = diffusion.diffuse(c, nu, grid)
    expected = grids.AlignedArray(jnp.zeros_like(diffused.data), offset)
    self.assertAllClose(expected, diffused)

  @parameterized.parameters(
      dict(solve=diffusion.solve_cg, atol=1e-6),
      dict(solve=diffusion.solve_fast_diag, atol=1e-6),
  )
  def test_implicit_diffusion(self, solve, atol):
    nu = 1.
    dt = 0.1
    shape = (100, 100)
    offset = (0, 0)
    grid = grids.Grid(shape, step=1)
    v = 2 * (grids.AlignedArray(jnp.ones(shape), offset),)
    diffused = solve(v, nu, dt, grid)
    self.assertAllClose(v[0], diffused[0], atol=atol)
    self.assertAllClose(v[1], diffused[1], atol=atol)


if __name__ == '__main__':
  absltest.main()
