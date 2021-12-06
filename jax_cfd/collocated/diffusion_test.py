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
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import test_util
from jax_cfd.collocated import diffusion


class DiffusionTest(test_util.TestCase):
  """Some simple sanity tests for diffusion on constant fields."""

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


if __name__ == '__main__':
  absltest.main()
