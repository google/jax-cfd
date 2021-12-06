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


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
