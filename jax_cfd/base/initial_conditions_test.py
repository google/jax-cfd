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

"""Tests for jax_cfd.initial_conditions."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import initial_conditions as ic
from jax_cfd.base import pressure
from jax_cfd.base import test_util
import numpy as np


def get_grid(grid_size, ndim, domain_size_multiple=1):
  domain = ((0, 2 * np.pi * domain_size_multiple),) * ndim
  shape = (grid_size,) * ndim
  return grids.Grid(shape=shape, domain=domain)


class InitialConditionsTest(test_util.TestCase):

  @parameterized.parameters(
      dict(seed=3232,
           grid=get_grid(128, ndim=3),
           maximum_velocity=1.,
           peak_wavenumber=2),
      dict(seed=2323,
           grid=get_grid(1024, ndim=2),
           maximum_velocity=10.,
           peak_wavenumber=17),
  )
  def test_filtered_velocity_field(
      self, seed, grid, maximum_velocity, peak_wavenumber):
    v = ic.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid, maximum_velocity, peak_wavenumber)
    actual_maximum_velocity = jnp.linalg.norm([u.data for u in v], axis=0).max()
    max_divergence = fd.divergence(v).data.max()

    # Assert that initial velocity is divergence free
    self.assertAllClose(0., max_divergence, atol=5e-4)

    # Assert that the specified maximum velocity is obtained.
    self.assertAllClose(maximum_velocity, actual_maximum_velocity, atol=1e-5)

  def test_initial_velocity_field_no_projection(self):
    # Test on an already incompressible velocity field
    grid = grids.Grid((10, 10), step=1.0)
    x_velocity_fn = lambda x, y: jnp.ones_like(x)
    y_velocity_fn = lambda x, y: jnp.zeros_like(x)
    v0 = ic.initial_velocity_field((x_velocity_fn, y_velocity_fn), grid)
    expected_v0 = (
        grids.GridVariable(
            grids.GridArray(jnp.ones((10, 10)), (1, 0.5), grid),
            boundaries.periodic_boundary_conditions(grid.ndim)),
        grids.GridVariable(
            grids.GridArray(jnp.zeros((10, 10)), (0.5, 1), grid),
            boundaries.periodic_boundary_conditions(grid.ndim)),
        )
    for d in range(len(v0)):
      self.assertArrayEqual(expected_v0[d], v0[d])

    with self.subTest('correction does not change answer'):
      v0_corrected = ic.initial_velocity_field(
          (x_velocity_fn, y_velocity_fn), grid, iterations=5)
      for d in range(len(v0)):
        self.assertIsInstance(v0_corrected[d], grids.GridVariable)
        self.assertArrayEqual(expected_v0[d], v0_corrected[d])

  @parameterized.parameters(
      dict(
          velocity_bc=(boundaries.dirichlet_boundary_conditions(2),
                       boundaries.dirichlet_boundary_conditions(2)),
          pressure_solve=pressure.solve_cg,
          ),
      dict(
          velocity_bc=(boundaries.channel_flow_boundary_conditions(2),
                       boundaries.channel_flow_boundary_conditions(2)),
          pressure_solve=pressure.solve_cg,
          ),
      dict(
          velocity_bc=(boundaries.channel_flow_boundary_conditions(2),
                       boundaries.channel_flow_boundary_conditions(2)),
          pressure_solve=pressure.solve_fast_diag_channel_flow,
          ),
      dict(velocity_bc=None,  # default is all periodic BC.
           pressure_solve=pressure.solve_fast_diag,
          ),
      )
  def test_initial_velocity_field_with_projection(self, velocity_bc,
                                                  pressure_solve):
    grid = grids.Grid((20, 20), step=0.1)
    # Use a mask to make the random noise zero on the boundaries, consistent
    # with Dirichlet BC (and still valid for periodic BC).
    masks = grids.domain_interior_masks(grid)

    def x_velocity_fn(x, y):
      return jnp.zeros_like(x + y) + 0.2 * np.random.normal(
          size=grid.shape) * masks[0]

    def y_velocity_fn(x, y):
      return jnp.zeros_like(x + y) + 0.2 * np.random.normal(
          size=grid.shape) * masks[1]

    with self.subTest('corrected'):
      v0_corrected = ic.initial_velocity_field((x_velocity_fn, y_velocity_fn),
                                               grid,
                                               velocity_bc,
                                               pressure_solve,
                                               iterations=5)
      self.assertAllClose(fd.divergence(v0_corrected).data, 0, atol=1e-5)

    with self.subTest('not corrected'):
      v0_uncorrected = ic.initial_velocity_field((x_velocity_fn, y_velocity_fn),
                                                 grid,
                                                 velocity_bc,
                                                 pressure_solve,
                                                 iterations=None)
      self.assertGreater(abs(fd.divergence(v0_uncorrected).data).max(), 0.1)


if __name__ == '__main__':
  absltest.main()
