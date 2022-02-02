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
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import test_util
from jax_cfd.collocated import initial_conditions
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
  )
  def test_filtered_velocity_field(
      self, seed, grid, maximum_velocity, peak_wavenumber):
    v = initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid, maximum_velocity, peak_wavenumber)
    actual_maximum_velocity = jnp.linalg.norm([u.data for u in v], axis=0).max()
    max_divergence = fd.centered_divergence(v).data.max()

    # Assert that initial velocity is divergence free
    self.assertAllClose(0., max_divergence, atol=1e-4)

    # Assert that the specified maximum velocity is obtained.
    self.assertAllClose(maximum_velocity, actual_maximum_velocity, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
