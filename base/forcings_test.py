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

"""Tests for jax_cfd.forcings."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.base import test_util


class ForcingsTest(test_util.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      dict(testcase_name='_low_frequency_unchanged_2D',
           ndim=2,
           grid_size=16,
           lower_wavenumber=0,
           upper_wavenumber=2,
           coefficient=1,
           # velocity is concentrated on wavenumber < sqrt(2)
           # so we expect it to pass through the filter.
           velocity_function=lambda x, y: (jnp.cos(2 * jnp.pi * x),
                                           jnp.cos(2 * jnp.pi * y)),
           expected_force_function=lambda x, y: (jnp.cos(2 * jnp.pi * x),
                                                 jnp.cos(2 * jnp.pi * y))),
      dict(testcase_name='_high_frequency_zeros_3D',
           ndim=3,
           grid_size=16,
           lower_wavenumber=0,
           upper_wavenumber=1,
           coefficient=1,
           # velocity is concentrated on wave numbers 2 to 2 * sqrt(3)
           # so we expect it to be filtered entirely.
           velocity_function=lambda x, y, z: (jnp.cos(4 * jnp.pi * x),
                                              jnp.cos(4 * jnp.pi * y),
                                              jnp.cos(4 * jnp.pi * z),),
           expected_force_function=lambda x, y, z: (jnp.zeros_like(x),
                                                    jnp.zeros_like(y),
                                                    jnp.zeros_like(z))),
  )
  def test_filtered_linear_forcing(self,
                                   ndim,
                                   grid_size,
                                   lower_wavenumber,
                                   upper_wavenumber,
                                   coefficient,
                                   velocity_function,
                                   expected_force_function):
    grid = grids.Grid((grid_size,) * ndim,
                      domain=((0, 1),) * ndim)
    velocity = tuple(grids.AlignedArray(u, offset) for u, offset in
                     zip(velocity_function(*grid.mesh()), grid.cell_faces))
    expected_force = expected_force_function(*grid.mesh())
    actual_force = forcings.filtered_linear_forcing(
        lower_wavenumber, upper_wavenumber, coefficient)(velocity, grid)
    for expected, actual in zip(expected_force, actual_force):
      self.assertAllClose(expected, actual.data, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
