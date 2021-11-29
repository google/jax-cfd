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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np


def _make_zero_velocity_field(grid):
  ndim = grid.ndim
  offsets = (np.eye(ndim) + np.ones([ndim, ndim])) / 2.
  bc = boundaries.periodic_boundary_conditions(grid.ndim)
  return tuple(
      grids.GridVariable(
          grids.GridArray(jnp.zeros(grid.shape), tuple(offset), grid), bc)
      for ax, offset in enumerate(offsets))


class ForcingsTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_taylor_green_forcing',
          partial_force_fn=functools.partial(
              forcings.taylor_green_forcing, scale=1.0, k=2),
      ),
      dict(
          testcase_name='_kolmogorov_forcing',
          partial_force_fn=functools.partial(
              forcings.kolmogorov_forcing, scale=1.0, k=2),
      ),
      dict(
          testcase_name='_linear_forcing',
          partial_force_fn=functools.partial(
              forcings.linear_forcing, coefficient=2.0),
      ),
      dict(
          testcase_name='_no_forcing',
          partial_force_fn=functools.partial(forcings.no_forcing)),
      dict(
          testcase_name='_simple_turbulence_forcing',
          partial_force_fn=functools.partial(
              forcings.simple_turbulence_forcing,
              constant_magnitude=0.0,
              constant_wavenumber=2,
              linear_coefficient=0.0,
              forcing_type='kolmogorov'),
      ),
  )
  def test_forcing_function(self, partial_force_fn):
    for ndim in [2, 3]:
      with self.subTest(f'ndim={ndim}'):
        grid = grids.Grid((16,) * ndim)
        v = _make_zero_velocity_field(grid)
        force_fn = partial_force_fn(grid)
        force = force_fn(v)
        # Check that offset and grid match velocity input
        for d in range(ndim):
          self.assertAllClose(0 * force[d], v[d].array)

  def test_sum_forcings(self):
    grid = grids.Grid((16, 16))
    force_fn_1 = forcings.kolmogorov_forcing(grid, scale=1.0, k=2)
    force_fn_2 = forcings.no_forcing(grid)
    force_fn_sum = forcings.sum_forcings(force_fn_1, force_fn_2)
    v = _make_zero_velocity_field(grid)
    force_1 = force_fn_1(v)
    force_sum = force_fn_sum(v)
    for d in range(grid.ndim):
      self.assertArrayEqual(force_sum[d], force_1[d])

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
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    velocity = tuple(
        grids.GridVariable(grids.GridArray(u, offset, grid), bc)
        for u, offset in zip(velocity_function(*grid.mesh()), grid.cell_faces))
    expected_force = expected_force_function(*grid.mesh())
    actual_force = forcings.filtered_linear_forcing(
        lower_wavenumber, upper_wavenumber, coefficient, grid)(velocity)
    for expected, actual in zip(expected_force, actual_force):
      self.assertAllClose(expected, actual.data, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
