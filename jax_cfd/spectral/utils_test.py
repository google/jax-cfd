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

"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax_cfd.base import finite_differences
from jax_cfd.base import grids
from jax_cfd.base import initial_conditions
from jax_cfd.base import interpolation
from jax_cfd.base import test_util
from jax_cfd.spectral import utils


class ThreeOverTwoRuleTest1D(test_util.TestCase):

  def test_rfft_padding_and_truncation(self):
    # This test is essentially recreating Figure 4 of go/uecker
    n = 8
    grid = grids.Grid((n,), domain=((0, 2 * jnp.pi),))
    xs, = grid.axes()
    u = jnp.cos(3 * xs)
    uhat = jnp.fft.rfft(u)
    k, = uhat.shape
    uhat_squared = utils.truncated_rfft(utils.padded_irfft(uhat)**2)
    assert len(uhat_squared) == k
    u_squared = jnp.fft.irfft(uhat_squared)
    self.assertAllClose(.5, u_squared, atol=1e-4)


class NavierStokesHelpersTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_seed=0', seed=0),
      dict(testcase_name='_seed=1', seed=1))
  def test_construct_circular_filter(self, seed):
    grid = grids.Grid((8, 8), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    mask = utils.circular_filter_2d(grid)

    # check that masking decreasing the l2-norm.
    key = jax.random.PRNGKey(seed)
    signal = jax.random.normal(key, (8, 8))
    signal_hat = jnp.fft.rfftn(signal)
    self.assertLess(
        jnp.linalg.norm(mask * signal_hat), jnp.linalg.norm(signal_hat))

  @parameterized.named_parameters(
      dict(testcase_name='_atol=1e-2',
           atol=1e-2,
           grid=grids.Grid((128, 128),
                           domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))))
  def test_vorticity_to_velocity_round_trip(self, atol, grid):
    """Check that velocity solve and curl 2d are inverses."""

    u, v = initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(42), grid, maximum_velocity=7, peak_wavenumber=1)

    velocity_solve = utils.vorticity_to_velocity(grid)
    vorticity = finite_differences.curl_2d((u, v))
    vorticity_hat = jnp.fft.rfftn(vorticity.data)
    uhat, vhat = velocity_solve(vorticity_hat)

    self.assertAllClose(
        jnp.fft.irfftn(uhat),
        interpolation.linear(u, vorticity.offset).data,
        atol=atol)

    self.assertAllClose(
        jnp.fft.irfftn(vhat),
        finite_differences.interpolation.linear(v, vorticity.offset).data,
        atol=atol)


if __name__ == '__main__':
  absltest.main()
