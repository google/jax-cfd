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

"""Tests for jax_cfd.pressure."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import pressure
from jax_cfd.base import test_util
import numpy as np

USE_FLOAT64 = True


def _offsets(n):
  return tuple(tuple(o) for o in (np.eye(n) + np.ones([n, n])) / 2.)


solve_cg = functools.partial(pressure.solve_cg, atol=1e-6, maxiter=10**5)


class PressureTest(test_util.TestCase):

  def setUp(self):
    jax.config.update('jax_enable_x64', USE_FLOAT64)
    super(PressureTest, self).setUp()

  @parameterized.named_parameters(
      dict(testcase_name='_1D_cg',
           shape=(301,),
           solve=solve_cg,
           step=(.1,),
           density=10.,
           seed=111),
      dict(testcase_name='_2D_cg',
           shape=(100, 100),
           solve=solve_cg,
           step=(1., 1.),
           density=1.,
           seed=222),
      dict(testcase_name='_3D_cg',
           shape=(10, 10, 10),
           solve=solve_cg,
           step=(.1, .1, .1),
           density=3.,
           seed=333),
      dict(testcase_name='_1D_fast_diag',
           shape=(301,),
           solve=pressure.solve_fast_diag,
           step=(.1,),
           density=10.,
           seed=111),
      dict(testcase_name='_2D_fast_diag',
           shape=(100, 100),
           solve=pressure.solve_fast_diag,
           step=(1., 1.),
           density=1.,
           seed=222),
      dict(testcase_name='_3D_fast_diag',
           shape=(10, 10, 10),
           solve=pressure.solve_fast_diag,
           step=(.1, .1, .1),
           density=3.,
           seed=333),
  )
  def testPressureCorrectionZeroDivergence(
      self, shape, solve, step, density, seed):
    """Returned velocity should be divergence free."""
    grid = grids.Grid(shape, step)

    # The uncorrected velocity is a 1 + a small amount of noise in each
    # dimension.
    ks = jax.random.split(jax.random.PRNGKey(seed), 2 * len(shape))
    v = tuple(
        grids.AlignedArray(1. + .3 * jax.random.normal(k, shape), offset)
        for k, offset in zip(ks[:len(shape)], _offsets(len(shape))))
    v_corrected = pressure.projection(v, grid, solve)

    # The corrected velocity should be divergence free.
    div = fd.divergence(v_corrected, grid)
    for u, u_corrected in zip(v, v_corrected):
      np.testing.assert_allclose(u.offset, u_corrected.offset)
    np.testing.assert_allclose(div.data, 0., atol=1e-4)


if __name__ == '__main__':
  absltest.main()
