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

"""Tests for jax_cfd.fast_diagonalization."""
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import fast_diagonalization
from jax_cfd.base import test_util
import numpy as np
import scipy.linalg


Array = fast_diagonalization.Array


def apply_operators(
    operators: Sequence[np.ndarray],
    rhs: Array,
) -> Array:
  """Apply a sum of linear operators along all array axes."""
  assert len(operators) == rhs.ndim
  out = 0
  for axis, matrix in enumerate(operators):
    axes = [i if i != axis else rhs.ndim for i in range(rhs.ndim)]
    out += jnp.einsum(
        matrix, [axis, rhs.ndim], rhs, list(range(rhs.ndim)), axes)
  return out


class FastDiagonalizationTest(test_util.TestCase):

  def test_random_1d_matmul(self):
    rs = np.random.RandomState(0)
    a = rs.randn(3, 3)
    a = jnp.array(a + a.T, np.float32)
    b = rs.randn(3).astype(np.float32)
    a_inv = fast_diagonalization.psuedoinverse(
        [a], b.dtype, hermitian=True, implementation='matmul')
    actual = a_inv(b)
    expected = jnp.linalg.solve(a, b)
    self.assertAllClose(actual, expected, atol=1e-6)

  @parameterized.parameters('fft', 'rfft')
  def test_random_1d_fft(self, implementation):
    rs = np.random.RandomState(0)
    a = jnp.array(scipy.linalg.circulant(rs.randn(4)), np.float32)
    b = rs.randn(4).astype(np.float32)
    a_inv = fast_diagonalization.psuedoinverse(
        [a], b.dtype, circulant=True, implementation=implementation)
    actual = a_inv(b)
    expected = jnp.linalg.solve(a, b)
    self.assertAllClose(actual, expected, atol=1e-5)

  @parameterized.parameters(
      *[(ndim, 'matmul') for ndim in [1, 2, 3]],
      *[(ndim, 'fft') for ndim in [1, 2, 3]],
      *[(ndim, 'rfft') for ndim in [1, 2, 3]],
  )
  def test_identity_nd(self, ndim, implementation):
    rs = np.random.RandomState(0)
    b = rs.randn(*(2, 4, 6)[:ndim]).astype(np.float32)
    ops = [np.eye(2), 2 * np.eye(4), 3 * np.eye(6)]
    a_inv = fast_diagonalization.psuedoinverse(
        ops[:ndim], b.dtype, hermitian=True, circulant=True,
        implementation=implementation)
    actual = a_inv(b)
    expected = b / sum(range(1, 1 + ndim))
    self.assertAllClose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.parameters('matmul', 'fft', 'rfft')
  def test_poisson_1d(self, implementation):
    rs = np.random.RandomState(0)
    a = jnp.array([[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]],
                  np.float32)
    b = rs.randn(4).astype(np.float32)
    a_inv = fast_diagonalization.psuedoinverse(
        [a], a.dtype, hermitian=True, circulant=True,
        implementation=implementation)
    x = a_inv(b)
    self.assertAllClose(jnp.dot(a, x), b - b.mean(), atol=1e-5)

  @parameterized.parameters(
      dict(periodic_x=False, periodic_y=False),
      dict(periodic_x=False, periodic_y=True),
      dict(periodic_x=True, periodic_y=True),
  )
  def test_poisson_2d_matmul(self, periodic_x, periodic_y):
    a1 = jnp.array([[-2, 1, 0, periodic_x], [1, -2, 1, 0], [0, 1, -2, 1],
                    [periodic_x, 0, 1, -2]], dtype=np.float32)
    a2 = jnp.array([[-2, 1, periodic_y], [1, -2, 1], [periodic_y, 1, -2]],
                   dtype=np.float32)

    b = np.random.RandomState(0).randn(4, 3).astype(np.float32)
    operators = [a1, a2]
    a_inv = fast_diagonalization.psuedoinverse(
        operators, b.dtype, hermitian=True)
    x = a_inv(b)
    actual = apply_operators(operators, x)
    expected = b.copy()
    if periodic_x and periodic_y:
      expected -= expected.mean()
    self.assertAllClose(actual, expected, atol=1e-5)

  @parameterized.parameters('fft', 'rfft')
  def test_poisson_2d_fft(self, implementation):
    a1 = array_utils.laplacian_matrix(size=4, step=1.0)
    a2 = array_utils.laplacian_matrix(size=6, step=1.0)
    b = np.random.RandomState(0).randn(4, 6).astype(np.float32)
    operators = [a1, a2]
    a_inv = fast_diagonalization.psuedoinverse(
        operators, b.dtype, circulant=True, implementation=implementation)
    x = a_inv(b)
    actual = apply_operators(operators, x)
    expected = b - b.mean()
    self.assertAllClose(actual, expected, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
