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

"""Tests of array utils."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import test_util
import numpy as np
import skimage.measure as skm


class ArrayUtilsTest(test_util.TestCase):

  @parameterized.parameters(
      dict(array=np.random.RandomState(1234).randn(3, 6, 9),
           block_size=(3, 3, 3),
           f=jnp.mean),
      dict(array=np.random.RandomState(1234).randn(12, 24, 36),
           block_size=(6, 6, 6),
           f=jnp.max),
  )
  def test_block_reduce(self, array, block_size, f):
    """Test `block_reduce` is equivalent to `skimage.measure.block_reduce`."""
    expected_output = skm.block_reduce(array, block_size, f)
    actual_output = array_utils.block_reduce(array, block_size, f)
    self.assertAllClose(expected_output, actual_output, atol=1e-6)

  def test_laplacian_matrix(self):
    actual = array_utils.laplacian_matrix(4, step=0.5)
    expected = 4.0 * np.array(
        [[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]])
    self.assertAllClose(expected, actual)

  @parameterized.parameters(
      dict(matrix=(np.random.RandomState(1234).randn(16, 2))),
      dict(matrix=(np.random.RandomState(1234).randn(24, 1))),
      dict(matrix=(np.random.RandomState(1234).randn(74, 4))),
  )
  def test_gram_schmidt_qr(self, matrix):
    """Tests that gram-schmidt_qr is close to numpy for slim matrices."""
    q_actual, r_actual = array_utils.gram_schmidt_qr(matrix)
    q, r = jnp.linalg.qr(matrix)
    # we rearrange the result to make the diagonal of `r` positive.
    d = jnp.diag(jnp.sign(jnp.diagonal(r)))
    q_expected = q @ d
    r_expected = d @ r
    self.assertAllClose(q_expected, q_actual, atol=1e-4)
    self.assertAllClose(r_expected, r_actual, atol=1e-4)

  @parameterized.parameters(
      dict(pytree=(np.zeros((6, 3)), np.ones((6, 2, 2))), idx=3, axis=0),
      dict(pytree=(np.zeros((3, 8)), np.ones((6, 8))), idx=3, axis=-1),
      dict(pytree={'a': np.zeros((3, 9)), 'b': np.ones((6, 9))}, idx=3, axis=1),
      dict(pytree=np.zeros((13, 5)), idx=6, axis=0),
      dict(pytree=(np.zeros(9), (np.ones((9, 1)), np.ones(9))), idx=6, axis=0),
  )
  def test_split_and_concat(self, pytree, idx, axis):
    """Tests that split_along_axis, concat_along_axis return expected shapes."""
    split_a, split_b = array_utils.split_along_axis(pytree, idx, axis, False)
    with self.subTest('split_shape'):
      self.assertEqual(jax.tree_leaves(split_a)[0].shape[axis], idx)

    reconstruction = array_utils.concat_along_axis([split_a, split_b], axis)
    with self.subTest('split_concat_roundtrip_structure'):
      actual_tree_def = jax.tree_structure(reconstruction)
      expected_tree_def = jax.tree_structure(pytree)
      self.assertSameStructure(actual_tree_def, expected_tree_def)

    actual_values = jax.tree_leaves(reconstruction)
    expected_values = jax.tree_leaves(pytree)
    with self.subTest('split_concat_roundtrip_values'):
      for actual, expected in zip(actual_values, expected_values):
        self.assertAllClose(actual, expected)

    same_ndims = len(set(a.ndim for a in actual_values)) == 1
    if not same_ndims:
      with self.subTest('raises_when_wrong_ndims'):
        with self.assertRaisesRegex(ValueError, 'arrays in `inputs` expected'):
          split_a, split_b = array_utils.split_along_axis(pytree, idx, axis)

    with self.subTest('multiple_concat_shape'):
      arrays = [split_a, split_a, split_b, split_b]
      double_concat = array_utils.concat_along_axis(arrays, axis)
      actual_shape = jax.tree_leaves(double_concat)[0].shape[axis]
      expected_shape = jax.tree_leaves(pytree)[0].shape[axis] * 2
      self.assertEqual(actual_shape, expected_shape)

  @parameterized.parameters(
      dict(pytree=(np.zeros((6, 3)), np.ones((6, 2, 2))), axis=0),
      dict(pytree={'a': np.zeros((3, 9)), 'b': np.ones((6, 9))}, axis=1),
      dict(pytree=np.zeros((13, 5)), axis=0),
      dict(pytree=(np.zeros(9), (np.ones((9, 1)), np.ones(9))), axis=0),
  )
  def test_split_along_axis_shapes(self, pytree, axis):
    with self.subTest('with_keep_dims'):
      splits = array_utils.split_axis(pytree, axis, keep_dims=True)
      get_expected_shape = lambda x: x.shape[:axis] + (1,) + x.shape[axis + 1:]
      expected_shapes = jax.tree_map(get_expected_shape, pytree)
      for split in splits:
        actual = jax.tree_map(lambda x: x.shape, split)
        self.assertEqual(expected_shapes, actual)

    with self.subTest('without_keep_dims'):
      splits = array_utils.split_axis(pytree, axis, keep_dims=False)
      get_expected_shape = lambda x: x.shape[:axis] + x.shape[axis + 1:]
      expected_shapes = jax.tree_map(get_expected_shape, pytree)
      for split in splits:
        actual = jax.tree_map(lambda x: x.shape, split)
        self.assertEqual(expected_shapes, actual)


if __name__ == '__main__':
  absltest.main()
