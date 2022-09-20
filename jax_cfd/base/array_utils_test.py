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
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np
import scipy.interpolate as spi
import skimage.measure as skm

BCType = boundaries.BCType


class ArrayUtilsTest(test_util.TestCase):

  @parameterized.parameters(
      dict(array=np.random.RandomState(1234).randn(3, 6, 9),
           block_size=(3, 3, 3),
           f=jnp.mean),
      dict(array=np.random.RandomState(1234).randn(12, 24, 36),
           block_size=(6, 6, 6),
           f=jnp.max),
      dict(array=np.random.RandomState(1234).randn(12, 24, 36),
           block_size=(3, 4, 6),
           f=jnp.min),
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
      # Periodic BC
      dict(
          offset=(0,),
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          expected=[[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1,
                                                                  -2]]),
      dict(
          offset=(0.5,),
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          expected=[[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1,
                                                                  -2]]),
      dict(
          offset=(1.,),
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          expected=[[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1,
                                                                  -2]]),
      # Dirichlet BC
      dict(
          offset=(0,),
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          expected=[[-2, 1, 0], [1, -2, 1], [0, 1, -2]]),
      dict(
          offset=(0.5,),
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          expected=[[-3, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1,
                                                                  -3]]),
      dict(
          offset=(1.,),
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          expected=[[-2, 1, 0], [1, -2, 1], [0, 1, -2]]),
      # Neumann BC
      dict(
          offset=(0.5,),
          bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
          expected=[[-1, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1,
                                                                  -1]]),
      # Neumann-Dirichlet BC
      dict(
          offset=(0.5,),
          bc_types=((BCType.NEUMANN, BCType.DIRICHLET),),
          expected=[[-1, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1,
                                                                  -3]]),

  )
  def test_laplacian_matrix_w_boundaries(self, offset, bc_types, expected):
    grid = grids.Grid((4,), step=(.5,))
    bc = boundaries.HomogeneousBoundaryConditions(bc_types)
    actual = array_utils.laplacian_matrix_w_boundaries(grid, offset, bc)
    actual = np.squeeze(actual)
    expected = 4.0 * np.array(expected)
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


class Interp1DTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='FillWithExtrapolate', fill_value='extrapolate'),
      dict(testcase_name='FillWithScalar', fill_value=123),
  )
  def test_same_as_scipy_on_scalars_and_check_grads(self, fill_value):
    rng = np.random.RandomState(45)

    n = 10
    y = rng.randn(n)

    x_low = 5.
    x_high = 9.
    x = np.linspace(x_low, x_high, num=n)

    sp_func = spi.interp1d(
        x, y, kind='linear', fill_value=fill_value, bounds_error=False)
    cfd_func = array_utils.interp1d(x, y, fill_value=fill_value)

    # Check x_new at the table definition points `x`, points outside, and points
    # in between.
    x_to_check = np.concatenate(([x_low - 1], x, x * 1.051, [x_high + 1]))

    for x_new in x_to_check:
      sp_y_new = sp_func(x_new).astype(np.float32)
      cfd_y_new = cfd_func(x_new)
      self.assertAllClose(sp_y_new, cfd_y_new, rtol=1e-5, atol=1e-6)

      grad_cfd_y_new = jax.grad(cfd_func)(x_new)

      # Gradients should be nonzero except when outside the range of `x` and
      # filling with a constant.
      # Why check this? Because, some indexing methods result in gradients == 0
      # at the interpolation table points.
      if fill_value == 'extrapolate' or x_low <= x_new <= x_high:
        self.assertLess(0, np.abs(grad_cfd_y_new))
      else:
        self.assertTrue(np.isfinite(grad_cfd_y_new))

  @parameterized.named_parameters(
      dict(
          testcase_name='TableIs1D_XNewIs1D_Axis0_FillWithScalar',
          table_ndim=1,
          x_new_ndim=1,
          axis=0,
          fill_value=12345,
      ),
      dict(
          testcase_name='TableIs1D_XNewIs1D_Axis0_FillWithScalar_NoAssumeSorted',
          table_ndim=1,
          x_new_ndim=1,
          axis=0,
          fill_value=12345,
          assume_sorted=False,
      ),
      dict(
          testcase_name='TableIs1D_XNewIs1D_Axis0_FillWithExtrapolate',
          table_ndim=1,
          x_new_ndim=1,
          axis=0,
          fill_value='extrapolate',
      ),
      dict(
          testcase_name='TableIs2D_XNewIs1D_Axis0_FillWithExtrapolate',
          table_ndim=2,
          x_new_ndim=1,
          axis=0,
          fill_value='extrapolate',
      ),
      dict(
          testcase_name=(
              'TableIs2D_XNewIs1D_Axis0_FillWithExtrapolate_NoAssumeSorted'),
          table_ndim=2,
          x_new_ndim=1,
          axis=0,
          fill_value='extrapolate',
          assume_sorted=False,
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axis1_FillWithScalar',
          table_ndim=3,
          x_new_ndim=2,
          axis=1,
          fill_value=12345,
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn1_FillWithScalar',
          table_ndim=3,
          x_new_ndim=2,
          axis=-1,
          fill_value=12345,
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn1_FillWithExtrapolate',
          table_ndim=3,
          x_new_ndim=2,
          axis=-1,
          fill_value='extrapolate',
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn3_FillWithScalar',
          table_ndim=3,
          x_new_ndim=2,
          axis=-3,
          fill_value=1234,
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn3_FillWithConstantExtrapolate',
          table_ndim=3,
          x_new_ndim=2,
          axis=-3,
          fill_value='constant_extrapolate',
      ),
      dict(
          testcase_name=(
              'Table3D_XNew2D_Axisn3_FillConstantExtrapolate_NoAssumeSorted'),
          table_ndim=3,
          x_new_ndim=2,
          axis=-3,
          fill_value='constant_extrapolate',
          assume_sorted=False,
      ),
  )
  def test_same_as_scipy_on_arrays(
      self,
      table_ndim,
      x_new_ndim,
      axis,
      fill_value,
      assume_sorted=True,
  ):
    """Test results are the same as scipy.interpolate.interp1d."""
    rng = np.random.RandomState(45)

    # Arbitrary shape that ensures all dims are different to prevent
    # broadcasting from hiding bugs.
    y_shape = tuple(range(5, 5 + table_ndim))
    y = rng.randn(*y_shape)
    n = y_shape[axis]

    x_low = 5
    x_high = 9
    x = np.linspace(x_low, x_high, num=n)**2  # Arbitrary non-linearly spaced x
    if not assume_sorted:
      rng.shuffle(x)

    # Scipy doesn't have 'constant_extrapolate', so treat it special.
    # Here use np.nan as the fill value, which will be easy to spot if we handle
    # it wrong.
    if fill_value == 'constant_extrapolate':
      sp_fill_value = np.nan
    else:
      sp_fill_value = fill_value

    sp_func = spi.interp1d(
        x,
        y,
        kind='linear',
        axis=axis,
        fill_value=sp_fill_value,
        bounds_error=False,
        assume_sorted=assume_sorted,
    )
    cfd_func = array_utils.interp1d(
        x, y, axis=axis, fill_value=fill_value, assume_sorted=assume_sorted)

    # Make n_x_new > n, so we can selectively fill values as below.
    n_x_new = max(2 * n, 20)

    # Make x_new over the same range as x.
    x_new_shape = tuple(range(2, x_new_ndim + 1)) + (n_x_new,)
    x_new = (x_low + rng.rand(*x_new_shape) * (x_high - x_low))**2

    x_new[..., 0] = np.min(x) - 1  # Out of bounds low
    x_new[..., -1] = np.max(x) + 1  # Out of bounds high
    x_new[..., 1:n + 1] = x  # All the grid points

    # Scipy doesn't have the 'constant_extrapolate' feature, but
    # constant_extrapolate is achieved by clipping the input.
    if fill_value == 'constant_extrapolate':
      sp_x_new = np.clip(x_new, np.min(x), np.max(x))
    else:
      sp_x_new = x_new

    sp_y_new = sp_func(sp_x_new).astype(np.float32)
    cfd_y_new = cfd_func(x_new)
    self.assertAllClose(sp_y_new, cfd_y_new, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
