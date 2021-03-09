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

"""Utility methods for manipulating array-like objects."""

from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax_cfd.base import grids
import numpy as np
import scipy.linalg


# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any
Array = grids.Array


def _normalize_axis(axis: int, ndim: int) -> int:
  """Validates and returns positive `axis` value."""
  if not -ndim <= axis < ndim:
    raise ValueError(f'invalid axis {axis} for ndim {ndim}')
  if axis < 0:
    axis += ndim
  return axis


def slice_along_axis(
    inputs: PyTree,
    axis: int,
    idx: Union[slice, int],
    expect_same_dims: bool = True
) -> PyTree:
  """Returns slice of `inputs` defined by `idx` along axis `axis`.

  Args:
    inputs: array or a tuple of arrays to slice.
    axis: axis along which to slice the `inputs`.
    idx: index or slice along axis `axis` that is returned.
    expect_same_dims: whether all arrays should have same number of dimensions.

  Returns:
    Slice of `inputs` defined by `idx` along axis `axis`.
  """
  arrays, tree_def = jax.tree_flatten(inputs)
  ndims = set(a.ndim for a in arrays)
  if expect_same_dims and len(ndims) != 1:
    raise ValueError('arrays in `inputs` expected to have same ndims, but have '
                     f'{ndims}. To allow this, pass expect_same_dims=False')
  sliced = []
  for array in arrays:
    ndim = array.ndim
    slc = tuple(idx if j == _normalize_axis(axis, ndim) else slice(None)
                for j in range(ndim))
    sliced.append(array[slc])
  return jax.tree_unflatten(tree_def, sliced)


def split_along_axis(
    inputs: PyTree,
    split_idx: int,
    axis: int,
    expect_same_dims: bool = True
) -> Tuple[PyTree, PyTree]:
  """Returns a tuple of slices of `inputs` split along `axis` at `split_idx`.

  Args:
    inputs: pytree of arrays to split.
    split_idx: index along `axis` where the second split starts.
    axis: axis along which to split the `inputs`.
    expect_same_dims: whether all arrays should have same number of dimensions.

  Returns:
    Tuple of slices of `inputs` split along `axis` at `split_idx`.
  """

  first_slice = slice_along_axis(
      inputs, axis, slice(0, split_idx), expect_same_dims)
  second_slice = slice_along_axis(
      inputs, axis, slice(split_idx, None), expect_same_dims)
  return first_slice, second_slice


def split_axis(
    inputs: PyTree,
    axis: int,
    keep_dims: bool = False
) -> Tuple[PyTree, ...]:
  """Splits the arrays in `inputs` along `axis`.

  Args:
    inputs: pytree to be split.
    axis: axis along which to split the `inputs`.
    keep_dims: whether to keep `axis` dimension.

  Returns:
    Tuple of pytrees that correspond to slices of `inputs` along `axis`. The
    `axis` dimension is removed if `squeeze is set to True.

  Raises:
    ValueError: if arrays in `inputs` don't have unique size along `axis`.
  """
  arrays, tree_def = jax.tree_flatten(inputs)
  axis_shapes = set(a.shape[axis] for a in arrays)
  if len(axis_shapes) != 1:
    raise ValueError(f'Arrays must have equal sized axis but got {axis_shapes}')
  axis_shape, = axis_shapes
  splits = [jnp.split(a, axis_shape, axis=axis) for a in arrays]
  if not keep_dims:
    splits = jax.tree_map(lambda a: jnp.squeeze(a, axis), splits)
  splits = zip(*splits)
  return tuple(jax.tree_unflatten(tree_def, leaves) for leaves in splits)


def concat_along_axis(pytrees, axis):
  """Concatenates `pytrees` along `axis`."""
  concat_leaves_fn = lambda *args: jnp.concatenate(args, axis)
  return jax.tree_multimap(concat_leaves_fn, *pytrees)


def block_reduce(
    array: Array,
    block_size: Tuple[int, ...],
    reduction_fn: Callable[[Array], Array]
) -> Array:
  """Breaks `array` into `block_size` pieces and applies `f` to each.

  This function is equivalent to `scikit-image.measure.block_reduce`:
  https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce

  Args:
    array: an array.
    block_size: the size of the blocks on which the reduction is performed.
      Must evenly divide `array.shape`.
    reduction_fn: a reduction function that will be applied to each block of
      size `block_size`.
  Returns:
    The result of applying `f` to each block of size `block_size`.
  """
  new_shape = []
  for b, s in zip(block_size, array.shape):
    multiple, residual = divmod(s, b)
    if residual != 0:
      raise ValueError('`block_size` must divide `array.shape`;'
                       f'got {block_size}, {array.shape}.')
    new_shape += [multiple, b]
  multiple_axis_reduction_fn = reduction_fn
  for j in reversed(range(array.ndim)):
    multiple_axis_reduction_fn = jax.vmap(multiple_axis_reduction_fn, j)
  return multiple_axis_reduction_fn(array.reshape(new_shape))


def laplacian_matrix(size: int, step: float) -> np.ndarray:
  """Create a matrix representing the Laplacian operator in 1D."""
  column = np.zeros(size)
  column[0] = - 2 / step ** 2
  column[1] = column[-1] = 1 / step ** 2
  return scipy.linalg.circulant(column)


def unstack(array, axis):
  """Returns a tuple of slices of `array` along axis `axis`."""
  squeeze_fn = lambda x: jnp.squeeze(x, axis=axis)
  return tuple(squeeze_fn(x) for x in jnp.split(array, array.shape[axis], axis))


def gram_schmidt_qr(
    matrix: Array,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST
) -> Tuple[Array, Array]:
  """Computes QR decomposition using gramm-schmidt orthogonalization.

  This algorithm is suitable for tall matrices with very few columns. This
  method is more memory efficient compared to `jnp.linalg.qr`, but is less
  numerically stable, especially for matrices with many columns.

  Args:
    matrix: 2D array representing the matrix to be decomposed into orthogonal
      and upper triangular.
    precision: numerical precision for matrix multplication. Only relevant on
      TPUs.

  Returns:
    tuple of matrix Q whose columns are orthonormal and R that is upper
    triangular.
  """

  def orthogonalize(vector, others):
    """Returns the orthogonal component of `vector` with respect to `others`."""
    if not others:
      return vector / jnp.linalg.norm(vector)
    orthogonalize_step = lambda c, x: tuple([c - jnp.dot(c, x) * x, None])
    vector, _ = jax.lax.scan(orthogonalize_step, vector, jnp.stack(others))
    return vector / jnp.linalg.norm(vector)

  num_columns = matrix.shape[1]
  columns = unstack(matrix, axis=1)
  q_columns = []
  r_rows = []
  for vec_index, column in enumerate(columns):
    next_q_column = orthogonalize(column, q_columns)
    r_rows.append(jnp.asarray([
        jnp.dot(columns[i], next_q_column) if i >= vec_index else 0.
        for i in range(num_columns)]))
    q_columns.append(next_q_column)
  q = jnp.stack(q_columns, axis=1)
  r = jnp.stack(r_rows)
  # permute q columns to make entries of r on the diagonal positive.
  d = jnp.diag(jnp.sign(jnp.diagonal(r)))
  q = jnp.matmul(q, d, precision=precision)
  r = jnp.matmul(d, r, precision=precision)
  return q, r
