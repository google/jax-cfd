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

from typing import Any, Callable, List, Tuple, Union

import jax
import jax.numpy as jnp
from jax_ib.base import boundaries
from jax_ib.base import grids
import numpy as np
import scipy.linalg

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any
Array = Union[np.ndarray, jax.Array]


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
  return jax.tree_map(concat_leaves_fn, *pytrees)


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
  """Create 1D Laplacian operator matrix, with periodic BC."""
  column = np.zeros(size)
  column[0] = -2 / step**2
  column[1] = column[-1] = 1 / step**2
  return scipy.linalg.circulant(column)

def laplacian_matrix_neumann(size: int, step: float) -> np.ndarray:
  """Create 1D Laplacian operator matrix, with homogeneous Neumann BC."""
  column = np.zeros(size)
  column[0] = -2 / step ** 2
  column[1] = 1 / step ** 2
  matrix = scipy.linalg.toeplitz(column)
  matrix[0, 0] = matrix[-1, -1] = -1 / step**2
  return matrix

def _laplacian_boundary_dirichlet_cell_centered(laplacians: List[Array],
                                                grid: grids.Grid, axis: int,
                                                side: str) -> None:
  """Converts 1d laplacian matrix to satisfy dirichlet homogeneous bc.

  laplacians[i] contains a 3 point stencil matrix L that approximates
  d^2/dx_i^2.
  For detailed documentation on laplacians input type see
  array_utils.laplacian_matrix.
  The default return of array_utils.laplacian_matrix makes a matrix for
  periodic boundary. For dirichlet boundary, the correct equation is
  L(u_interior) = rhs_interior and BL_boundary = u_fixed_boundary. So
  laplacian_boundary_dirichlet restricts the matrix L to
  interior points only.

  This function assumes RHS has cell-centered offset.
  Args:
    laplacians: list of 1d laplacians
    grid: grid object
    axis: axis along which to impose dirichlet bc.
    side: lower or upper side to assign boundary to.

  Returns:
    updated list of 1d laplacians.
  """
  # This function assumes homogeneous boundary, in which case if the offset
  # is 0.5 away from the wall, the ghost cell value u[0] = -u[1]. So the
  # 3 point stencil [1 -2 1] * [u[0] u[1] u[2]] = -3 u[1] + u[2].
  if side == 'lower':
    laplacians[axis][0, 0] = laplacians[axis][0, 0] - 1 / grid.step[axis]**2
  else:
    laplacians[axis][-1, -1] = laplacians[axis][-1, -1] - 1 / grid.step[axis]**2
  # deletes corner dependencies on the "looped-around" part.
  # this should be done irrespective of which side, since one boundary cannot
  # be periodic while the other is.
  laplacians[axis][0, -1] = 0.0
  laplacians[axis][-1, 0] = 0.0
  return


def _laplacian_boundary_neumann_cell_centered(laplacians: List[Array],
                                              grid: grids.Grid, axis: int,
                                              side: str) -> None:
  """Converts 1d laplacian matrix to satisfy neumann homogeneous bc.

  This function assumes the RHS will have a cell-centered offset.
  Neumann boundaries are not defined for edge-aligned offsets elsewhere in the
  code.

  Args:
    laplacians: list of 1d laplacians
    grid: grid object
    axis: axis along which to impose dirichlet bc.
    side: which boundary side to convert to neumann homogeneous bc.

  Returns:
    updated list of 1d laplacians.
  """
  if side == 'lower':
    laplacians[axis][0, 0] = laplacians[axis][0, 0] + 1 / grid.step[axis]**2
  else:
    laplacians[axis][-1, -1] = laplacians[axis][-1, -1] + 1 / grid.step[axis]**2
  # deletes corner dependencies on the "looped-around" part.
  # this should be done irrespective of which side, since one boundary cannot
  # be periodic while the other is.
  laplacians[axis][0, -1] = 0.0
  laplacians[axis][-1, 0] = 0.0
  return


def laplacian_matrix_w_boundaries(
    grid: grids.Grid,
    offset: Tuple[float, ...],
    bc: boundaries.BoundaryConditions,
) -> List[Array]:
  """Returns 1d laplacians that satisfy boundary conditions bc on grid.

  Given grid, offset and boundary conditions, returns a list of 1 laplacians
  (one along each axis).

  Currently, only homogeneous or periodic boundary conditions are supported.

  Args:
    grid: The grid used to construct the laplacian.
    offset: The offset of the variable on which laplacian acts.
    bc: the boundary condition of the variable on which the laplacian acts.

  Returns:
    A list of 1d laplacians.
  """
  if not isinstance(bc, boundaries.ConstantBoundaryConditions):
    raise NotImplementedError(
        f'Explicit laplacians are not implemented for {bc}.')
  laplacians = list(map(laplacian_matrix, grid.shape, grid.step))
  for axis in range(grid.ndim):
    if np.isclose(offset[axis], 0.5):
      for i, side in enumerate(['lower', 'upper']):  # lower and upper boundary
        if bc.types[axis][i] == boundaries.BCType.NEUMANN:
          _laplacian_boundary_neumann_cell_centered(
              laplacians, grid, axis, side)
        elif bc.types[axis][i] == boundaries.BCType.DIRICHLET:
          _laplacian_boundary_dirichlet_cell_centered(
              laplacians, grid, axis, side)
    if np.isclose(offset[axis] % 1, 0.):
      if bc.types[axis][0] == boundaries.BCType.DIRICHLET and bc.types[
          axis][1] == boundaries.BCType.DIRICHLET:
        # This function assumes homogeneous boundary and acts on the interior.
        # Thus, the laplacian can be cut off past the edge.
        # The interior grid has one fewer grid cell than the actual grid, so
        # the size of the laplacian should be reduced.
        laplacians[axis] = laplacians[axis][:-1, :-1]
      elif boundaries.BCType.NEUMANN in bc.types[axis]:
        raise NotImplementedError(
            'edge-aligned Neumann boundaries are not implemented.')
  return laplacians


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


def interp1d(  # pytype: disable=annotation-type-mismatch  # jnp-type
    x: Array,
    y: Array,
    axis: int = -1,
    fill_value: Union[str, Array] = jnp.nan,
    assume_sorted: bool = True,
) -> Callable[[Array], jax.Array]:
  """Build an interpolation function to approximate `y = f(x)`.

  x and y are arrays of values used to approximate some function f: y = f(x).
  This returns a function that uses linear interpolation to approximate f
  evaluated at new points.

  ```
  x = jnp.linspace(0, 10)
  y = jnp.sin(x)
  f = interp1d(x, y)

  x_new = 1.23
  f(x_new)
  ==> Approximately sin(1.23).

  x_new = ...  # Shape (4, 5) array
  f(x_new)
  ==> Shape (4, 5) array, approximating sin(x_new).
  ```

  Args:
    x: Length N 1-D array of values.
    y: Shape (..., N, ...) array of values corresponding to f(x).
    axis: Specifies the axis of y along which to interpolate. Interpolation
      defaults to the last axis of y.
    fill_value: Scalar array or string. If array, this value will be used to
      fill in for requested points outside of the data range. If not provided,
      then the default is NaN. If "extrapolate", then linear extrapolation is
      used.  If "constant_extrapolate", then the function is extended as a
      constant.
    assume_sorted: Whether to assume x is sorted. If True, x must be
      monotonically increasing. If False, this function sorts x and reorders
      y appropriately.

  Returns:
    Callable mapping array x_new to values y_new, where
      y_new.shape = y.shape[:axis] + x_new.shape + y.shape[axis + 1:]
  """
  allowed_fill_value_strs = {'constant_extrapolate', 'extrapolate'}
  if isinstance(fill_value, str):
    if fill_value not in allowed_fill_value_strs:
      raise ValueError(
          f'`fill_value` "{fill_value}" not in {allowed_fill_value_strs}')
  else:
    fill_value = np.asarray(fill_value)
    if fill_value.ndim > 0:
      raise ValueError(f'Only scalar `fill_value` supported. '
                       f'Found shape: {fill_value.shape}')

  x = jnp.asarray(x)
  if x.ndim != 1:
    raise ValueError(f'Expected `x` to be 1D. Found shape {x.shape}')
  if x.size < 2:
    raise ValueError(f'`x` must have at least 2 entries. Found shape {x.shape}')
  n_x = x.shape[0]
  if not assume_sorted:
    ind = jnp.argsort(x)
    x = x[ind]
    y = jnp.take(y, ind, axis=axis)

  y = jnp.asarray(y)
  if y.ndim < 1:
    raise ValueError(f'Expected `y` to have rank >= 1. Found shape {y.shape}')

  if x.size != y.shape[axis]:
    raise ValueError(
        f'x and y arrays must be equal in length along interpolation axis. '
        f'Found x.shape={x.shape} and y.shape={y.shape} and axis={axis}')

  axis = _normalize_axis(axis, ndim=y.ndim)

  def interp_func(x_new: jax.Array) -> jax.Array:
    """Implementation of the interpolation function."""
    x_new = jnp.asarray(x_new)

    # We will flatten x_new, then interpolate, then reshape the output.
    x_new_shape = x_new.shape
    x_new = jnp.ravel(x_new)

    # This construction of indices ensures that below_idx < above_idx, even at
    # x_new = x[i] exactly for some i.
    x_new_clipped = jnp.clip(x_new, np.min(x), np.max(x))
    above_idx = jnp.minimum(n_x - 1,
                            jnp.searchsorted(x, x_new_clipped, side='right'))
    below_idx = jnp.maximum(0, above_idx - 1)

    def expand(array):
      """Add singletons to rightmost dims of `array` so it bcasts with y."""
      array = jnp.asarray(array)
      return jnp.reshape(array, array.shape + (1,) * (y.ndim - axis - 1))

    x_above = jnp.take(x, above_idx)
    x_below = jnp.take(x, below_idx)
    y_above = jnp.take(y, above_idx, axis=axis)
    y_below = jnp.take(y, below_idx, axis=axis)
    slope = (y_above - y_below) / expand(x_above - x_below)

    if isinstance(fill_value, str) and fill_value == 'extrapolate':
      delta_x = expand(x_new - x_below)
      y_new = y_below + delta_x * slope
    elif isinstance(fill_value, str) and fill_value == 'constant_extrapolate':
      delta_x = expand(x_new_clipped - x_below)
      y_new = y_below + delta_x * slope
    else:  # Else fill_value is an Array.
      delta_x = expand(x_new - x_below)
      fill_value_ = expand(fill_value)
      y_new = y_below + delta_x * slope
      y_new = jnp.where(
          (delta_x < 0) | (delta_x > expand(x_above - x_below)),
          fill_value_, y_new)
    return jnp.reshape(
        y_new, y_new.shape[:axis] + x_new_shape + y_new.shape[axis + 1:])

  return interp_func
