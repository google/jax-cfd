"""Utility functions for layers.py."""
import enum
import functools
import itertools
import math
from typing import Iterator, Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
from jax_cfd.ml import tiling
import numpy as np
import scipy

Array = Union[np.ndarray, jnp.DeviceArray]


class Method(enum.Enum):
  """Discretization method."""
  FINITE_DIFFERENCE = 1
  FINITE_VOLUME = 2


def _kronecker_product(arrays: Sequence[np.ndarray]) -> np.ndarray:
  """Returns a kronecker product of a sequence of arrays."""
  return functools.reduce(np.kron, arrays)


def _exponents_up_to_degree(
    degree: int,
    num_dimensions: int
) -> Iterator[Tuple[int, ...]]:
  """Generate all exponents up to given degree.

  Args:
    degree: a non-negative integer representing the maximum degree.
    num_dimensions: a non-negative integer representing the number of
      dimensions.

  Yields:
    An iterator over all tuples of non-negative integers of length
    `num_dimensions`, whose sum is at most `degree`.

  Example:
    For degree=2 and num_dimensions=2, this iterates through
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)].
  """
  if num_dimensions == 0:
    yield tuple()
  else:
    for d in range(degree + 1):
      for exponents in _exponents_up_to_degree(degree - d, num_dimensions - 1):
        yield (d,) + exponents


def polynomial_accuracy_constraints(
    stencils: Sequence[np.ndarray],
    method: Method,
    derivative_orders: Sequence[int],
    accuracy_order: int,
    grid_step: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Setup a linear equation A @ c = b for finite difference coefficients.

  Elements are returned in row-major order, e.g., if two stencils of length 2
  are provided, then coefficients for the stencil is organized as:
  [s_00, s_01, s_10, s_11], where s_ij corresponds to a value at
  (stencil_1[i], stencil_2[j]). The returned constraints assume that
  coefficients aim to appoximate the derivative of `derivative_order` at (0, 0).

  Example:
    For stencils = [np.array([-0.5, 0.5])] * 2, coefficients approximate the
    derivative at point (0., 0.) using values located at (-0.5, -0.5),
    (-0.5, 0.5), (0.5, 0.5), (0.5, 0.5)

  Args:
    stencils: list of arrays giving 1D stencils in each direction.
    method: discretization method (i.e., finite volumes or finite differences).
    derivative_orders: integer derivative orders to approximate in each grid
      direction.
    accuracy_order: minimum accuracy orders for the solution in each grid
      direction.
    grid_step: spacing between grid cells.

  Returns:
    Tuple of arrays `(A, b)` where `A` is 2D and `b` is 1D providing linear
    constraints. Any vector of finite difference coefficients `c` such that
    `A @ c = b` satisfies the requested accuracy order. The matrix `A` is
    guaranteed not to have more rows than columns.

  Raises:
    ValueError: if the linear constraints are not satisfiable.

  References:
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Fornberg, Bengt (1988), "Generation of Finite Difference Formulas on
      Arbitrarily Spaced Grids", Mathematics of Computation, 51 (184): 699-706,
      doi:10.1090/S0025-5718-1988-0935077-0, ISSN 0025-5718.
  """
  if len(stencils) != len(derivative_orders):
    raise ValueError('mismatched lengths for stencils and derivative_orders')

  if accuracy_order < 1:
    raise ValueError('cannot compute constriants with non-positive '
                     'accuracy_order: {}'.format(accuracy_order))

  all_constraints = {}

  # See http://g3doc/third_party/py/datadrivenpdes/g3doc/polynomials.md.
  num_dimensions = len(stencils)
  max_degree = accuracy_order + sum(derivative_orders) - 1
  for exponents in _exponents_up_to_degree(max_degree, num_dimensions):
    # build linear constraints for a single polynomial term:
    # \prod_i {x_i}^{m_i}
    lhs_terms = []
    rhs_terms = []
    exp_stencil_derivative = zip(exponents, stencils, derivative_orders)
    for exponent, stencil, derivative_order in exp_stencil_derivative:

      if method is Method.FINITE_VOLUME:
        if grid_step is None:
          raise ValueError('grid_step is required for finite volumes')
        # average value of x**m over a centered grid cell
        lhs_terms.append(
            1 / grid_step * ((stencil + grid_step / 2)**(exponent + 1) -
                             (stencil - grid_step / 2)**(exponent + 1)) /
            (exponent + 1))

      elif method is Method.FINITE_DIFFERENCE:
        lhs_terms.append(stencil**exponent)
      else:
        raise ValueError('unexpected method: {}'.format(method))

      if exponent == derivative_order:
        # we get a factor of m! for m-th order derivative in each direction
        rhs_term = scipy.special.factorial(exponent)
      else:
        rhs_term = 0
      rhs_terms.append(rhs_term)

    lhs = tuple(_kronecker_product(lhs_terms))
    rhs = np.prod(rhs_terms)

    if lhs in all_constraints and all_constraints[lhs] != rhs:
      raise ValueError('conflicting constraints')
    all_constraints[lhs] = rhs

  lhs_rows, rhs_rows = zip(*sorted(all_constraints.items()))
  A = np.array(lhs_rows)  # pylint: disable=invalid-name
  b = np.array(rhs_rows)
  return A, b


def _high_order_coefficients_1d(
    stencil: np.ndarray,
    method: Method,
    derivative_order: int,
    grid_step: Optional[float] = None,
) -> np.ndarray:
  """Calculate highest-order coefficients that appoximate `derivative_order`.

  Args:
    stencil: 1D array representing locations of the stencil's cells.
    method: discretization method (i.e., finite volumes or finite differences).
    derivative_order: derivative order being approximated.
    grid_step: grid step size.

  Returns:
    Array representing stencil coefficients that approximate `derivative_order`
    spatial derivative on provided `stencil`.
  """
  # Use the highest order accuracy we can ensure in general. (In some cases,
  # e.g., centered finite differences, this solution actually has higher order
  # accuracy.)
  accuracy_order = stencil.size - derivative_order
  A, b = polynomial_accuracy_constraints(  # pylint: disable=invalid-name
      [stencil], method, [derivative_order], accuracy_order, grid_step)
  return np.linalg.solve(A, b)


def polynomial_accuracy_coefficients(
    stencils: Sequence[np.ndarray],
    method: Method,
    derivative_orders: Sequence[int],
    accuracy_order: Optional[int] = None,
    grid_step: Optional[float] = None,
) -> np.ndarray:
  """Calculate standard finite volume coefficients.

  These coefficients are constructed by taking an outer product of coefficients
  along each dimension independently. The resulting coefficients have *at least*
  the requested accuracy order. The derivative is approximated at `0.` position
  along each stencil.

  Args:
    stencils: sequence of 1d stencils, one per grid dimension.
    method: discretization method (i.e., finite volumes or finite differences).
    derivative_orders: integer derivative orders to approximate, per grid
      dimension.
    accuracy_order: accuracy order for the solution. By default, the highest
      possible accuracy is used in each direction.
    grid_step: spacing between grid cells. Required if calculating a finite
      volume stencil.

  Returns:
    NumPy array with one-dimension per stencil giving first order finite
    difference coefficients on the grid.
  """
  slices = []
  sizes = []
  all_coefficients = []
  for stencil, derivative_order in zip(stencils, derivative_orders):
    if accuracy_order is None:
      excess = 0
    else:
      excess = stencil.size - derivative_order - accuracy_order
    start = excess // 2
    stop = stencil.size - excess // 2
    slice_ = slice(start, stop)
    axis_coefficients = _high_order_coefficients_1d(
        stencil[slice_], method, derivative_order, grid_step)

    slices.append(slice_)
    sizes.append(stencil[slice_].size)
    all_coefficients.append(axis_coefficients)

  result = np.zeros(tuple(stencil.size for stencil in stencils))
  result[tuple(slices)] = _kronecker_product(all_coefficients).reshape(sizes)
  return result


def get_roll_and_shift(
    input_offset: Tuple[float, ...],
    target_offset: Tuple[float, ...]
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
  """Decomposes delta as integer `roll` and positive fractional `shift`."""
  delta = [t - i for t, i in zip(target_offset, input_offset)]
  roll = tuple(-math.floor(d) for d in delta)
  shift = tuple(d + r for d, r in zip(delta, roll))
  return roll, shift


def get_stencils(
    stencil_sizes: Tuple[int, ...],
    offset: Tuple[float, ...],
    steps: Tuple[float, ...]
) -> Tuple[np.ndarray]:
  """Computes stencils locations.

  Generates stencils such that the target offset is placed at (0.,)*ndims. This
  is needed to obtain correct polynomial constraints. This approach places
  equal number of cells on the right and left for half-cell difference for even
  stencils and for same offset for odd stencils. Otherwise adds an extra cell on
  one of the sides. The order of the returned `stencil_shifts` is row-major,
  i.e. an outer product of shifts along axes.

  Args:
    stencil_sizes: sizes of 1d stencils along each directions.
    offset: the target offset relative to the current offset i.e
      `target_offset - input_offset`.
    steps: distances between adjacent grid cells.

  Returns:
    stencils: list of 1d stencils representing locations of the centers of cells
      of inputs array.
  """
  stencils = []
  for size, o, step, in zip(stencil_sizes, offset, steps):
    left = -((size - 1) // 2)
    shifts = range(left, left + size)
    stencils.append(np.array([(-o + s) * step for s in shifts]))
  return tuple(stencils)


def _get_padding(
    kernel_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, int], ...]:
  """Returns the padding for convolutions used in `extract_patches`.

  Note that the padding here is "flipped" compared to the padding used in
  `PeriodicConvGeneral`.

  Args:
    kernel_shape: the shape of the convolutional kernel.

  Returns:
    A tuple of pairs of ints. Each pair indicates the padding that should be
    added before and after the array for a periodic convolution with shape
    `kernel_shape`.
  """
  # TODO(jamieas): use this function to compute padding for
  # `PeriodicConvGeneral`.
  padding = []
  for kernel_size in kernel_shape[:-2]:
    pad_right = kernel_size // 2
    pad_left = kernel_size - pad_right - 1
    padding.append((pad_left, pad_right))
  return tuple(padding)


_DIMENSION_NUMBERS = {
    1: ('NWC', 'WIO', 'NWC'),
    2: ('NHWC', 'HWIO', 'NHWC'),
    3: ('NHWDC', 'HWDIO', 'NHWDC'),
}


def periodic_convolution(
    x: Array,
    kernel: Array,
    tile_layout: Optional[Tuple[int, ...]] = None,
    precision: lax.Precision = lax.Precision.HIGHEST,
) -> Array:
  """Applies a periodic convolution."""
  num_spatial_dims = kernel.ndim - 2
  padding = _get_padding(kernel.shape)
  strides = [1] * num_spatial_dims
  dimension_numbers = _DIMENSION_NUMBERS[num_spatial_dims]
  conv = functools.partial(jax.lax.conv_general_dilated,
                           rhs=kernel,
                           window_strides=strides,
                           padding='VALID',
                           dimension_numbers=dimension_numbers,
                           precision=precision)
  return tiling.apply_convolution(conv, x, layout=tile_layout, padding=padding)


# Caching the result of _patch_kernel() ensures that only one constant value is
# used as a side-input into JAX's jit/pmap. This ensures that XLA's Common
# Subexpression Elimination (CSE) pass can consolidate calls to extract patches
# on the same array.
@functools.lru_cache()
def _patch_kernel(
    patch_shape: Tuple[int, ...],
    dtype: np.dtype = np.float32
) -> np.ndarray:
  """Returns a convolutional kernel that extracts patches."""
  patch_size = np.prod(patch_shape)
  kernel_2d = np.eye(patch_size, dtype=dtype)
  kernel_shape = (patch_size, 1) + patch_shape
  kernel_nd = kernel_2d.reshape(kernel_shape)
  return np.moveaxis(kernel_nd, (0, 1), (-1, -2))


@functools.partial(jax.jit, static_argnums=(1,))
def _extract_patches_roll(
    x: Array,
    patch_shape: Tuple[int, ...]
) -> Array:
  """Extract patches of the given shape using a vmapped `roll` operation."""
  # Computes shifts required for the given `patch_shape`.
  x = jnp.squeeze(x, -1)
  shifts = []
  for size in patch_shape:
    shifts.append(range(-size // 2 + 1, size // 2 + 1))
  rolls = -np.stack(tuple(itertools.product(*shifts)))
  out_axis = x.ndim
  roll_axes = range(out_axis)
  in_axes = (None, 0, None)
  return jax.vmap(jnp.roll, in_axes, out_axis)(x, rolls, roll_axes)


@functools.partial(jax.jit, static_argnums=(1, 2))
def _extract_patches_conv(
    x: Array,
    patch_shape: Tuple[int, ...],
    tile_layout: Optional[Tuple[int, ...]],
) -> Array:
  """Extract patches of the given shape using a tiled convolution."""
  kernel = _patch_kernel(patch_shape, dtype=x.dtype)
  # the kernel can be represented exactly in bfloat16
  precision = (lax.Precision.HIGHEST, lax.Precision.DEFAULT)
  return periodic_convolution(x, kernel, tile_layout, precision=precision)


def extract_patches(
    x: Array,
    patch_shape: Tuple[int, ...],
    method: str = 'roll',
    tile_layout: Optional[Tuple[int, ...]] = None):
  """Extracts patches of given shape, stacks them along the channel dimension.

  For example,

  ```
  x = [[0,  1,  2,  3],
       [4,  5,  6,  7],
       [8,  9,  10, 11],
       [12, 13, 14, 15]]
  x = np.expand_dims(x, -1)  # Add 'channel' dimension.

  y = extract_patches(x, [3, 3])

  y[0, 0]  # [15, 12, 13, 3, 0, 1, 7, 4, 5]
  y[1, 1]  # [0, 1, 2, 4, 5, 6, 7, 8, 9]

  z = extract_patches(x, [2, 2])

  z[0, 0]  # [0, 1, 4, 5]
  z[2, 2]  # [10, 11, 14, 15]
  ```

  In particular, for even patch sizes, `extract_patches` includes extra values
  with _higher_ indices.

  Args:
    x: the array with shape [d0, ..., dk] from which we will extract patches.
    patch_shape: a tuple (p0, ..., pk) describing the shape of the patches to
      extract.
    method: determines which method is used for extracting patches. Must be
      either 'roll' or 'conv'.
    tile_layout: an optional tuple (t0, ..., tk) describing the tiling that will
      be used to perform the convolutions that extract patches. If `None`, then
      no tiling is performed. If `method == 'roll'`, this argument has not
      effect.

  Returns:
    An array of shape [d0, ..., dk, c] where `c = prod(patch_shape)`.
  """
  # TODO(jamieas): consider removing the 'roll' method once the convolutional
  # one has been optimized.
  if method == 'roll':
    return _extract_patches_roll(x, patch_shape)
  elif method == 'conv':
    return _extract_patches_conv(x, patch_shape, tile_layout)
  else:
    raise ValueError(f'Unknown `method` passed to `extract_patches`: {method}.')


def fused_extract_patches(
    x: Array,
    patch_shapes: Sequence[Tuple[int, ...]],
    tile_layout: Optional[Tuple[int, ...]] = None,
):
  kernel = np.concatenate(
      [_patch_kernel(s, dtype=x.dtype) for s in patch_shapes], axis=-1)
  # the kernel can be represented exactly in bfloat16
  precision = (lax.Precision.HIGHEST, lax.Precision.DEFAULT)
  return periodic_convolution(x, kernel, tile_layout, precision=precision)


# TODO(dkochkov) consider alternative ops for better efficiency on MXU.
def apply_coefficients(coefficients, stencil_values):
  """Constructs array as a weighted sum of `stencil_values`."""
  return jnp.sum(coefficients * stencil_values, axis=-1, keepdims=True)
