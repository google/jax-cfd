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

"""Functions for interpolating `GridVariables`s."""

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import boundaries
from jax_cfd.base import grids
import numpy as np

Array = Union[np.ndarray, jnp.DeviceArray]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = Callable[
    [GridVariable, Tuple[float, ...], GridVariableVector, float],
    GridVariable]
FluxLimiter = Callable[[grids.Array], grids.Array]


def _linear_along_axis(c: GridVariable,
                       offset: float,
                       axis: int) -> GridVariable:
  """Linear interpolation of `c` to `offset` along a single specified `axis`."""
  offset_delta = offset - c.offset[axis]

  # If offsets are the same, `c` is unchanged.
  if offset_delta == 0:
    return c

  new_offset = tuple(offset if j == axis else o
                     for j, o in enumerate(c.offset))

  # If offsets differ by an integer, we can just shift `c`.
  if int(offset_delta) == offset_delta:
    return grids.GridVariable(
        array=grids.GridArray(data=c.shift(int(offset_delta), axis).data,
                              offset=new_offset,
                              grid=c.grid),
        bc=c.bc)

  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  floor_weight = ceil - offset_delta
  ceil_weight = 1. - floor_weight
  data = (floor_weight * c.shift(floor, axis).data +
          ceil_weight * c.shift(ceil, axis).data)
  return grids.GridVariable(
      array=grids.GridArray(data, new_offset, c.grid), bc=c.bc)


def linear(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None
) -> grids.GridVariable:
  """Multi-linear interpolation of `c` to `offset`.

  Args:
    c: quantitity to be interpolated.
    offset: offset to which we will interpolate `c`. Must have the same length
      as `c.offset`.
    v: velocity field. Not used.
    dt: size of the time step. Not used.

  Returns:
    An `GridArray` containing the values of `c` after linear interpolation
    to `offset`. The returned value will have offset equal to `offset`.
  """
  del v, dt  # unused
  if len(offset) != len(c.offset):
    raise ValueError('`c.offset` and `offset` must have the same length;'
                     f'got {c.offset} and {offset}.')
  interpolated = c
  for a, o in enumerate(offset):
    interpolated = _linear_along_axis(interpolated, offset=o, axis=a)
  return interpolated


def upwind(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: GridVariableVector,
    dt: Optional[float] = None
) -> GridVariable:
  """Upwind interpolation of `c` to `offset` based on velocity field `v`.

  Interpolates values of `c` to `offset` in two steps:
  1) Identifies the axis along which `c` is interpolated. (must be single axis)
  2) For positive (negative) velocity along interpolation axis uses value from
     the previous (next) cell along that axis correspondingly.

  Args:
    c: quantitity to be interpolated.
    offset: offset to which `c` will be interpolated. Must have the same
      length as `c.offset` and differ in at most one entry.
    v: velocity field with offsets at faces of `c`. One of the components
      must have the same offset as `offset`.
    dt: size of the time step. Not used.

  Returns:
    A `GridVariable` that containins the values of `c` after interpolation to
    `offset`.

  Raises:
    InconsistentOffsetError: if `offset` and `c.offset` differ in more than one
    entry.
  """
  del dt  # unused
  if c.offset == offset: return c
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if current != target
  )
  if len(interpolation_axes) != 1:
    raise grids.InconsistentOffsetError(
        f'for upwind interpolation `c.offset` and `offset` must differ at most '
        f'in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  u = v[axis]
  offset_delta = u.offset[axis] - c.offset[axis]

  # If offsets differ by an integer, we can just shift `c`.
  if int(offset_delta) == offset_delta:
    return grids.GridVariable(
        array=grids.GridArray(data=c.shift(int(offset_delta), axis).data,
                              offset=offset,
                              grid=grids.consistent_grid(c, u)),
        bc=c.bc)

  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  array = grids.applied(jnp.where)(
      u.array > 0, c.shift(floor, axis).data, c.shift(ceil, axis).data
  )
  grid = grids.consistent_grid(c, u)
  return grids.GridVariable(
      array=grids.GridArray(array.data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


def lax_wendroff(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None
) -> GridVariable:
  """Lax_Wendroff interpolation of `c` to `offset` based on velocity field `v`.

  Interpolates values of `c` to `offset` in two steps:
  1) Identifies the axis along which `c` is interpolated. (must be single axis)
  2) For positive (negative) velocity along interpolation axis uses value from
     the previous (next) cell along that axis plus a correction originating
     from expansion of the solution at the half step-size.

  This method is second order accurate with fixed coefficients and hence can't
  be monotonic due to Godunov's theorem.
  https://en.wikipedia.org/wiki/Godunov%27s_theorem

  Lax-Wendroff method can be used to form monotonic schemes when augmented with
  a flux limiter. See https://en.wikipedia.org/wiki/Flux_limiter

  Args:
    c: quantitity to be interpolated.
    offset: offset to which we will interpolate `c`. Must have the same
      length as `c.offset` and differ in at most one entry.
    v: velocity field with offsets at faces of `c`. One of the components must
      have the same offset as `offset`.
    dt: size of the time step. Not used.

  Returns:
    A `GridVariable` that containins the values of `c` after interpolation to
    `offset`.
  Raises:
    InconsistentOffsetError: if `offset` and `c.offset` differ in more than one
    entry.
  """
  # TODO(dkochkov) add a function to compute interpolation axis.
  if c.offset == offset: return c
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if current != target
  )
  if len(interpolation_axes) != 1:
    raise grids.InconsistentOffsetError(
        f'for Lax-Wendroff interpolation `c.offset` and `offset` must differ at'
        f' most in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  u = v[axis]
  offset_delta = u.offset[axis] - c.offset[axis]
  floor = int(np.floor(offset_delta))  # used for positive velocity
  ceil = int(np.ceil(offset_delta))  # used for negative velocity
  grid = grids.consistent_grid(c, u)
  courant_numbers = (dt / grid.step[axis]) * u.data
  positive_u_case = (
      c.shift(floor, axis).data + 0.5 * (1 - courant_numbers) *
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
  negative_u_case = (
      c.shift(ceil, axis).data - 0.5 * (1 + courant_numbers) *
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
  array = grids.where(u.array > 0, positive_u_case, negative_u_case)
  grid = grids.consistent_grid(c, u)
  return grids.GridVariable(
      array=grids.GridArray(array.data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


def safe_div(x, y, default_numerator=1):
  """Safe division of `Array`'s."""
  return x / jnp.where(y != 0, y, default_numerator)


def van_leer_limiter(r):
  """Van-leer flux limiter."""
  return jnp.where(r > 0, safe_div(2 * r, 1 + r), 0.0)


def apply_tvd_limiter(
    interpolation_fn: InterpolationFn,
    limiter: FluxLimiter = van_leer_limiter
) -> InterpolationFn:
  """Combines low and high accuracy interpolators to get TVD method.

  Generates high accuracy interpolator by combining stable lwo accuracy `upwind`
  interpolation and high accuracy (but not guaranteed to be stable)
  `interpolation_fn` to obtain stable higher order method. This implementation
  follows the procedure outined in:
  http://www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2012/Chapter_4.pdf

  Args:
    interpolation_fn: higher order interpolation methods. Must follow the same
      interface as other interpolation methods (take `c`, `offset`, `grid`, `v`
      and `dt` arguments and return value of `c` at offset `offset`).
    limiter: flux limiter function that evaluates the portion of the correction
      (high_accuracy - low_accuracy) to add to low_accuracy solution based on
      the ratio of the consequtive gradients. Takes array as input and return
      array of weights. For more details see:
      https://en.wikipedia.org/wiki/Flux_limiter

  Returns:
    Interpolation method that uses a combination of high and low order methods
    to produce monotonic interpolation method.
  """

  def tvd_interpolation(
      c: GridVariable,
      offset: Tuple[float, ...],
      v: GridVariableVector,
      dt: float,
  ) -> GridVariable:
    """Interpolated `c` to offset `offset`."""
    for axis, axis_offset in enumerate(offset):
      interpolation_offset = tuple([
          c_offset if i != axis else axis_offset
          for i, c_offset in enumerate(c.offset)
      ])
      if interpolation_offset != c.offset:
        if interpolation_offset[axis] - c.offset[axis] != 0.5:
          raise NotImplementedError('tvd_interpolation only supports forward '
                                    'interpolation to control volume faces.')
        c_low = upwind(c, offset, v, dt)
        c_high = interpolation_fn(c, offset, v, dt)

        # because we are interpolating to the right we are using 2 points ahead
        # and 2 points behind: `c`, `c_left`.
        c_left = c.shift(-1, axis)
        c_right = c.shift(1, axis)
        c_next_right = c.shift(2, axis)
        # Velocities of different sign are evaluated with limiters at different
        # points. See equations (4.34) -- (4.39) from the reference above.
        positive_u_r = safe_div(c.data - c_left.data, c_right.data - c.data)
        negative_u_r = safe_div(c_next_right.data - c_right.data,
                                c_right.data - c.data)
        positive_u_phi = grids.GridArray(
            limiter(positive_u_r), c_low.offset, c.grid)
        negative_u_phi = grids.GridArray(
            limiter(negative_u_r), c_low.offset, c.grid)
        u = v[axis]
        phi = grids.applied(jnp.where)(
            u.array > 0, positive_u_phi, negative_u_phi)
        c_interpolated = c_low.array - (c_low.array - c_high.array) * phi
        c = grids.GridVariable(
            grids.GridArray(c_interpolated.data, interpolation_offset, c.grid),
            c.bc)
    return c

  return tvd_interpolation


# TODO(pnorgaard) Consider changing c to GridVariable
# Not required since no .shift() method is used
def point_interpolation(
    point: Array,
    c: GridArray,
    order: int = 1,
    mode: str = 'nearest',
    cval: float = 0.0,
) -> jnp.DeviceArray:
  """Interpolate `c` at `point`.

  Args:
    point: length N 1-D Array. The point to interpolate to.
    c: N-dimensional GridArray. The values that will be interpolated.
    order: Integer in the range 0-1. The order of the spline interpolation.
    mode: one of {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}.
      The `mode` parameter determines how the input array is extended
      beyond its boundaries. Default is 'constant'. Behavior for each valid
      value is as follows:
      'reflect' (`d c b a | a b c d | d c b a`)
          The input is extended by reflecting about the edge of the last
          pixel.
      'constant' (`k k k k | a b c d | k k k k`)
          The input is extended by filling all values beyond the edge with
          the same constant value, defined by the `cval` parameter.
      'nearest' (`a a a a | a b c d | d d d d`)
          The input is extended by replicating the last pixel.
      'mirror' (`d c b | a b c d | c b a`)
          The input is extended by reflecting about the center of the last
          pixel.
      'wrap' (`a b c d | a b c d | a b c d`)
          The input is extended by wrapping around to the opposite edge.
    cval: Value to fill past edges of input if `mode` is 'constant'. Default 0.0

  Returns:
    the interpolated value at `point`.
  """
  point = jnp.asarray(point)

  domain_lower, domain_upper = zip(*c.grid.domain)
  domain_lower = jnp.array(domain_lower)
  domain_upper = jnp.array(domain_upper)
  shape = jnp.array(c.grid.shape)
  offset = jnp.array(c.offset)
  # For each dimension `i` in point,
  # The map from `point[i]` to index is linear.
  # index(domain_lower[i]) = -offset[i]
  # index(domain_upper[i]) = shape[i] - offset[i]
  # This is easily vectorized as
  index = (-offset + (point - domain_lower) * shape /
           (domain_upper - domain_lower))

  return jax.scipy.ndimage.map_coordinates(
      c.data, coordinates=index, order=order, mode=mode, cval=cval)


def interp1d(
    x: Array,
    y: Array,
    axis: int = -1,
    fill_value: Union[str, Array] = jnp.nan,
) -> Callable[[Array], jnp.DeviceArray]:
  """Build a function to interpolate `y = f(x)`.

  x and y are arrays of values used to approximate some function f: y = f(x).
  This returns a function that uses linear interpolation to find the value of
  new points.

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
    x: Length N 1-D array of values to build the interpolation function with.
      x is assumed to be strictly increasing, but this is not checked. If x is
      not strictly increasing, unpredictable behavior will result.
    y: Shape (..., N, ...) array of values corresponding to f(x).
    axis: Specifies the axis of y along which to interpolate. Interpolation
      defaults to the last axis of y.
    fill_value: Scalar array or string. If array, this value will be used to
      fill in for requested points outside of the data range. If not provided,
      then the default is NaN. If "extrapolate", then extrapolation is used.
      If "constant_extension", then the function is extended as a constant equal
      to the value at the edge of the data.

  Returns:
    Callable mapping array x_new to values y_new, where
      y_new.shape = y.shape[:axis] + x_new.shape + y.shape[axis + 1:]
  """
  allowed_fill_value_strs = {'constant_extension', 'extrapolate'}
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
  n_x = x.shape[0]

  y = jnp.asarray(y)
  if y.ndim < 1:
    raise ValueError(f'Expected `y` to have rank >= 1. Found shape {y.shape}')

  if x.size < 2:
    raise ValueError(f'`x` must have at least 2 entries. Found shape {x.shape}')

  if x.size != y.shape[axis]:
    raise ValueError(
        f'x and y arrays must be equal in length along interpolation axis. '
        f'Found x.shape={x.shape} and y.shape={y.shape} and axis={axis}')

  axis = array_utils.normalize_axis(axis, ndim=y.ndim)

  def interp_func(x_new: jnp.DeviceArray) -> jnp.DeviceArray:
    """Implementation of the interpolation function."""
    x_new = jnp.asarray(x_new)

    # It is easiest if we assume x_new is 1D
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
    elif isinstance(fill_value, str) and fill_value == 'constant_extension':
      delta_x = expand(x_new_clipped - x_below)
      y_new = y_below + delta_x * slope
    else:  # Else fill_value is an Array.
      delta_x = expand(x_new - x_below)
      fill_value_ = expand(fill_value)
      y_new = y_below + delta_x * slope
      y_new = jnp.where(
          delta_x < 0, fill_value_,
          jnp.where(delta_x > expand(x_above - x_below), fill_value_, y_new))
    return jnp.reshape(
        y_new, y_new.shape[:axis] + x_new_shape + y_new.shape[axis + 1:])

  return interp_func
