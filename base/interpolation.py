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

"""Functions for interpolating `AlignedArray`s."""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax_cfd.base import grids
import numpy as np


AlignedArray = grids.AlignedArray
AlignedField = Tuple[AlignedArray, ...]
InterpolationFn = Callable[
    [AlignedArray, Tuple[float, ...], grids.Grid, AlignedField, float],
    AlignedArray]
FluxLimiter = Callable[[grids.Array], grids.Array]


def _linear_along_axis(c: grids.AlignedArray,
                       offset: float,
                       axis: int,
                       grid: grids.Grid) -> grids.AlignedArray:
  """Linear interpolation of `c` to `offset` along a single specified `axis`."""
  offset_delta = offset - c.offset[axis]

  # If offsets are the same, `c` is unchanged.
  if offset_delta == 0:
    return c

  new_offset = tuple(offset if j == axis else o
                     for j, o in enumerate(c.offset))

  # If offsets differ by an integer, we can just shift `c`.
  if int(offset_delta) == offset_delta:
    return grids.AlignedArray(
        data=grid.shift(c, int(offset_delta), axis).data,
        offset=tuple(new_offset))

  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  floor_weight = ceil - offset_delta
  ceil_weight = 1. - floor_weight
  data = (floor_weight * grid.shift(c, floor, axis).data +
          ceil_weight * grid.shift(c, ceil, axis).data)
  return grids.AlignedArray(data, new_offset)


def linear(
    c: grids.AlignedArray,
    offset: Tuple[float, ...],
    grid: grids.Grid,
    v: Tuple[grids.AlignedArray, ...] = None,
    dt: float = None
) -> grids.AlignedArray:
  """Multi-linear interpolation of `c` to `offset`.

  Args:
    c: an `AlignedArray`.
    offset: a tuple of floats describing the offset to which we will interpolate
      `c`. Must have the same length as `c.offset`.
    grid: a `Grid` compatible with the values `c`.
    v: a tuple of `AlignedArray`s representing velocity field. Not used.
    dt: size of the time step. Not used.

  Returns:
    An `AlignedArray` containing the values of `c` after linear interpolation
    to `offset`. The returned value will have offset equal to `offset`.
  """
  del v, dt  # unused
  if len(offset) != len(c.offset):
    raise ValueError('`c.offset` and `offset` must have the same length;'
                     f'got {c.offset} and {offset}.')
  interpolated = c
  for a, o in enumerate(offset):
    interpolated = _linear_along_axis(interpolated, o, a, grid)
  return interpolated


def upwind(
    c: grids.AlignedArray,
    offset: Tuple[float, ...],
    grid: grids.Grid,
    v: Tuple[grids.AlignedArray, ...],
    dt: float = None
) -> grids.AlignedArray:
  """Upwind interpolation of `c` to `offset` based on velocity field `v`.

  Interpolates values of `c` to `offset` in two steps:
  1) Identifies the axis along which `c` is interpolated. (must be single axis)
  2) For positive (negative) velocity along interpolation axis uses value from
     the previous (next) cell along that axis correspondingly.

  Args:
    c: an `AlignedArray`, the quantitity to be interpolated.
    offset: a tuple of floats describing the offset to which we will interpolate
      `c`. Must have the same length as `c.offset` and differ in at most one
      entry.
    grid: a `Grid` on which `c` and `u` are located.
    v: a tuple of `AlignedArray`s representing velocity field with offsets at
      faces of `c`. One of the components must have the same offset as `offset`.
    dt: size of the time step. Not used.

  Returns:
    An `AlignedArray` that containins the values of `c` after interpolation to
    `offset`.
  Raises:
    AlignmentError: if `offset` and `c.offset` differ in more than one entry.
  """
  del dt  # unused
  if c.offset == offset: return c
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if current != target
  )
  if len(interpolation_axes) != 1:
    raise grids.AlignmentError(
        f'for upwind interpolation `c.offset` and `offset` must differ at most '
        f'in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  u = v[axis]
  offset_delta = u.offset[axis] - c.offset[axis]

  # If offsets differ by an integer, we can just shift `c`.
  if int(offset_delta) == offset_delta:
    return grids.AlignedArray(
        data=grid.shift(u, int(offset_delta), axis).data,
        offset=offset)

  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  return grids.applied(jnp.where)(
      u > 0, grid.shift(c, floor, axis).data, grid.shift(c, ceil, axis).data
  )


def lax_wendroff(
    c: grids.AlignedArray,
    offset: Tuple[float, ...],
    grid: grids.Grid,
    v: Tuple[grids.AlignedArray, ...] = None,
    dt: float = None
) -> grids.AlignedArray:
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
    c: an `AlignedArray`, the quantitity to be interpolated.
    offset: a tuple of floats describing the offset to which we will interpolate
      `c`. Must have the same length as `c.offset` and differ in at most one
      entry.
    grid: a `Grid` on which `c` and `u` are located.
    v: a tuple of `AlignedArray`s representing velocity field with offsets at
      faces of `c`. One of the components must have the same offset as `offset`.
    dt: size of the time step. Not used.

  Returns:
    An `AlignedArray` that containins the values of `c` after interpolation to
    `offset`.
  Raises:
    AlignmentError: if `offset` and `c.offset` differ in more than one entry.
  """
  # TODO(dkochkov) add a function to compute interpolation axis.
  if c.offset == offset: return c
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if current != target
  )
  if len(interpolation_axes) != 1:
    raise grids.AlignmentError(
        f'for Lax-Wendroff interpolation `c.offset` and `offset` must differ at'
        f' most in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  u = v[axis]
  offset_delta = u.offset[axis] - c.offset[axis]
  floor = int(np.floor(offset_delta))  # used for positive velocity
  ceil = int(np.ceil(offset_delta))  # used for negative velocity
  courant_numbers = (dt / grid.step[axis]) * u.data
  positive_u_case = (
      grid.shift(c, floor, axis).data + 0.5 * (1 - courant_numbers) *
      (grid.shift(c, ceil, axis).data - grid.shift(c, floor, axis).data))
  negative_u_case = (
      grid.shift(c, ceil, axis).data - 0.5 * (1 + courant_numbers) *
      (grid.shift(c, ceil, axis).data - grid.shift(c, floor, axis).data))
  return grids.where(u > 0, positive_u_case, negative_u_case)


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
  def tvd_interpolation(c, offset, grid, v, dt):
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
        c_low = upwind(c, offset, grid, v, dt)
        c_high = interpolation_fn(c, offset, grid, v, dt)

        # because we are interpolating to the right we are using 2 points ahead
        # and 2 points behind: `c`, `c_left`.
        c_left = grid.shift(c, -1, axis)
        c_right = grid.shift(c, 1, axis)
        c_next_right = grid.shift(c, 2, axis)
        # Velocities of different sign are evaluated with limiters at different
        # points. See equations (4.34) -- (4.39) from the reference above.
        positive_u_r = safe_div(c.data - c_left.data, c_right.data - c.data)
        negative_u_r = safe_div(c_next_right.data - c_right.data,
                                c_right.data - c.data)
        positive_u_phi = grids.AlignedArray(limiter(positive_u_r), c_low.offset)
        negative_u_phi = grids.AlignedArray(limiter(negative_u_r), c_low.offset)
        u = v[axis]
        phi = grids.applied(jnp.where)(u > 0, positive_u_phi, negative_u_phi)
        c_interpolated = c_low - (c_low - c_high) * phi
        c = grids.AlignedArray(c_interpolated.data, interpolation_offset)
    return c

  return tvd_interpolation
