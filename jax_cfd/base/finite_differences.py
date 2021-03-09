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

"""Functions for approximating derivatives."""

import typing
from typing import Optional, Sequence, Tuple
from jax_cfd.base import grids
from jax_cfd.base import interpolation
import numpy as np

AlignedArray = grids.AlignedArray
Tensor = grids.Tensor


def stencil_sum(*arrays: AlignedArray) -> AlignedArray:
  """Sum arrays across a stencil, with an averaged offset."""
  # pylint: disable=line-too-long
  offset = grids.averaged_offset(*arrays)
  # pytype appears to have a bad type signature for sum():
  # Built-in function sum was called with the wrong arguments [wrong-arg-types]
  #          Expected: (iterable: Iterable[complex])
  #   Actually passed: (iterable: Generator[Union[jax.interpreters.xla.DeviceArray, numpy.ndarray], Any, None])
  result = sum(array.data for array in arrays)  # type: ignore
  return grids.AlignedArray(result, offset)


# incompatible with typing.overload
# pylint: disable=pointless-statement
# pylint: disable=function-redefined
# pylint: disable=unused-argument


@typing.overload
def central_difference(
    u: AlignedArray, grid: grids.Grid, axis: int
) -> AlignedArray:
  ...


@typing.overload
def central_difference(
    u: AlignedArray, grid: grids.Grid, axis: Optional[Tuple[int, ...]]
) -> Tuple[AlignedArray, ...]:
  ...


def central_difference(u, grid, axis=None):
  """Approximates grads with central differences."""
  if axis is None:
    axis = range(grid.ndim)
  if not isinstance(axis, int):
    return tuple(central_difference(u, grid, a) for a in axis)
  diff = stencil_sum(grid.shift(u, +1, axis), -grid.shift(u, -1, axis))
  return diff / (2 * grid.step[axis])


@typing.overload
def backward_difference(
    u: AlignedArray, grid: grids.Grid, axis: int
) -> AlignedArray:
  ...


@typing.overload
def backward_difference(
    u: AlignedArray, grid: grids.Grid, axis: Optional[Tuple[int, ...]]
) -> Tuple[AlignedArray, ...]:
  ...


def backward_difference(u, grid, axis=None):
  """Approximates grads with finite differences in the backward direction."""
  if axis is None:
    axis = range(grid.ndim)
  if not isinstance(axis, int):
    return tuple(backward_difference(u, grid, a) for a in axis)
  diff = stencil_sum(u, -grid.shift(u, -1, axis))
  return diff / grid.step[axis]


@typing.overload
def forward_difference(
    u: AlignedArray, grid: grids.Grid, axis: int
) -> AlignedArray:
  ...


@typing.overload
def forward_difference(
    u: AlignedArray, grid: grids.Grid, axis: Optional[Tuple[int, ...]] = ...
) -> Tuple[AlignedArray, ...]:
  ...


def forward_difference(u, grid, axis=None):
  """Approximates grads with finite differences in the forward direction."""
  if axis is None:
    axis = range(grid.ndim)
  if not isinstance(axis, int):
    return tuple(forward_difference(u, grid, a) for a in axis)
  diff = stencil_sum(grid.shift(u, +1, axis), -u)
  return diff / grid.step[axis]


def laplacian(u: AlignedArray, grid: grids.Grid) -> AlignedArray:
  """Approximates the Laplacian of `u`."""
  scales = np.square(1 / np.array(grid.step))
  result = -2 * u * np.sum(scales)
  for axis in range(grid.ndim):
    result += stencil_sum(grid.shift(u, -1, axis),
                          grid.shift(u, +1, axis)) * scales[axis]
  return result


def divergence(v: Sequence[AlignedArray], grid: grids.Grid) -> AlignedArray:
  """Approximates the divergence of `v` using backward differences."""
  if len(v) != grid.ndim:
    raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                     f'Expected length {grid.ndim}; got {len(v)}.')
  differences = [backward_difference(u, grid, axis) for axis, u in enumerate(v)]
  return sum(differences)


@typing.overload
def gradient_tensor(v: AlignedArray, grid: grids.Grid) -> Tensor:
  ...


@typing.overload
def gradient_tensor(v: Sequence[AlignedArray], grid: grids.Grid) -> Tensor:
  ...


def gradient_tensor(v, grid):
  """Approximates the cell-centered gradient of `v`."""
  if not isinstance(v, AlignedArray):
    return Tensor(np.stack([gradient_tensor(u, grid) for u in v], axis=-1))

  grad = []
  for axis in range(grid.ndim):
    offset = v.offset[axis]
    if offset < 0.5:
      deriv_fn = forward_difference
    elif offset > 0.5:
      deriv_fn = backward_difference
    else:  # offset == 0.5
      deriv_fn = central_difference

    derivative = deriv_fn(v, grid, axis)
    grad.append(interpolation.linear(derivative, grid.cell_center, grid))
  return Tensor(grad)


def curl_2d(v: Sequence[AlignedArray], grid: grids.Grid) -> AlignedArray:
  """Approximates the curl of `v` in 2D using forward differences."""
  if len(v) != 2:
    raise ValueError(f'Length of `v` is not 2: {len(v)}')
  if grid.ndim != 2:
    raise ValueError(f'Grid dimensionality is not 2: {grid.ndim}')
  return (forward_difference(v[1], grid, axis=0)
          - forward_difference(v[0], grid, axis=1))


def curl_3d(
    v: Sequence[AlignedArray], grid: grids.Grid,
) -> Tuple[AlignedArray, AlignedArray, AlignedArray]:
  """Approximates the curl of `v` in 2D using forward differences."""
  if len(v) != 3:
    raise ValueError(f'Length of `v` is not 3: {len(v)}')
  if grid.ndim != 3:
    raise ValueError(f'Grid dimensionality is not 3: {grid.ndim}')
  curl_x = (forward_difference(v[2], grid, axis=1)
            - forward_difference(v[1], grid, axis=2))
  curl_y = (forward_difference(v[0], grid, axis=2)
            - forward_difference(v[2], grid, axis=0))
  curl_z = (forward_difference(v[1], grid, axis=0)
            - forward_difference(v[0], grid, axis=1))
  return (curl_x, curl_y, curl_z)
