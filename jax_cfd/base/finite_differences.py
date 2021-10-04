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
"""Functions for approximating derivatives.

Finite difference methods operate on GridVariable and return GridArray.
Evaluating finite differences requires boundary conditions, which are defined
for a GridVariable. But the operation of taking a derivative makes the boundary
condition undefined, which is why the return type is GridArray.

For example, if the variable c has the boundary condition c_b = 0, and we take
the derivate dc/dx, then it is unclear what the boundary condition on dc/dx
should be. So programmatically, after taking finite differences and doing
operations, the user has to explicitly assign boundary conditions to the result.

Example:
  c = GridVariable(c_array, c_boundary_condition)
  dcdx = finite_differences.forward_difference(c)  # returns GridArray
  c_new = c + dt * (-velocity * dcdx)  # operations on GridArrays
  c_new = GridVariable(c_new, c_boundary_condition)  # assocaite BCs
"""

import typing
from typing import Optional, Sequence, Tuple
from jax_cfd.base import grids
from jax_cfd.base import interpolation
import numpy as np

GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridArrayTensor = grids.GridArrayTensor


def stencil_sum(*arrays: GridArray) -> GridArray:
  """Sum arrays across a stencil, with an averaged offset."""
  # pylint: disable=line-too-long
  offset = grids.averaged_offset(*arrays)
  # pytype appears to have a bad type signature for sum():
  # Built-in function sum was called with the wrong arguments [wrong-arg-types]
  #          Expected: (iterable: Iterable[complex])
  #   Actually passed: (iterable: Generator[Union[jax.interpreters.xla.DeviceArray, numpy.ndarray], Any, None])
  result = sum(array.data for array in arrays)  # type: ignore
  grid = grids.consistent_grid(*arrays)
  return grids.GridArray(result, offset, grid)


# incompatible with typing.overload
# pylint: disable=pointless-statement
# pylint: disable=function-redefined
# pylint: disable=unused-argument


@typing.overload
def central_difference(u: GridVariable, axis: int) -> GridArray:
  ...


@typing.overload
def central_difference(
    u: GridVariable, axis: Optional[Tuple[int, ...]]) -> Tuple[GridArray, ...]:
  ...


def central_difference(u, axis=None):
  """Approximates grads with central differences."""
  if axis is None:
    axis = range(u.grid.ndim)
  if not isinstance(axis, int):
    return tuple(central_difference(u, a) for a in axis)
  diff = stencil_sum(u.shift(+1, axis), -u.shift(-1, axis))
  return diff / (2 * u.grid.step[axis])


@typing.overload
def backward_difference(u: GridVariable, axis: int) -> GridArray:
  ...


@typing.overload
def backward_difference(
    u: GridVariable, axis: Optional[Tuple[int, ...]]) -> Tuple[GridArray, ...]:
  ...


def backward_difference(u, axis=None):
  """Approximates grads with finite differences in the backward direction."""
  if axis is None:
    axis = range(u.grid.ndim)
  if not isinstance(axis, int):
    return tuple(backward_difference(u, a) for a in axis)
  diff = stencil_sum(u.array, -u.shift(-1, axis))
  return diff / u.grid.step[axis]


@typing.overload
def forward_difference(u: GridVariable, axis: int) -> GridArray:
  ...


@typing.overload
def forward_difference(
    u: GridVariable,
    axis: Optional[Tuple[int, ...]] = ...) -> Tuple[GridArray, ...]:
  ...


def forward_difference(u, axis=None):
  """Approximates grads with finite differences in the forward direction."""
  if axis is None:
    axis = range(u.grid.ndim)
  if not isinstance(axis, int):
    return tuple(forward_difference(u, a) for a in axis)
  diff = stencil_sum(u.shift(+1, axis), -u.array)
  return diff / u.grid.step[axis]


def laplacian(u: GridArray) -> GridArray:
  """Approximates the Laplacian of `u`."""
  scales = np.square(1 / np.array(u.grid.step, dtype=u.dtype))
  result = -2 * u * np.sum(scales)
  # TODO(pnorgaard) remove temporary GridVariable hack
  u = grids.make_gridvariable_from_gridarray(u)
  for axis in range(u.grid.ndim):
    result += stencil_sum(u.shift(-1, axis), u.shift(+1, axis)) * scales[axis]
  return result


def divergence(v: Sequence[GridArray]) -> GridArray:
  """Approximates the divergence of `v` using backward differences."""
  grid = grids.consistent_grid(*v)
  if len(v) != grid.ndim:
    raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                     f'Expected length {grid.ndim}; got {len(v)}.')
  # TODO(pnorgaard) remove temporary GridVariable hack
  v = tuple(grids.make_gridvariable_from_gridarray(u) for u in v)
  differences = [backward_difference(u, axis) for axis, u in enumerate(v)]
  return sum(differences)


@typing.overload
def gradient_tensor(v: GridVariable) -> GridArrayTensor:
  ...


@typing.overload
def gradient_tensor(v: Sequence[GridVariable]) -> GridArrayTensor:
  ...


def gradient_tensor(v):
  """Approximates the cell-centered gradient of `v`."""
  if not isinstance(v, GridVariable):
    return GridArrayTensor(np.stack([gradient_tensor(u) for u in v], axis=-1))
  grad = []
  for axis in range(v.grid.ndim):
    offset = v.offset[axis]
    if offset < 0.5:
      deriv_fn = forward_difference
    elif offset > 0.5:
      deriv_fn = backward_difference
    else:  # offset == 0.5
      deriv_fn = central_difference

    derivative = deriv_fn(v, axis)
    grad.append(interpolation.linear(derivative, v.grid.cell_center))
  return GridArrayTensor(grad)


def curl_2d(v: Sequence[GridVariable]) -> GridArray:
  """Approximates the curl of `v` in 2D using forward differences."""
  if len(v) != 2:
    raise ValueError(f'Length of `v` is not 2: {len(v)}')
  grid = grids.consistent_grid(*v)
  if grid.ndim != 2:
    raise ValueError(f'Grid dimensionality is not 2: {grid.ndim}')
  return forward_difference(v[1], axis=0) - forward_difference(v[0], axis=1)


def curl_3d(
    v: Sequence[GridVariable]) -> Tuple[GridArray, GridArray, GridArray]:
  """Approximates the curl of `v` in 2D using forward differences."""
  if len(v) != 3:
    raise ValueError(f'Length of `v` is not 3: {len(v)}')
  grid = grids.consistent_grid(*v)
  if grid.ndim != 3:
    raise ValueError(f'Grid dimensionality is not 3: {grid.ndim}')
  curl_x = (forward_difference(v[2], axis=1) - forward_difference(v[1], axis=2))
  curl_y = (forward_difference(v[0], axis=2) - forward_difference(v[2], axis=0))
  curl_z = (forward_difference(v[1], axis=0) - forward_difference(v[0], axis=1))
  return (curl_x, curl_y, curl_z)
