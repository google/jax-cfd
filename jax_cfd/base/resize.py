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

"""Resize velocity fields to a different resolution grid."""
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils as arr_utils
from jax_cfd.base import grids
from jax_cfd.base import interpolation
import numpy as np

Array = grids.Array
Field = Tuple[Array, ...]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
RawArray = jnp.ndarray


def downsample_staggered_velocity_component(u: Array, direction: int,
                                            factor: int) -> Array:
  """Downsamples `u`, an array of velocities in the given `direction`.

  Downsampling consists of the following steps:
    * Establish new downsampled control volumes. Each will consist of
      `factor ** dimension` of the fine-grained control volumes.
    * Discard all of the `u` values that do not lie on a face of the new control
      volume in `direction`.
    * Compute the mean of all `u` values that lie on each control volume face in
      the given `direction`.

  This procedure guarantees that if our source velocity has zero divergence
  (i.e., corresponds to an incompressible flow), the downsampled velocity field
  also has zero divergence.

  For example,

  ```
  u = [[0, 1, 2, 3],
       [4, 5, 6, 7],
       [8, 9, 10, 11],
       [12, 13, 14, 15]]
  w = downsample_velocity(u, direction=0, factor=2)

  assert w == np.array([[4.5, 6.5],
                        [12.5, 14.5]])
  ```

  Args:
    u: an array of velocity values.
    direction: an integer indicating the direction of the velocities `u`.
    factor: the factor by which to downsample.

  Returns:
    Coarse-grained array, reduced in size by ``factor`` along each dimension.
  """
  w = arr_utils.slice_along_axis(u, direction, slice(factor - 1, None, factor))
  block_size = tuple(1 if j == direction else factor for j in range(u.ndim))
  return arr_utils.block_reduce(w, block_size, jnp.mean)


def top_hat_downsample(
    source_grid: grids.Grid,
    destination_grid: grids.Grid,
    variables: GridVariableVector,
    filter_size: Optional[Union[int, Tuple[int, ...]]] = None
) -> GridVariableVector:
  """Filters each variable by filter_size and subsamples onto destination_grid.

  Downsampling consists of the following steps:
    * Filter the data by averaging
    * Interpolate the averaged data onto the destination_grid


  This procedure corresponds to standard top-hat filter + comb downsampling.

  Note that the filter size does not have to equal the factor difference between
  the two grids. The intended use case is for filter size >= factor.

  Args:
    source_grid: the grid of variable u. Note: this is legacy implementation,
      variables[i] is an instance of GridVariable and has a grid attribute.
    destination_grid: the grid on which to interpolate filtered variables.
    variables: a tuple of GridVariables. Note that the  grid attribute of each
      variable has to agree with source_grid.
    filter_size: the number of grid points used in the filter. If it's an int,
      it specifies the same number of points to filter in all directions. If it
      is a tuple. each direction is specified separately.

  Returns:
    a tuple of GridVariables interpolated on destination_grid
  """
  # assumes different filtering can be done in different directions
  factor = tuple(
      dx / dx_source
      for dx, dx_source in zip(destination_grid.step, source_grid.step))
  if filter_size is None:
    filter_size = factor
  if isinstance(filter_size, int):
    filter_size = tuple(filter_size for _ in range(source_grid.ndim))
  assert destination_grid.domain == source_grid.domain
  assert all([round(f) == f for f in factor])
  assert all([round(f) == f for f in filter_size])  # this can be relaxed
  assert all(abs(
      np.array(filter_size) % 2)) == 0  # only even filters are implemented
  assert all(abs(
      np.array(factor) % 2)) == 0  # only even factors are implemented
  result = []
  for c in variables:
    if c.grid != source_grid:
      raise grids.InconsistentGridError(
          f'source_grid for downsampling is {source_grid}, but c is defined'
          f' on {c.grid}')
    bc = c.bc
    offset = c.offset
    c_centered = interpolation.linear(c, c.grid.cell_center).array
    center_offset = np.array(source_grid.cell_center)
    grid_shape = np.array(source_grid.shape)
    for axis in range(c.grid.ndim):
      c_centered = bc.pad(
          c_centered, round(filter_size[axis]) // 2, axis=axis)
      c_centered = bc.pad(
          c_centered, -(round(filter_size[axis]) // 2), axis=axis)
      convolution_filter = jnp.ones(round(
          filter_size[axis])) / filter_size[axis]
      convolve_1d = lambda arr, convolution_filter=convolution_filter: jnp.convolve(  # pylint: disable=g-long-lambda
          arr, convolution_filter, 'valid')
      axes = list(range(source_grid.ndim))
      axes.remove(axis)
      for ax in axes:
        convolve_1d = jax.vmap(convolve_1d, in_axes=ax, out_axes=ax)
      c_centered = convolve_1d(c_centered.data)
      if np.isclose(offset[axis], 0):
        start = 0
        end = c_centered.shape[axis] - 1
      elif np.isclose(offset[axis], 0.5):
        start = int(factor[axis]) // 2
        end = None
      elif np.isclose(offset[axis], 1.0):
        start = int(factor[axis])
        end = None
      else:
        raise NotImplementedError(f'offset {offset} is not implemented.')
      c_centered = arr_utils.slice_along_axis(
          c_centered, axis, slice(start, end, int(factor[axis])))
      center_offset[axis] = offset[axis]
      grid_shape[axis] = destination_grid.shape[axis]
      c_centered = grids.GridArray(
          c_centered,
          offset=tuple(center_offset),
          grid=grids.Grid(shape=tuple(grid_shape), domain=source_grid.domain))
    c = grids.GridVariable(c_centered, bc)
    result.append(c)
  return tuple(result)


def downsample_staggered_velocity(
    source_grid: grids.Grid,
    destination_grid: grids.Grid,
    velocity: Union[Field, GridArrayVector, GridVariableVector],
):
  """Downsamples each component of `v` by `factor`."""
  factor = destination_grid.step[0] / source_grid.step[0]
  assert destination_grid.domain == source_grid.domain
  assert round(factor) == factor, factor
  result = []
  for j, u in enumerate(velocity):
    if isinstance(u, GridVariable):
      def downsample(u: GridVariable, direction: int,
                     factor: int) -> GridVariable:
        if u.grid != source_grid:
          raise grids.InconsistentGridError(
              f'source_grid for downsampling is {source_grid}, but u is defined'
              f' on {u.grid}')
        array = downsample_staggered_velocity_component(u.data, direction,
                                                        round(factor))
        grid_array = GridArray(array, offset=u.offset, grid=destination_grid)
        return GridVariable(grid_array, bc=u.bc)
    elif isinstance(u, GridArray):
      def downsample(u: GridArray, direction: int, factor: int) -> GridArray:
        if u.grid != source_grid:
          raise grids.InconsistentGridError(
              f'source_grid for downsampling is {source_grid}, but u is defined'
              f' on {u.grid}')
        array = downsample_staggered_velocity_component(u.data, direction,
                                                        round(factor))
        return GridArray(array, offset=u.offset, grid=destination_grid)
    else:
      downsample = downsample_staggered_velocity_component
    result.append(downsample(u, j, round(factor)))
  return tuple(result)


# TODO(dresdner) gin usage should be restricted to jax_cfd.ml
def downsample_spectral(_: grids.Grid, destination_grid: grids.Grid,
                        signal_hat: RawArray):
  """Downsamples a 2D signal in the Fourier basis to the `destination_grid`."""
  kx, ky = destination_grid.rfft_axes()
  (num_x,), (num_y,) = kx.shape, ky.shape

  input_num_x, _ = signal_hat.shape

  downed = jnp.concatenate(
      [signal_hat[:num_x // 2, :num_y], signal_hat[-num_x // 2:, :num_y]])

  scale = (num_x / input_num_x)
  downed *= scale**2
  return downed
