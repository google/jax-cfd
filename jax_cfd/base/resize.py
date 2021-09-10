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
import functools

import gin
import jax.numpy as jnp
from jax_cfd.base import array_utils as arr_utils
from jax_cfd.base import grids
from google3.research.simulation.whirl.experiments.convection import nonperiodic_boundary_conditions as bc

NONPERIODIC = 'dirichlet'
PERIODIC = 'periodic'


def downsample_staggered_velocity_component(u, direction, factors):
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
    factors: the tuple of factor by which to downsamplein each direction.

  Returns:
    Coarse-grained array, reduced in size by ``factors[axis]`` along each axis.
  """
  if isinstance(factors, int):
    factors = tuple(factors for _ in range(u.ndim))
  w = arr_utils.slice_along_axis(
      u, direction, slice(factors[direction] - 1, None, factors[direction]))
  block_size = tuple(1 if j == direction else factors[j] for j in range(u.ndim))
  return arr_utils.block_reduce(w, block_size, jnp.mean)


def subsample_staggered_velocity_component(u, direction, factors):
  """Downsamples `u`, an array of velocities in the given `direction`.

  Downsampling consists of the following steps:
    * Establish new downsampled control volumes. Each will consist of
      `factor ** dimension` of the fine-grained control volumes.
    * Discard all of the `u` values that do not lie on a face of the new control
      volume in all directions.

  This procedure guarantees that if our source velocity has zero divergence
  (i.e., corresponds to an incompressible flow), the downsampled velocity field
  also has zero divergence.

  For example,

  ```
  u = [[0, 1, 2, 3],
       [4, 5, 6, 7],
       [8, 9, 10, 11],
       [12, 13, 14, 15]]
  w = subsample_velocity(u, direction=None, factor=2)

  assert w == np.array([[5, 7],
                        [9, 11]])
  ```

  Args:
    u: an array of velocity values.
    direction: an integer indicating the direction of the velocities `u`.
    factors: the tuple of factor by which to downsamplein each direction.

  Returns:
    Coarse-grained array, reduced in size by ``factor`` along each dimension.
  """
  del direction
  if isinstance(factors, int):
    factors = tuple(factors for _ in range(u.ndim))
  for direction, factor in enumerate(factors):
    u = arr_utils.slice_along_axis(u, direction, slice(factor - 1, None,
                                                       factor))
  return u


@gin.configurable
def downsample_staggered_velocity(source_grid,
                                  destination_grid,
                                  velocity,
                                  boundary_fn=None):
  """Downsamples each component of `v` by `factor`.

  1) This function preserves 0 divergence if the velocity field has 0
  divergence: if the grid is reduced by the same factor in each direction,
  this uses the old version. If the grid is reduced by different factors in
  different directions, or if the grid has nonperiodic boundary, then a
  different method is used: for each nonperiodic boundary, since boundary
  location has to be preserved, e.g. if
  source.grid.boundaries[0] == NONPERIODIC and if u is x velocity, then every
  "factor" value of u is taken on the interior. That way the offset of the
  wall is preserved. At the same time, v is downsampled using
  downsample_staggered_velocity_component in 0th direction. That preserves
  divergence.
  2) This function allows to the input to contain as many scalars as needed.
  Each scalar is subsampled using subsample_staggered_velocity_component.

  Args:
   source_grid: grids.Grid where velocity is located.
   destination_grid: grids.Grid onto which to interpolate.
   velocity: [AlighedArray,..], velocity=(x_velocity, y_velocity, [z_velocity],
     scalar1, scalar2, ..)
   boundary_fn: for nonperiodic conditions, boundary_fn updates ghost cells.

  Returns:
   downsampled velocity and scalar field.
  """
  if boundary_fn is None:

    def boundary_fn(grid):  # pylint: disable=function-redefined
      # default velocity offsets.
      if grid.ndim == 2:
        offsets = [(1.0, 0.5), (0.5, 1.0)]
      elif grid.ndim == 3:
        offsets = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)]

      def _apply(v):
        v = tuple([
            grids.AlignedArray(u.data, offset) for u, offset in zip(v, offsets)
        ])
        return v

      return _apply

  result = [u for u in velocity]
  factors_tuple = [
      [1 for _ in range(source_grid.ndim)] for _ in range(source_grid.ndim)
  ]
  for axis in range(source_grid.ndim):
    factor = destination_grid.step[axis] / source_grid.step[axis]
    # uncomment when ghost cells are implemented separately from Grids.domain
    # assert destination_grid.domain == source_grid.domain
    assert round(factor) == factor, factor
    factors_tuple[axis][axis] = round(factor)
  downsample = downsample_staggered_velocity_component
  subsample = subsample_staggered_velocity_component
  # use downsample_staggered_velocity_component in all directions if the factors
  # are the same in all directions.
  if len(set([
      factors_tuple[axis][axis] for axis in range(source_grid.ndim)
  ])) == 1 and len(set(
      source_grid.boundaries)) == 1 and source_grid.boundaries[0] == PERIODIC:
    subsample = downsample_staggered_velocity_component
  # these functions are identity if the domain is fully periodic.
  # otherwise interior_points reduces a scalar to interior points only.
  # offset = (0.5, 0.5) guarantees that cells 0 and -1 will be deleted.
  # This will work for 3D periodic case as well.
  interior_points = functools.partial(
      bc.reduce_to_interior, grid=source_grid, offset=(0.5, 0.5))
  # pad adds 0 valued ghost cells along nonperioidic boundaries
  pad = functools.partial(
      bc.interior_pad, grid=destination_grid, offset=(0.5, 0.5))

  # velocity:
  for axis, factors in enumerate(factors_tuple):
    for j, u in enumerate(result[:source_grid.ndim]):
      if axis == j:
        reduction_fn = subsample
      else:
        reduction_fn = downsample

      result[j] = pad(reduction_fn(interior_points(u), j, factors))

  # scalars:
  # currently simply takes every nth scalar where n=factor.
  subsample = subsample_staggered_velocity_component
  for j in range(source_grid.ndim, len(velocity)):
    result[j] = pad(
        subsample(
            interior_points(result[j]), None,
            [factors_tuple[axis][axis] for axis in range(source_grid.ndim)]))
  # applies boundary function and converts result to AlighedArray. Default only
  # converts to AlighedArray.
  result = boundary_fn(destination_grid)(tuple(result))
  if not isinstance(velocity[0], grids.AlignedArray):
    result = tuple([u.data for u in result])
  return result
