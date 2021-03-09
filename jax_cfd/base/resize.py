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
import jax.numpy as jnp
from jax_cfd.base import array_utils as arr_utils
from jax_cfd.base import grids


def downsample_staggered_velocity_component(u, direction, factor):
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


def downsample_staggered_velocity(
    source_grid,
    destination_grid,
    velocity,
):
  """Downsamples each component of `v` by `factor`."""
  factor = destination_grid.step[0] / source_grid.step[0]
  assert destination_grid.domain == source_grid.domain
  assert round(factor) == factor, factor
  result = []
  for j, u in enumerate(velocity):
    if isinstance(u, grids.AlignedArray):
      downsample = grids.applied(downsample_staggered_velocity_component)
    else:
      downsample = downsample_staggered_velocity_component
    result.append(downsample(u, j, round(factor)))
  return tuple(result)
