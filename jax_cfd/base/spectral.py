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

"""Utilities for spectral filtering."""

from typing import Callable, Union

import jax.numpy as jnp
from jax_cfd.base import fft
from jax_cfd.base import grids
import numpy as np


AlignedArray = grids.AlignedArray
Array = Union[np.ndarray, jnp.DeviceArray]


def _angular_frequency_magnitude(grid: grids.Grid) -> Array:
  frequencies = [2 * jnp.pi * jnp.fft.fftfreq(size, step)
                 for size, step in zip(grid.shape, grid.step)]
  freq_vector = jnp.stack(jnp.meshgrid(*frequencies, indexing='ij'), axis=0)
  return jnp.linalg.norm(freq_vector, axis=0)


def filter(  # pylint: disable=redefined-builtin
    spectral_density: Callable[[Array], Array],
    array: Array,
    grid: grids.Grid,
) -> Array:
  """Filter an Array with white noise to match a prescribed spectral density."""
  k = _angular_frequency_magnitude(grid)
  filters = jnp.where(k > 0, spectral_density(k), 0.0)
  # The output signal can safely be assumed to be real if our input signal was
  # real, because our spectral density only depends on norm(k).
  return fft.ifftn(fft.fftn(array) * filters).real
