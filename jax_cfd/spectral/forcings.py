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

"""Functions that implement different forcing terms."""

import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.spectral import types as spectral_types


# pylint: disable=unused-argument
def kolmogorov_forcing_fn(grid: grids.Grid,
                          state: spectral_types.Array) -> spectral_types.Array:
  """Constant Kolmogorov forcing function."""
  _, ys = grid.mesh(offset=(0, 0))
  constant_forcing = -4 * jnp.cos(4 * ys)
  return constant_forcing
