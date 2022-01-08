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
from jax_cfd.base import forcings
from jax_cfd.base import grids


def kolmogorov_forcing_fn(grid: grids.Grid) -> forcings.ForcingFn:
  """Constant Kolmogorov forcing function."""
  offset = (0, 0)
  _, ys = grid.mesh(offset=offset)
  f = -4 * jnp.cos(4 * ys)
  f = (grids.GridArray(f, offset, grid),)

  def forcing(v):
    del v  # unused
    return f

  return forcing


def spectral_no_forcing(grid):
  """Zero-valued forcing field for unforced simulations."""
  dim = 1  # TODO(dresdner) hard coded for vorticity field

  # TODO(dresdner) double check that types are consistent
  # this seems like a bit of a hack.
  f = tuple(
      grids.GridArray(jnp.array(0), offset=(0, 0), grid=grid)
      for _ in range(dim))

  def forcing(v):
    del v  # unused
    return f

  return forcing
