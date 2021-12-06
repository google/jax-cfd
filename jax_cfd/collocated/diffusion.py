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

# TODO(pnorgaard) Implement bicgstab for non-symmetric operators

"""Module for functionality related to diffusion."""

from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids

GridArray = grids.GridArray
GridVariable = grids.GridVariable


# TODO(pnorgaard): Implement the equivalent expanded 5-point laplacian operator
def diffuse(c: GridVariable, nu: float) -> GridArray:
  """Returns the rate of change in a concentration `c` due to diffusion."""
  if not boundaries.has_all_periodic_boundary_conditions(c):
    raise ValueError('Expected periodic BC')
  gradient = fd.central_difference(c, axis=None)
  gradient = tuple(grids.GridVariable(g, c.bc) for g in gradient)
  return nu * fd.centered_divergence(gradient)
