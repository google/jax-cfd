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

"""Module for functionality related to advection."""

from typing import Optional, Tuple

from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def advect_linear(c: GridVariable,
                  v: GridVariableVector,
                  dt: Optional[float] = None) -> GridArray:
  """Computes advection for collocated scalar `c` with velocity `v`."""
  del dt
  flux_bc = [
      boundaries.get_flux_bc_from_velocity_and_scalar(u, c, direction)
      for direction, u in enumerate(v)
  ]
  flux = tuple(flux_bc[axis].impose_bc(c.array * v[axis].array)
               for axis in range(c.grid.ndim))
  return -fd.centered_divergence(flux)


def _velocities_to_flux(v: GridVariableVector) -> Tuple[GridVariableVector]:
  """Computes the cell-centered convective flux for a velocity field.

  This is the flux associated with the nonlinear term `vv` for velocity `v`.
  The boundary condition on the flux is inherited from `v`.

  Args:
    v: velocity vector.

  Returns:
    A tuple of tuples `flux` of `GridVariable`s with the values `v[i]*v[j]`
  """
  ndim = len(v)
  flux = [tuple() for _ in range(ndim)]
  ndim = len(v)
  flux = [tuple() for _ in range(ndim)]
  for i in range(ndim):
    for j in range(ndim):
      if i <= j:
        bc = boundaries.get_flux_bc_from_velocity_and_scalar(
            v[j], v[i], j)
        flux[i] += (bc.impose_bc(v[i].array * v[j].array),)
      else:
        flux[i] += (flux[j][i],)
  return tuple(flux)


def convect_linear(v: GridVariableVector) -> GridArrayVector:
  """Computes convection/self-advection of the velocity field `v`.

  Args:
    v: velocity vector.

  Returns:
    A tuple containing the time derivative of each component of `v` due to
    convection.
  """
  fluxes = _velocities_to_flux(v)
  return tuple(-fd.centered_divergence(flux) for flux in fluxes)
