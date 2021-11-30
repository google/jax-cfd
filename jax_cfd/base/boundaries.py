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
"""Classes that specify how boundary conditions are applied to arrays."""

import dataclasses
from typing import Sequence, Tuple
from jax import lax
import jax.numpy as jnp
from jax_cfd.base import grids

BoundaryConditions = grids.BoundaryConditions
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


class BCType:
  PERIODIC = 'periodic'
  DIRICHLET = 'dirichlet'
  NEUMANN = 'neumann'


@dataclasses.dataclass(init=False, frozen=True)
class HomogeneousBoundaryConditions(BoundaryConditions):
  """Boundary conditions for a PDE variable.

  Example usage:
    grid = Grid((10, 10))
    array = GridArray(np.zeros((10, 10)), offset=(0.5, 0.5), grid)
    bc = HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                        (BCType.DIRICHLET, BCType.DIRICHLET)))
    u = GridVariable(array, bc)

  Attributes:
    types: `types[i]` is a tuple specifying the lower and upper BC types for
      dimension `i`.
  """
  types: Tuple[Tuple[str, str], ...]

  def __init__(self, types: Sequence[Tuple[str, str]]):
    types = tuple(types)
    object.__setattr__(self, 'types', types)

  def shift(
      self,
      u: GridArray,
      offset: int,
      axis: int,
  ) -> GridArray:
    """Shift an GridArray by `offset`.

    Args:
      u: an `GridArray` object.
      offset: positive or negative integer offset to shift.
      axis: axis to shift along.

    Returns:
      A copy of `u`, shifted by `offset`. The returned `GridArray` has offset
      `u.offset + offset`.
    """
    padded = self._pad(u, offset, axis)
    trimmed = self._trim(padded, -offset, axis)
    return trimmed

  def _pad(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """Pad a GridArray by `padding`.

    Args:
      u: a `GridArray` object.
      width: number of elements to pad along axis. Use negative value for lower
        boundary or positive value for upper boundary.
      axis: axis to pad along.

    Returns:
      Padded array, elongated along the indicated axis.
    """
    if width < 0:  # pad lower boundary
      bc_type = self.types[axis][0]
      padding = (-width, 0)
    else:  # pad upper boundary
      bc_type = self.types[axis][1]
      padding = (0, width)

    full_padding = [(0, 0)] * u.data.ndim
    full_padding[axis] = padding

    offset = list(u.offset)
    offset[axis] -= padding[0]

    if bc_type == BCType.PERIODIC:
      pad_kwargs = dict(mode='wrap')
    elif bc_type == BCType.DIRICHLET:
      pad_kwargs = dict(mode='constant')
    elif bc_type == BCType.NEUMANN:
      pad_kwargs = dict(mode='edge')
    else:
      raise ValueError('invalid boundary type')

    data = jnp.pad(u.data, full_padding, **pad_kwargs)
    return GridArray(data, tuple(offset), u.grid)

  def _trim(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """Trim padding from a GridArray.

    Args:
      u: a `GridArray` object.
      width: number of elements to trim along axis. Use negative value for lower
        boundary or positive value for upper boundary.
      axis: axis to trim along.

    Returns:
      Trimmed array, shrunk along the indicated axis.
    """
    if width < 0:  # trim lower boundary
      padding = (-width, 0)
    else:  # trim upper boundary
      padding = (0, width)

    limit_index = u.data.shape[axis] - padding[1]
    data = lax.slice_in_dim(u.data, padding[0], limit_index, axis=axis)
    offset = list(u.offset)
    offset[axis] += padding[0]
    return GridArray(data, tuple(offset), u.grid)


# Convenience utilities to ease updating of BoundaryConditions implementation
def periodic_boundary_conditions(ndim: int) -> BoundaryConditions:
  """Returns periodic BCs for a variable with `ndim` spatial dimension."""
  return HomogeneousBoundaryConditions(
      ((BCType.PERIODIC, BCType.PERIODIC),) * ndim)


def dirichlet_boundary_conditions(ndim: int) -> BoundaryConditions:
  """Returns Dirichelt BCs for a variable with `ndim` spatial dimension."""
  return HomogeneousBoundaryConditions(
      ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim)


def neumann_boundary_conditions(ndim: int) -> BoundaryConditions:
  """Returns Neumann BCs for a variable with `ndim` spatial dimension."""
  return HomogeneousBoundaryConditions(
      ((BCType.NEUMANN, BCType.NEUMANN),) * ndim)


def periodic_and_dirichlet_boundary_conditions() -> BoundaryConditions:
  """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1."""
  return HomogeneousBoundaryConditions(
      ((BCType.PERIODIC, BCType.PERIODIC),
       (BCType.DIRICHLET, BCType.DIRICHLET)))


def periodic_and_neumann_boundary_conditions() -> BoundaryConditions:
  """Returns BCs periodic for dimension 0 and Neumann for dimension 1."""
  return HomogeneousBoundaryConditions(
      ((BCType.PERIODIC, BCType.PERIODIC),
       (BCType.NEUMANN, BCType.NEUMANN)))


def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
  """Returns True if arrays have periodic BC in every dimension, else False."""
  for array in arrays:
    for lower_bc_type, upper_bc_type in array.bc.types:
      if lower_bc_type != BCType.PERIODIC or upper_bc_type != BCType.PERIODIC:
        return False
  return True


def get_pressure_bc_from_velocity(v: GridVariableVector) -> BoundaryConditions:
  """Returns pressure boundary conditions for the specified velocity."""
  # Expect each component of v to have the same BC, either both PERIODIC or
  # both DIRICHLET.
  velocity_bc_types = grids.consistent_boundary_conditions(*v).types
  pressure_bc_types = []
  for velocity_bc_lower, velocity_bc_upper in velocity_bc_types:
    if velocity_bc_lower == BCType.PERIODIC:
      pressure_bc_lower = BCType.PERIODIC
    elif velocity_bc_lower == BCType.DIRICHLET:
      pressure_bc_lower = BCType.NEUMANN
    else:
      raise ValueError('Expected periodic or dirichlete velocity BC, '
                       f'got {velocity_bc_lower}')
    if velocity_bc_upper == BCType.PERIODIC:
      pressure_bc_upper = BCType.PERIODIC
    elif velocity_bc_upper == BCType.DIRICHLET:
      pressure_bc_upper = BCType.NEUMANN
    else:
      raise ValueError('Expected periodic or dirichlete velocity BC, '
                       f'got {velocity_bc_upper}')
    pressure_bc_types.append((pressure_bc_lower, pressure_bc_upper))
  return HomogeneousBoundaryConditions(pressure_bc_types)

