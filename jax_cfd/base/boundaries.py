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
from typing import Sequence, Tuple, Optional, Union
from jax import lax
import jax.numpy as jnp
from jax_cfd.base import grids
import numpy as np

BoundaryConditions = grids.BoundaryConditions
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Array = Union[np.ndarray, jnp.DeviceArray]


class BCType:
  PERIODIC = 'periodic'
  DIRICHLET = 'dirichlet'
  NEUMANN = 'neumann'


@dataclasses.dataclass(init=False, frozen=True)
class ConstantBoundaryConditions(BoundaryConditions):
  """Boundary conditions for a PDE variable that are constant in space and time.

  Example usage:
    grid = Grid((10, 10))
    array = GridArray(np.zeros((10, 10)), offset=(0.5, 0.5), grid)
    bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                        (BCType.DIRICHLET, BCType.DIRICHLET)),
                                        ((0.0, 10.0),(1.0, 0.0)))
    u = GridVariable(array, bc)

  Attributes:
    types: `types[i]` is a tuple specifying the lower and upper BC types for
      dimension `i`.
  """
  types: Tuple[Tuple[str, str], ...]
  bc_values: Tuple[Tuple[Optional[float], Optional[float]], ...]

  def __init__(self, types: Sequence[Tuple[str, str]],
               values: Sequence[Tuple[Optional[float], Optional[float]]]):
    types = tuple(types)
    values = tuple(values)
    object.__setattr__(self, 'types', types)
    object.__setattr__(self, 'bc_values', values)

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
    """Pad a GridArray.

    For dirichlet boundary, u is mirrored.

    Important: For jax_cfd finite difference/finite-volume code, no more than 1
    ghost cell is required. More ghost cells are used only in LES filtering/CNN
    application.

    No padding past 1 ghost cell is implemented for Neumann BC.

    Args:
      u: a `GridArray` object.
      width: number of elements to pad along axis. Use negative value for lower
        boundary or positive value for upper boundary.
      axis: axis to pad along.

    Returns:
      Padded array, elongated along the indicated axis.
    """

    def make_padding(width):
      if width < 0:  # pad lower boundary
        bc_type = self.types[axis][0]
        padding = (-width, 0)
      else:  # pad upper boundary
        bc_type = self.types[axis][1]
        padding = (0, width)

      full_padding = [(0, 0)] * u.grid.ndim
      full_padding[axis] = padding
      return full_padding, padding, bc_type

    full_padding, padding, bc_type = make_padding(width)
    offset = list(u.offset)
    offset[axis] -= padding[0]
    data = u.data
    if not (bc_type == BCType.PERIODIC or
            bc_type == BCType.DIRICHLET) and abs(width) > 1:
      raise ValueError(
          f'Padding past 1 ghost cell is not defined in {bc_type} case.')

    if bc_type == BCType.PERIODIC:
      # for periodic, all grid points must be there. Otherwise padding doesn't
      # make sense.
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')

      # self.values are ignored here
      pad_kwargs = dict(mode='wrap')
      # first, need to remove all existing padding, or the padding will be wrong
      # because it will wrap an already wrapped section

      # checking if there was any padding already added
      if not sum(abs(np.array(data.shape) - np.array(u.grid.shape))):
        data = jnp.pad(data, full_padding, **pad_kwargs)
      else:
        negative_trim = 0
        data = grids.GridArray(data, u.offset, u.grid)
        if u.offset[axis] < 0:
          negative_trim = -round(-u.offset[axis])
          data = self._trim(data, negative_trim, axis)
        positive_trim = data.data.shape[axis] - u.grid.shape[axis]
        if positive_trim > 0:
          data = self._trim(data, positive_trim, axis)
        # combining existing padding with new padding
        padding = tuple(pad + pad_trimmed
                        for pad, pad_trimmed in zip(padding, (negative_trim,
                                                              positive_trim)))
        full_padding[axis] = padding
        data = jnp.pad(data.data, full_padding, **pad_kwargs)

    elif bc_type == BCType.DIRICHLET:
      if np.isclose(u.offset[axis] % 1, 0.5):  # cell center
        # make the linearly interpolated value equal to the boundary by setting
        # the padded values to the negative symmetric values
        # for dirichlet 0.5 offset, all grid points must be there.
        # Otherwise padding doesn't make sense.
        if u.grid.shape[axis] > u.shape[axis]:
          raise ValueError('the GridArray shape does not match the grid.')
        data = (2 * jnp.pad(
            data, full_padding, mode='constant', constant_values=self.bc_values)
                - jnp.pad(data, full_padding, mode='symmetric'))
      elif np.isclose(u.offset[axis] % 1, 0):  # cell edge
        # u specifies the values on the interior CV. Thus, first the value on
        # the boundary needs to be added to the array, if not specified by the
        # interior CV values.
        # Then the mirrored ghost cells need to be appended.

        # for dirichlet cell-face aligned offset, 1 grid_point can be missing.
        # Otherwise padding doesn't make sense.
        if u.grid.shape[axis] > u.shape[axis] + 1:
          raise ValueError('the GridArray shape does not match the grid.')

        if (np.isclose(u.offset[axis], 0) and
            width > 0) or (np.isclose(u.offset[axis], 1) and width < 0):
          if np.isclose(abs(width), 1):
            data = jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)
          elif abs(width) > 1:
            bc_padding, _, _ = make_padding(int(width /
                                                abs(width)))  # makes it 1 pad
            # subtract the padded cell
            full_padding_past_bc, _, _ = make_padding(
                (abs(width) - 1) * int(width / abs(width)))  # makes it 1 pad
            # here we are adding 0 boundary cell with 0 value
            expanded_data = jnp.pad(
                data, bc_padding, mode='constant', constant_values=(0, 0))
            padding_values = list(self.bc_values)
            padding_values[axis] = [pad / 2 for pad in padding_values[axis]]
            data = 2 * jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=tuple(padding_values)) - jnp.pad(
                    expanded_data, full_padding_past_bc, mode='reflect')
        else:
          padding_values = list(self.bc_values)
          padding_values[axis] = [pad / 2 for pad in padding_values[axis]]
          data = 2 * jnp.pad(
              data,
              full_padding,
              mode='constant',
              constant_values=tuple(padding_values)) - jnp.pad(
                  data, full_padding, mode='reflect')
      else:
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
    elif bc_type == BCType.NEUMANN:
      # for neumann, all grid points must be there.
      # Otherwise padding doesn't make sense.
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')
      if not (np.isclose(u.offset[axis] % 1, 0) or
              np.isclose(u.offset[axis] % 1, 0.5)):
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
      else:
        # When the data is cell-centered, computes the backward difference.
        # When the data is on cell edges, boundary is set such that
        # (u_last_interior - u_boundary)/grid_step = neumann_bc_value.
        data = (
            jnp.pad(data, full_padding, mode='edge') + u.grid.step[axis] *
            (jnp.pad(data, full_padding, mode='constant') - jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)))
    else:
      raise ValueError('invalid boundary type')

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

  def values(
      self, axis: int,
      grid: grids.Grid) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Returns boundary values on the grid along axis.

    Args:
      axis: axis along which to return boundary values.
      grid: a `Grid` object on which to evaluate boundary conditions.

    Returns:
      A tuple of arrays of grid.ndim - 1 dimensions that specify values on the
      boundary. In case of periodic boundaries, returns a tuple(None,None).
    """
    if None in self.bc_values[axis]:
      return (None, None)
    bc = tuple(
        jnp.full(grid.shape[:axis] +
                 grid.shape[axis + 1:], self.bc_values[axis][-i])
        for i in [0, 1])
    return bc

  def trim_bc(self, u: grids.GridArray) -> grids.GridArray:
    """Returns GridArray without the grid points on the boundary.

    Some grid points of GridArray might coincide with boundary. This trims those
    values.
    Args:
      u: a `GridArray` object.

    Returns:
      A GridArray shrunk along certain dimensions.
    """
    if u.shape != u.grid.shape:
      raise ValueError('the GridArray shape does not match the grid.')
    for axis in range(u.grid.ndim):
      if np.isclose(u.offset[axis],
                    0.0) and self.types[axis][0] == BCType.DIRICHLET:
        u = self._trim(u, -1, axis)
      elif np.isclose(u.offset[axis],
                      1.0) and self.types[axis][1] == BCType.DIRICHLET:
        u = self._trim(u, 1, axis)
    return u

  def pad_and_impose_bc(self, u: grids.GridArray) -> grids.GridArray:
    """Returns GridVariable with correct boundary condition.

    Some grid points of GridArray might coincide with boundary. This ensures
    that the GridVariable.array agrees with GridVariable.bc.
    Args:
      u: a `GridArray` object that specifies only scalar values on the internal
        nodes.

    Returns:
      A GridVariable that has correct boundary.
    """
    for axis in range(u.grid.ndim):
      if np.isclose(u.offset[axis],
                    0.0) and self.types[axis][0] == BCType.DIRICHLET:
        u = self._pad(u, -1, axis)
      elif np.isclose(u.offset[axis],
                      1.0) and self.types[axis][1] == BCType.DIRICHLET:
        u = self._pad(u, 1, axis)
      elif u.shape[axis] != u.grid.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')
    return u

  def impose_bc(self, u: grids.GridArray) -> grids.GridVariable:
    """Returns GridVariable with correct boundary condition.

    Some grid points of GridArray might coincide with boundary. This ensures
    that the GridVariable.array agrees with GridVariable.bc.
    Args:
      u: a `GridArray` object.

    Returns:
      A GridVariable that has correct boundary.
    """
    if u.shape == u.grid.shape:
      u = self.trim_bc(u)
    return grids.GridVariable(self.pad_and_impose_bc(u), self)

  trim = _trim
  pad = _pad


class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
  """Boundary conditions for a PDE variable.

  Example usage:
    grid = Grid((10, 10))
    array = GridArray(np.zeros((10, 10)), offset=(0.5, 0.5), grid)
    bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                        (BCType.DIRICHLET, BCType.DIRICHLET)))
    u = GridVariable(array, bc)

  Attributes:
    types: `types[i]` is a tuple specifying the lower and upper BC types for
      dimension `i`.
  """

  def __init__(self, types: Sequence[Tuple[str, str]]):

    ndim = len(types)
    values = ((0.0, 0.0),) * ndim
    super(HomogeneousBoundaryConditions, self).__init__(types, values)


# Convenience utilities to ease updating of BoundaryConditions implementation
def periodic_boundary_conditions(ndim: int) -> ConstantBoundaryConditions:
  """Returns periodic BCs for a variable with `ndim` spatial dimension."""
  return HomogeneousBoundaryConditions(
      ((BCType.PERIODIC, BCType.PERIODIC),) * ndim)


def dirichlet_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """Returns Dirichelt BCs for a variable with `ndim` spatial dimension.

  Args:
    ndim: spatial dimension.
    bc_vals: A tuple of lower and upper boundary values for each dimension.
      If None, returns Homogeneous BC.

  Returns:
    BoundaryCondition subclass.
  """
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim)
  else:
    return ConstantBoundaryConditions(
        ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim, bc_vals)


def neumann_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """Returns Neumann BCs for a variable with `ndim` spatial dimension.

  Args:
    ndim: spatial dimension.
    bc_vals: A tuple of lower and upper boundary values for each dimension.
      If None, returns Homogeneous BC.

  Returns:
    BoundaryCondition subclass.
  """
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.NEUMANN, BCType.NEUMANN),) * ndim)
  else:
    return ConstantBoundaryConditions(
        ((BCType.NEUMANN, BCType.NEUMANN),) * ndim, bc_vals)


def channel_flow_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.

  Args:
    ndim: spatial dimension.
    bc_vals: A tuple of lower and upper boundary values for each dimension.
      If None, returns Homogeneous BC. For periodic dimensions the lower, upper
      boundary values should be (None, None).

  Returns:
    BoundaryCondition subclass.
  """
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
  if not bc_vals:
    return HomogeneousBoundaryConditions(bc_type)
  else:
    return ConstantBoundaryConditions(bc_type, bc_vals)


def periodic_and_neumann_boundary_conditions(
    bc_vals: Optional[Tuple[float,
                            float]] = None) -> ConstantBoundaryConditions:
  """Returns BCs periodic for dimension 0 and Neumann for dimension 1.

  Args:
    bc_vals: the lower and upper boundary condition value for each dimension. If
      None, returns Homogeneous BC.

  Returns:
    BoundaryCondition subclass.
  """
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)))
  else:
    return ConstantBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)),
        ((None, None), bc_vals))


def periodic_and_dirichlet_boundary_conditions(
    bc_vals: Optional[Tuple[float,
                            float]] = None) -> ConstantBoundaryConditions:
  """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.

  Args:
    bc_vals: the lower and upper boundary condition value for each dimension. If
      None, returns Homogeneous BC.

  Returns:
    BoundaryCondition subclass.
  """
  if not bc_vals:
    return HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)))
  else:
    return ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                       (BCType.DIRICHLET, BCType.DIRICHLET)),
                                      ((None, None), bc_vals))


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
