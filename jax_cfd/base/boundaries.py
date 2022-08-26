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
import math
from typing import Optional, Sequence, Tuple, Union

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


class Padding:
  MIRROR = 'mirror'
  EXTEND = 'extend'


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
      mode: Optional[str] = None,
  ) -> GridArray:
    """Shift an GridArray by `offset`.

    Args:
      u: an `GridArray` object.
      offset: positive or negative integer offset to shift.
      axis: axis to shift along.
      mode: type of padding to use in non-periodic case.
        Mirror mirrors the flow across the boundary.
        Extend extends the last well-defined value past the boundary.

    Returns:
      A copy of `u`, shifted by `offset`. The returned `GridArray` has offset
      `u.offset + offset`.
    """
    padded = self._pad(u, offset, axis, mode=mode)
    trimmed = self._trim(padded, -offset, axis)
    return trimmed

  def _is_aligned(self, u: GridArray, axis: int) -> bool:
    """Checks if array u contains all interior domain information.

    For dirichlet edge aligned boundary, the value that lies exactly on the
    boundary does not have to be specified by u.
    Neumann edge aligned boundary is not defined.

    Args:
      u: array that should contain interior data
      axis: axis along which to check

    Returns:
      True if u is aligned, and raises error otherwise.
    """
    size_diff = u.shape[axis] - u.grid.shape[axis]
    if self.types[axis][0] == BCType.DIRICHLET and np.isclose(
        u.offset[axis], 1):
      size_diff += 1
    if self.types[axis][1] == BCType.DIRICHLET and np.isclose(
        u.offset[axis], 1):
      size_diff += 1
    if self.types[axis][0] == BCType.NEUMANN and np.isclose(
        u.offset[axis] % 1, 0):
      raise NotImplementedError('Edge-aligned neumann BC are not implemented.')
    if size_diff < 0:
      raise ValueError(
          'the GridArray does not contain all interior grid values.')
    return True

  def _pad(
      self,
      u: GridArray,
      width: int,
      axis: int,
      mode: Optional[str] = None,
  ) -> GridArray:
    """Pad a GridArray.

    For dirichlet boundary, u is mirrored.

    Important: For jax_cfd finite difference/finite-volume code, no more than 1
    ghost cell is required. More ghost cells are used only in LES filtering/CNN
    application.

    Args:
      u: a `GridArray` object.
      width: number of elements to pad along axis. Use negative value for lower
        boundary or positive value for upper boundary.
      axis: axis to pad along.
      mode: type of padding to use in non-periodic case.
        Mirror mirrors the array values across the boundary.
        Extend extends the last well-defined array value past the boundary.
        Mode is only needed if the padding extends past array values that are
          defined by the physics. In these cases, no mode is necessary. This
          also means periodic boundaries do not require a mode.

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
    if bc_type == BCType.PERIODIC:
      need_trimming = 'both'  # need to trim both sides
    elif width >= 0:
      need_trimming = 'right'  # only one side needs to be trimmed
    else:
      need_trimming = 'left'  # only one side needs to be trimmed
    u, trimmed_padding = self._trim_padding(u, axis, need_trimming)
    data = u.data
    full_padding[axis] = tuple(
        pad + trimmed_pad
        for pad, trimmed_pad in zip(full_padding[axis], trimmed_padding))

    if bc_type == BCType.PERIODIC:
      # for periodic, all grid points must be there. Otherwise padding doesn't
      # make sense.

      # self.values are ignored here
      pad_kwargs = dict(mode='wrap')
      data = jnp.pad(data, full_padding, **pad_kwargs)

    elif bc_type == BCType.DIRICHLET:
      if np.isclose(u.offset[axis] % 1, 0.5):  # cell center
        # If only one or 0 value is needed, no mode is necessary.
        # All modes would return the same values.
        if np.isclose(sum(full_padding[axis]), 1) or np.isclose(
            sum(full_padding[axis]), 0):
          mode = Padding.MIRROR

        if mode == Padding.MIRROR:
          # make the linearly interpolated value equal to the boundary by
          # setting the padded values to the negative symmetric values
          data = (2 * jnp.pad(
              data,
              full_padding,
              mode='constant',
              constant_values=self.bc_values) -
                  jnp.pad(data, full_padding, mode='symmetric'))
        elif mode == Padding.EXTEND:
          # computes the well-defined ghost cell and sets the rest of padding
          # values equal to the ghost cell.
          data = (2 * jnp.pad(
              data,
              full_padding,
              mode='constant',
              constant_values=self.bc_values) -
                  jnp.pad(data, full_padding, mode='edge'))
        else:
          raise NotImplementedError(f'Mode {mode} is not implemented yet.')
      elif np.isclose(u.offset[axis] % 1, 0):  # cell edge
        # u specifies the values on the interior CV. Thus, first the value on
        # the boundary needs to be added to the array, if not specified by the
        # interior CV values.
        # Then the mirrored ghost cells need to be appended.

        # if only one value is needed, no mode is necessary.
        if np.isclose(sum(full_padding[axis]), 1) or np.isclose(
            sum(full_padding[axis]), 0):
          data = jnp.pad(
              data,
              full_padding,
              mode='constant',
              constant_values=self.bc_values)
        elif sum(full_padding[axis]) > 1:
          if mode == Padding.MIRROR:
            # make boundary-only padding
            bc_padding = [(0, 0)] * u.grid.ndim
            bc_padding[axis] = tuple(
                1 if pad > 0 else 0 for pad in full_padding[axis])
            # subtract the padded cell
            full_padding_past_bc = [(0, 0)] * u.grid.ndim
            full_padding_past_bc[axis] = tuple(
                pad - 1 if pad > 0 else 0 for pad in full_padding[axis])
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
          elif mode == Padding.EXTEND:
            data = jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)
          else:
            raise NotImplementedError(f'Mode {mode} is not implemented yet.')
      else:
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
    elif bc_type == BCType.NEUMANN:
      if not np.isclose(u.offset[axis] % 1, 0.5):
        raise ValueError(
            'expected offset to be cell center for neumann bc, got '
            f'offset[axis]={u.offset[axis]}')
      else:
        # When the data is cell-centered, computes the backward difference.

        # if only one value is needed, no mode is necessary. Default mode is
        # provided, although all modes would return the same values.
        if np.isclose(sum(full_padding[axis]), 1) or np.isclose(
            sum(full_padding[axis]), 0):
          np_mode = 'symmetric'
        elif mode == Padding.MIRROR:
          np_mode = 'symmetric'
        elif mode == Padding.EXTEND:
          np_mode = 'edge'
        else:
          raise NotImplementedError(f'Mode {mode} is not implemented yet.')
        # ensures that finite_differences.backward_difference satisfies the
        # boundary condition.
        derivative_direction = float(width // max(1, abs(width)))
        data = (
            jnp.pad(data, full_padding, mode=np_mode) -
            derivative_direction * u.grid.step[axis] *
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

  def _trim_padding(self,
                    u: grids.GridArray,
                    axis: int = 0,
                    trim_side: str = 'both'):
    """Trims padding from a GridArray along axis and returns the array interior.

    Args:
      u: a `GridArray` object.
      axis: axis to trim along.
      trim_side: if 'both', trims both sides. If 'right', trims the right side.
        If 'left', the left side.

    Returns:
      Trimmed array, shrunk along the indicated axis side.
    """
    padding = (0, 0)
    if u.shape[axis] >= u.grid.shape[axis]:
      # number of cells that were padded on the left
      negative_trim = 0
      if u.offset[axis] <= 0 and (trim_side == 'both' or trim_side == 'left'):
        negative_trim = -math.ceil(-u.offset[axis])
        # periodic is a special case. Shifted data might still contain all the
        # information.
        if self.types[axis][0] == BCType.PERIODIC:
          negative_trim = max(negative_trim, u.grid.shape[axis] - u.shape[axis])
        # for both DIRICHLET and NEUMANN cases the value on grid.domain[0] is
        # a dependent value.
        elif np.isclose(u.offset[axis] % 1, 0):
          negative_trim -= 1
        u = self._trim(u, negative_trim, axis)
      # number of cells that were padded on the right
      positive_trim = 0
      if (trim_side == 'right' or trim_side == 'both'):
        # periodic is a special case. Boundary on one side depends on the other
        # side.
        if self.types[axis][1] == BCType.PERIODIC:
          positive_trim = max(u.shape[axis] - u.grid.shape[axis], 0)
        else:
          # for other cases, where to trim depends only on the boundary type
          # and data offset.
          last_u_offset = u.shape[axis] + u.offset[axis] - 1
          boundary_offset = u.grid.shape[axis]
          if last_u_offset >= boundary_offset:
            positive_trim = math.ceil(last_u_offset - boundary_offset)
            if self.types[axis][1] == BCType.DIRICHLET and np.isclose(
                u.offset[axis] % 1, 0):
              positive_trim += 1
      if positive_trim > 0:
        u = self._trim(u, positive_trim, axis)
      # combining existing padding with new padding
      padding = (-negative_trim, positive_trim)
    return u, padding

  def pad(self,
          u: GridArray,
          width: Union[Tuple[int, int], int],
          axis: int,
          mode: Optional[str] = None,) -> GridArray:
    """Wrapper for _pad.

    Args:
      u: a `GridArray` object.
      width: number of elements to pad along axis. If width is an int, use
        negative value for lower boundary or positive value for upper boundary.
        If a tuple, pads with width[0] on the left and width[1] on the right.
      axis: axis to pad along.
      mode: type of padding to use in non-periodic case.
        Mirror mirrors the array values across the boundary.
        Extend extends the last well-defined array value past the boundary.

    Returns:
      Padded array, elongated along the indicated axis.
    """
    _ = self._is_aligned(u, axis)
    if isinstance(width, int):
      u = self._pad(u, width, axis, mode=mode)
    else:
      u = self._pad(u, -width[0], axis, mode=mode)
      u = self._pad(u, width[1], axis, mode=mode)
    return u

  def pad_all(self,
              u: GridArray,
              width: Tuple[Tuple[int, int], ...],
              mode: Optional[str] = None,) -> GridArray:
    """Pads along all axes with pad width specified by width tuple.

    Args:
      u: a `GridArray` object.
      width: Tuple of padding width for each side for each axis.
      mode: type of padding to use in non-periodic case.
        Mirror mirrors the array values across the boundary.
        Extend extends the last well-defined array value past the boundary.

    Returns:
      Padded array, elongated along all axes.
    """
    for axis in range(u.grid.ndim):
      u = self.pad(u, width[axis], axis, mode=mode)
    return u

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
    bc_values = tuple(
        jnp.full(grid.shape[:axis] +
                 grid.shape[axis + 1:], self.bc_values[axis][-i])
        for i in [0, 1])
    return bc_values

  def trim_boundary(self, u: grids.GridArray) -> grids.GridArray:
    """Returns GridArray without the grid points on the boundary.

    Some grid points of GridArray might coincide with boundary. This trims those
    values. If the array was padded beforehand, removes the padding.

    Args:
      u: a `GridArray` object.

    Returns:
      A GridArray shrunk along certain dimensions.
    """
    for axis in range(u.grid.ndim):
      _ = self._is_aligned(u, axis)
      u, _ = self._trim_padding(u, axis)
    return u

  def pad_and_impose_bc(
      self,
      u: grids.GridArray,
      offset_to_pad_to: Optional[Tuple[float,
                                       ...]] = None) -> grids.GridVariable:
    """Returns GridVariable with correct boundary values.

    Some grid points of GridArray might coincide with boundary. This ensures
    that the GridVariable.array agrees with GridVariable.bc.
    Args:
      u: a `GridArray` object that specifies only scalar values on the internal
        nodes.
      offset_to_pad_to: a Tuple of desired offset to pad to. Note that if the
        function is given just an interior array in dirichlet case, it can pad
        to both 0 offset and 1 offset.

    Returns:
      A GridVariable that has correct boundary values.
    """
    if offset_to_pad_to is None:
      offset_to_pad_to = u.offset
    for axis in range(u.grid.ndim):
      _ = self._is_aligned(u, axis)
      if self.types[axis][0] == BCType.DIRICHLET and np.isclose(
          u.offset[axis], 1.0):
        if np.isclose(offset_to_pad_to[axis], 1.0):
          u = self._pad(u, 1, axis)
        elif np.isclose(offset_to_pad_to[axis], 0.0):
          u = self._pad(u, -1, axis)
    return grids.GridVariable(u, self)

  def impose_bc(self, u: grids.GridArray) -> grids.GridVariable:
    """Returns GridVariable with correct boundary condition.

    Some grid points of GridArray might coincide with boundary. This ensures
    that the GridVariable.array agrees with GridVariable.bc.
    Args:
      u: a `GridArray` object.

    Returns:
      A GridVariable that has correct boundary values and is restricted to the
      domain.
    """
    offset = u.offset
    u = self.trim_boundary(u)
    return self.pad_and_impose_bc(u, offset)

  trim = _trim


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
    BoundaryCondition instance.
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
    BoundaryCondition instance.
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
    BoundaryCondition instance.
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
    BoundaryCondition instance.
  """
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)))
  else:
    return ConstantBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)),
        ((None, None), bc_vals))


def periodic_and_dirichlet_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
    periodic_axis=0) -> ConstantBoundaryConditions:
  """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.

  Args:
    bc_vals: the lower and upper boundary condition value for each dimension. If
      None, returns Homogeneous BC.
    periodic_axis: specifies which axis is periodic.

  Returns:
    BoundaryCondition subclass.
  """
  periodic = (BCType.PERIODIC, BCType.PERIODIC)
  dirichlet = (BCType.DIRICHLET, BCType.DIRICHLET)
  if periodic_axis == 0:
    if not bc_vals:
      return HomogeneousBoundaryConditions((periodic, dirichlet))
    else:
      return ConstantBoundaryConditions((periodic, dirichlet),
                                        ((None, None), bc_vals))
  else:
    if not bc_vals:
      return HomogeneousBoundaryConditions((dirichlet, periodic))
    else:
      return ConstantBoundaryConditions((dirichlet, periodic),
                                        (bc_vals, (None, None)))


def is_periodic_boundary_conditions(c: grids.GridVariable, axis: int) -> bool:
  """Returns true if scalar has periodic bc along axis."""
  if c.bc.types[axis][0] != BCType.PERIODIC:
    return False
  return True


def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
  """Returns True if arrays have periodic BC in every dimension, else False."""
  for array in arrays:
    for axis in range(array.grid.ndim):
      if not is_periodic_boundary_conditions(array, axis):
        return False
  return True


def consistent_boundary_conditions(*arrays: GridVariable) -> Tuple[str, ...]:
  """Returns whether BCs are periodic.

  Mixed periodic/nonperiodic boundaries along the same boundary do not make
  sense. The function checks that the boundary is either periodic or not and
  throws an error if its mixed.

  Args:
    *arrays: a list of gridvariables.

  Returns:
    a list of types of boundaries corresponding to each axis if
    they are consistent.
  """
  bc_types = []
  for axis in range(arrays[0].grid.ndim):
    bcs = {is_periodic_boundary_conditions(array, axis) for array in arrays}
    if len(bcs) != 1:
      raise grids.InconsistentBoundaryConditionsError(
          f'arrays do not have consistent bc: {arrays}')
    elif bcs.pop():
      bc_types.append('periodic')
    else:
      bc_types.append('nonperiodic')
  return tuple(bc_types)


def get_pressure_bc_from_velocity(
    v: GridVariableVector) -> HomogeneousBoundaryConditions:
  """Returns pressure boundary conditions for the specified velocity."""
  # assumes that if the boundary is not periodic, pressure BC is zero flux.
  velocity_bc_types = consistent_boundary_conditions(*v)
  pressure_bc_types = []
  for velocity_bc_type in velocity_bc_types:
    if velocity_bc_type == 'periodic':
      pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    else:
      pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
  return HomogeneousBoundaryConditions(pressure_bc_types)


def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable,
    flux_direction: int) -> ConstantBoundaryConditions:
  """Returns advection flux boundary conditions for the specified velocity.

  Infers advection flux boundary condition in flux direction
  from scalar c and velocity u in direction flux_direction.
  The flux boundary condition should be used only to compute divergence.
  If the boundaries are periodic, flux is periodic.
  In nonperiodic case, flux boundary parallel to flux direction is
  homogeneous dirichlet.
  In nonperiodic case if flux direction is normal to the wall, the
  function supports 2 cases:
    1) Nonporous boundary, corresponding to homogeneous flux bc.
    2) Pourous boundary with constant flux, corresponding to
      both the velocity and scalar with Homogeneous Neumann bc.

  This function supports only these cases because all other cases result in
  time dependent flux boundary condition.

  Args:
    u: velocity component in flux_direction.
    c: scalar to advect.
    flux_direction: direction of velocity.

  Returns:
    BoundaryCondition instance for advection flux of c in flux_direction.
  """
  # only no penetration and periodic boundaries are supported.
  flux_bc_types = []
  flux_bc_values = []
  if not isinstance(u.bc, HomogeneousBoundaryConditions):
    raise NotImplementedError(
        f'Flux boundary condition is not implemented for velocity with {u.bc}')
  for axis in range(c.grid.ndim):
    if u.bc.types[axis][0] == 'periodic':
      flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
      flux_bc_values.append((None, None))
    elif flux_direction != axis:
      # This is not technically correct. Flux boundary condition in most cases
      # is a time dependent function of the current values of the scalar
      # and velocity. However, because flux is used only to take divergence, the
      # boundary condition on the flux along the boundary parallel to the flux
      # direction has no influence on the computed divergence because the
      # boundary condition only alters ghost cells, while divergence is computed
      # on the interior.
      # To simplify the code and allow for flux to be wrapped in gridVariable,
      # we are setting the boundary to homogeneous dirichlet.
      # Note that this will not work if flux is used in any other capacity but
      # to take divergence.
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
      flux_bc_values.append((0.0, 0.0))
    else:
      flux_bc_types_ax = []
      flux_bc_values_ax = []
      for i in range(2):  # lower and upper boundary.

        # case 1: nonpourous boundary
        if (u.bc.types[axis][i] == BCType.DIRICHLET and
            u.bc.bc_values[axis][i] == 0.0):
          flux_bc_types_ax.append(BCType.DIRICHLET)
          flux_bc_values_ax.append(0.0)

        # case 2: zero flux boundary
        elif (u.bc.types[axis][i] == BCType.NEUMANN and
              c.bc.types[axis][i] == BCType.NEUMANN):
          if not isinstance(c.bc, ConstantBoundaryConditions):
            raise NotImplementedError(
                'Flux boundary condition is not implemented for scalar' +
                f' with {c.bc}')
          if not np.isclose(c.bc.bc_values[axis][i], 0.0):
            raise NotImplementedError(
                'Flux boundary condition is not implemented for scalar' +
                f' with {c.bc}')
          flux_bc_types_ax.append(BCType.NEUMANN)
          flux_bc_values_ax.append(0.0)

        # no other case is supported
        else:
          raise NotImplementedError(
              f'Flux boundary condition is not implemented for {u.bc, c.bc}')
      flux_bc_types.append(flux_bc_types_ax)
      flux_bc_values.append(flux_bc_values_ax)
  return ConstantBoundaryConditions(flux_bc_types, flux_bc_values)
