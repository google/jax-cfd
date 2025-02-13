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
from typing import Any, Callable, Iterable, Sequence, Tuple, Optional, Union
from jax import lax
import jax
import jax.numpy as jnp
from jax_ib.base import grids
import numpy as np
import scipy
from jax.tree_util import register_pytree_node_class
from jax_ib.base import particle_class

BoundaryConditions = grids.BoundaryConditions
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Array = Union[np.ndarray, jax.Array]
BCArray = grids.BCArray


class BCType:
  PERIODIC = 'periodic'
  DIRICHLET = 'dirichlet'
  NEUMANN = 'neumann'

@register_pytree_node_class
@dataclasses.dataclass(init=False, frozen=False)
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
  boundary_fn: Callable[...,Optional[float]]
  time_stamp: Optional[float]
  def __init__(self, 
               time_stamp: Optional[float],values: Sequence[Tuple[Optional[float], Optional[float]]],types: Sequence[Tuple[str, str]],boundary_fn:Callable[...,Optional[float]]):
    types = tuple(types)
    values = tuple(values)
    boundary_fn = boundary_fn
    time_stamp = time_stamp
    
    object.__setattr__(self, 'bc_values', values)
    object.__setattr__(self, 'boundary_fn', boundary_fn)
    object.__setattr__(self, 'time_stamp', time_stamp if time_stamp is not None else [])
    object.__setattr__(self, 'types', types)
    #if boundary_fn or not boundary_fn:
      
    #else:
    #  object.__setattr__(self, 'boundary_fn', None)
    # object.__setattr__(self, 'time_stamp', None)


  def tree_flatten(self):
    """Returns flattening recipe for GridVariable JAX pytree."""
    children = (self.time_stamp,self.bc_values,)
    aux_data = (self.types,self.boundary_fn)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Returns unflattening recipe for GridVariable JAX pytree."""
    return cls(*children, *aux_data)


  def update_bc_(self,time_stamp: float, dt: float):
    return time_stamp + dt
       

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
    if not (bc_type == BCType.PERIODIC or
            bc_type == BCType.DIRICHLET) and abs(width) > 1:
      raise ValueError(
          f'Padding past 1 ghost cell is not defined in {bc_type} case.')

    u, trimmed_padding = self._trim_padding(u)
    data = u.data
    full_padding[axis] = tuple(
        pad + trimmed_pad
        for pad, trimmed_pad in zip(full_padding[axis], trimmed_padding))

    if bc_type == BCType.PERIODIC:
      # for periodic, all grid points must be there. Otherwise padding doesn't
      # make sense.
      # Don't pad a trimmed periodic array.
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')
      # self.values are ignored here
      pad_kwargs = dict(mode='wrap')
      data = jnp.pad(data, full_padding, **pad_kwargs)

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
          raise ValueError('For a dirichlet cell-face boundary condition, ' +
                           'the GridArray has more than 1 grid point missing.')
        elif u.grid.shape[axis] == u.shape[axis] + 1 and not np.isclose(
            u.offset[axis], 1):
          raise ValueError('For a dirichlet cell-face boundary condition, ' +
                           'the GridArray has more than 1 grid point missing.')

        def _needs_pad_with_boundary_value():
          if (np.isclose(u.offset[axis], 0) and
              width > 0) or (np.isclose(u.offset[axis], 1) and width < 0):
            return True
          elif u.grid.shape[axis] == u.shape[axis] + 1:
            return True
          else:
            return False

        if _needs_pad_with_boundary_value():
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
        else:  # dirichlet cell-face aligned
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

  def _trim_padding(self, u: grids.GridArray, axis=0):
    """Trim all padding from a GridArray.

    Args:
      u: a `GridArray` object.
      axis: axis to trim along.

    Returns:
      Trimmed array, shrunk along the indicated axis to match
      u.grid.shape[axis].
    """
    padding = (0, 0)
    if u.shape[axis] > u.grid.shape[axis]:
      # number of cells that were padded on the left
      negative_trim = 0
      if u.offset[axis] < 0:
        negative_trim = -round(-u.offset[axis])
        u = self._trim(u, negative_trim, axis)
      # number of cells that were padded on the right
      positive_trim = u.shape[axis] - u.grid.shape[axis]
      if positive_trim > 0:
        u = self._trim(u, positive_trim, axis)
      # combining existing padding with new padding
      padding = (negative_trim, positive_trim)
    return u, padding

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
      u, _ = self._trim_padding(u, axis)
    if u.shape != u.grid.shape:
      raise ValueError('the GridArray has already been trimmed.')
    for axis in range(u.grid.ndim):
      if np.isclose(u.offset[axis],
                    0.0) and self.types[axis][0] == BCType.DIRICHLET:
        u = self._trim(u, -1, axis)
      elif np.isclose(u.offset[axis],
                      1.0) and self.types[axis][1] == BCType.DIRICHLET:
        u = self._trim(u, 1, axis)
    return u

  def pad_and_impose_bc(
      self,
      u: grids.GridArray,
      offset_to_pad_to: Optional[Tuple[float,
                                       ...]] = None) -> grids.GridVariable:
    """Returns GridVariable with correct boundary condition.

    Some grid points of GridArray might coincide with boundary. This ensures
    that the GridVariable.array agrees with GridVariable.bc.
    Args:
      u: a `GridArray` object that specifies only scalar values on the internal
        nodes.
      offset_to_pad_to: a Tuple of desired offset to pad to. Note that if the
        function is given just an interior array in dirichlet case, it can pad
        to both 0 offset and 1 offset.

    Returns:
      A GridVariable that has correct boundary.
    """
    if offset_to_pad_to is None:
      offset_to_pad_to = u.offset
    for axis in range(u.grid.ndim):
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
      A GridVariable that has correct boundary.
    """
    offset = u.offset
    if u.shape == u.grid.shape:
      u = self.trim_boundary(u)
    return self.pad_and_impose_bc(u, offset)

  trim = _trim
  pad = _pad

@register_pytree_node_class
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
    bc_fn = lambda x: x
    time_stamp = 0.0
    super(HomogeneousBoundaryConditions, self).__init__(time_stamp, values,types,bc_fn)




@register_pytree_node_class
class TimeDependentBoundaryConditions(ConstantBoundaryConditions):
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

  def __init__(self, types: Sequence[Tuple[str, str]],values: Sequence[Tuple[Optional[float], Optional[float]]],boundary_fn: Callable[..., Optional[float]],time_stamp: Optional[float]):

    #ndim = len(types)
    #values = ((0.0, 0.0),) * ndim
    
    super(TimeDependentBoundaryConditions, self).__init__(types, values,boundary_fn,time_stamp)

  def tree_flatten(self):
    """Returns flattening recipe for GridVariable JAX pytree."""
    children = (self.bc_values,)
    aux_data = (self.time_stamp,self.types,self.boundary_fn,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Returns unflattening recipe for GridVariable JAX pytree."""
    return cls(*children, *aux_data)


def boundary_function(t):
  A=1
  B = 1
  freq = 1
  return 1+0*(A*jnp.cos(freq*t)+B*jnp.sin(freq*t))    

def Reserve_BC(all_variable: particle_class.All_Variables,step_time: float) -> particle_class.All_Variables:
    v = all_variable.velocity
    particles = all_variable.particles
    pressure = all_variable.pressure
    Drag = all_variable.Drag
    Step_count = all_variable.Step_count
    MD_var = all_variable.MD_var
    bcfn = v[0].bc.boundary_fn
    bcfny = v[1].bc.boundary_fn
    
    dt = step_time
    ts = v[0].bc.time_stamp + dt# v[0].bc.time_stamp #v[0].bc.update_bc_(v[0].bc.time_stamp,dt)
    #ts = dt
    vx_bc = ((bcfn[0](ts),bcfn[1](0.0)),(bcfn[2](ts),bcfn[3](0.0)))
    vy_bc = ((bcfny[0](ts),bcfny[1](0.0)),(bcfny[2](ts),bcfny[3](0.0)))
    #vel_bc =(Moving_wall_boundary_conditions(ndim=2,bc_vals=vx_bc,time_stamp=ts,bc_fn=bcfn),Moving_wall_boundary_conditions(ndim=2,bc_vals=vy_bc,time_stamp=ts,bc_fn=bcfn))
    vel_bc = (ConstantBoundaryConditions(values=vx_bc,time_stamp=ts,types=v[0].bc.types,boundary_fn=bcfn),
              ConstantBoundaryConditions(values=vy_bc,time_stamp=ts,types=v[1].bc.types,boundary_fn=bcfny))
    #return v
    #return tuple(grids.GridVariable(u.array, u.bc) for u in v)
   
    v_updated =  tuple(     
      grids.GridVariable(u.array, bc) for u, bc in zip(v, vel_bc))
    return particle_class.All_Variables(particles,v_updated,pressure,Drag,Step_count,MD_var)
  


def update_BC(all_variable: particle_class.All_Variables,step_time: float) -> particle_class.All_Variables:
    v = all_variable.velocity
    particles = all_variable.particles
    pressure = all_variable.pressure
    Drag = all_variable.Drag
    Step_count = all_variable.Step_count
    MD_var = all_variable.MD_var
    bcfn = v[0].bc.boundary_fn
    bcfny = v[1].bc.boundary_fn
    
    dt = step_time
    ts = v[0].bc.time_stamp + dt# v[0].bc.time_stamp #v[0].bc.update_bc_(v[0].bc.time_stamp,dt)
    #ts = dt
    vx_bc = ((bcfn[0](ts),bcfn[1](ts)),(bcfn[2](ts),bcfn[3](ts)))
    vy_bc = ((bcfny[0](ts),bcfny[1](ts)),(bcfny[2](ts),bcfny[3](ts)))
    #vel_bc =(Moving_wall_boundary_conditions(ndim=2,bc_vals=vx_bc,time_stamp=ts,bc_fn=bcfn),Moving_wall_boundary_conditions(ndim=2,bc_vals=vy_bc,time_stamp=ts,bc_fn=bcfn))
    vel_bc = (ConstantBoundaryConditions(values=vx_bc,time_stamp=ts,types=v[0].bc.types,boundary_fn=bcfn),
              ConstantBoundaryConditions(values=vy_bc,time_stamp=ts,types=v[1].bc.types,boundary_fn=bcfny))
    #return v
    #return tuple(grids.GridVariable(u.array, u.bc) for u in v)
   
    v_updated =  tuple(     
      grids.GridVariable(u.array, bc) for u, bc in zip(v, vel_bc))
    return particle_class.All_Variables(particles,v_updated,pressure,Drag,Step_count,MD_var)

# Convenience utilities to ease updating of BoundaryConditions implementation
def periodic_boundary_conditions(ndim: int) -> ConstantBoundaryConditions:
  """Returns periodic BCs for a variable with `ndim` spatial dimension."""
  return HomogeneousBoundaryConditions(
      ((BCType.PERIODIC, BCType.PERIODIC),) * ndim)



def Radom_velocity_conditions(ndim: int) -> ConstantBoundaryConditions:
    """Returns periodic BCs for a variable with `ndim` spatial dimension."""
    
    values = ((0.0, 0.0),) * ndim
    bc_fn = lambda x: x
    time_stamp = 0.0
    return Moving_wall_boundary_conditions(
    ndim,
    bc_vals=values,
    time_stamp=time_stamp,    
    bc_fn=bc_fn,) 


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




def Moving_wall_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],    
    bc_fn: Callable[...,Optional[float]],

) -> ConstantBoundaryConditions:
  """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.

  Args:
    ndim: spatial dimension.
    bc_fn: function describing the time dependent boundary condition
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

  
    

  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)


def Far_field_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],    
    bc_fn: Callable[...,Optional[float]],

) -> ConstantBoundaryConditions:
  """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.

  Args:
    ndim: spatial dimension.
    bc_fn: function describing the time dependent boundary condition
    bc_vals: A tuple of lower and upper boundary values for each dimension.
      If None, returns Homogeneous BC. For periodic dimensions the lower, upper
      boundary values should be (None, None).

  Returns:
    BoundaryCondition instance.
  """
  bc_type = ((BCType.DIRICHLET, BCType.DIRICHLET),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.DIRICHLET, BCType.DIRICHLET),)

  
    

  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)

def find_extremum(fn,extrema,i_guess):
    if extrema == 'maximum':
      direc = -1
    elif extrema == 'minimum':
      direc = 1
    else:
      raise ValueError('No extrema was correctly identified. For maximum, type "maiximum". For minimization, type "minimum". ')
    return fn(scipy.optimize.fmin(lambda x: direc*fn(x), i_guess))

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


def get_pressure_bc_from_velocity(v: GridVariableVector) -> BoundaryConditions:
  """Returns pressure boundary conditions for the specified velocity."""
  # assumes that if the boundary is not periodic, pressure BC is zero flux.
  velocity_bc_types = consistent_boundary_conditions(*v)
  pressure_bc_types = []
  bc_value = ((0.0,0.0),(0.0,0.0))
  Bc_f = v[0].bc.boundary_fn
  for velocity_bc_type in velocity_bc_types:
    if velocity_bc_type == 'periodic':
      pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    else:
      pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
  return ConstantBoundaryConditions(values=bc_value,time_stamp=2.0,types=pressure_bc_types,boundary_fn=Bc_f) 



def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable,
    flux_direction: int) -> BoundaryConditions:
  """Returns advection flux boundary conditions for the specified velocity.
  Infers advection flux boundary condition in flux direction
  from scalar c and velocity u in direction flux_direction.
  If the boundaries are periodic, flux is periodic.
  In nonperiodic case, flux boundary parallel to flux direction is
  homogeneous dirichlet.
  In nonperiodic case if flux direction is normal to the wall, the
  function checks that the boundary needed is nonporous and returns the
  homogeneous bc. Otherwise throws an error.
  Args:
    u: velocity component in flux_direction.
    c: scalar to advect.
    flux_direction: direction of velocity.
  Returns:
    BoundaryCondition instance for advection flux of c in flux_direction.
  """
  # only no penetration and periodic boundaries are supported.
  flux_bc_types = []
  if not isinstance(u.bc, ConstantBoundaryConditions):
    raise NotImplementedError(
        f'Flux boundary condition is not implemented for {u.bc, c.bc}')
  for axis in range(c.grid.ndim):
    if u.bc.types[axis][0] == 'periodic':
      flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    elif flux_direction != axis:
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
    elif (u.bc.types[axis][0] == BCType.DIRICHLET and
          u.bc.types[axis][1] == BCType.DIRICHLET and
          u.bc.bc_values[axis][0] == 0.0 and u.bc.bc_values[axis][1] == 0.0):
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
    else:
      raise NotImplementedError(
          f'Flux boundary condition is not implemented for {u.bc, c.bc}')
  return HomogeneousBoundaryConditions(flux_bc_types)


def new_periodic_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],    
    bc_fn: Callable[...,Optional[float]],

) -> ConstantBoundaryConditions:
  """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.

  Args:
    ndim: spatial dimension.
    bc_fn: function describing the time dependent boundary condition
    bc_vals: A tuple of lower and upper boundary values for each dimension.
      If None, returns Homogeneous BC. For periodic dimensions the lower, upper
      boundary values should be (None, None).

  Returns:
    BoundaryCondition instance.
  """
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.PERIODIC, BCType.PERIODIC))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

  
    

  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)
