import dataclasses
import numbers
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
from jax_ib.base import grids


Array = Union[np.ndarray, jax.Array]
IntOrSequence = Union[int, Sequence[int]]

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any
@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
  """Describes the size and shape for an Arakawa C-Grid.

  See https://en.wikipedia.org/wiki/Arakawa_grids.

  This class describes domains that can be written as an outer-product of 1D
  grids. Along each dimension `i`:
  - `shape[i]` gives the whole number of grid cells on a single device.
  - `step[i]` is the width of each grid cell.
  - `(lower, upper) = domain[i]` gives the locations of lower and upper
    boundaries. The identity `upper - lower = step[i] * shape[i]` is enforced.
  """
  shape: Tuple[int, ...]
  step: Tuple[float, ...]
  domain: Tuple[Tuple[float, float], ...]

  def __init__(
      self,
      shape: Sequence[int],
      step: Optional[Union[float, Sequence[float]]] = None,
      domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
  ):
    """Construct a grid object."""
    shape = shape
    object.__setattr__(self, 'shape', shape)

 

    object.__setattr__(self, 'domain', domain)

    step = (domain[1] - domain[0]) / (shape-1) 
    object.__setattr__(self, 'step', step)

  @property
  def ndim(self) -> int:
    """Returns the number of dimensions of this grid."""
    return 1

  @property
  def cell_center(self) -> Tuple[float, ...]:
    """Offset at the center of each grid cell."""
    return self.ndim * (0.5,)



  def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """Returns a tuple of arrays containing the grid points along each axis.

    Args:
      offset: an optional sequence of length `ndim`. The grid will be shifted by
        `offset * self.step`.

    Returns:
      An tuple of `self.ndim` arrays. The jth return value has shape
      `[self.shape[j]]`.
    """
    if offset is None:
      offset = self.cell_center
    if len(offset) != self.ndim:
      raise ValueError(f'unexpected offset length: {len(offset)} vs '
                       f'{self.ndim}')

    return self.domain[0] + jnp.arange(self.shape)*self.step



  def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """Returns an tuple of arrays containing positions in each grid cell.

    Args:
      offset: an optional sequence of length `ndim`. The grid will be shifted by
        `offset * self.step`.

    Returns:
      An tuple of `self.ndim` arrays, each of shape `self.shape`. In 3
      dimensions, entry `self.mesh[n][i, j, k]` is the location of point
      `i, j, k` in dimension `n`.
    """
    
    return self.axes(offset)




@register_pytree_node_class
@dataclasses.dataclass
class particle: 
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    displacement_param: Sequence[Any]
    rotation_param: Sequence[Any]
    Grid: Grid1d
    shape: Callable
    Displacement_EQ: Callable
    Rotation_EQ: Callable
    
    

    
    
    def tree_flatten(self):
      """Returns flattening recipe for GridVariable JAX pytree."""
      children = (self.particle_center,self.geometry_param,self.displacement_param,self.rotation_param,)
 
      aux_data = (self.Grid,self.shape,self.Displacement_EQ,self.Rotation_EQ,)
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Returns unflattening recipe for GridVariable JAX pytree."""
       return cls(*children,*aux_data) 

    def generate_grid(self):
        
        return self.Grid.mesh()
       
    def calc_Rtheta(self):
      return self.shape(self.geometry_param,self.Grid) 




@register_pytree_node_class
@dataclasses.dataclass
class All_Variables: 
    particles: Sequence[particle,]
    velocity: grids.GridVariableVector
    pressure: grids.GridVariable
    Drag:Sequence[Any]
    Step_count:int
    MD_var:Any
    def tree_flatten(self):
      """Returns flattening recipe for GridVariable JAX pytree."""
      children = (self.particles,self.velocity,self.pressure,self.Drag,self.Step_count,self.MD_var,)
 
      aux_data = None
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Returns unflattening recipe for GridVariable JAX pytree."""
       return cls(*children) 
    

        
    
@register_pytree_node_class
@dataclasses.dataclass
class particle_lista: # SEQUENCE OF VARIABLES MATTER !
    particles: Sequence[particle,]

    
    def generate_grid(self):
        
        return np.stack([grid.mesh() for grid in self.Grid])
       
    def calc_Rtheta(self):
      return self.shape(self.geometry_param,self.Grid) 
    
    def tree_flatten(self):
      """Returns flattening recipe for GridVariable JAX pytree."""
      children = (*self.particles,)
      aux_data = None
      return children,aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Returns unflattening recipe for GridVariable JAX pytree."""
       return cls(*children)
    
    

    
    
