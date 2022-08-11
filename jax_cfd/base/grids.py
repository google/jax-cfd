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
"""Grid classes that contain discretization information and boundary conditions."""
from __future__ import annotations

import dataclasses
import numbers
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

# TODO(jamieas): consider moving common types to a separate module.
# TODO(shoyer): consider adding jnp.ndarray?
Array = Union[np.ndarray, jnp.DeviceArray]
IntOrSequence = Union[int, Sequence[int]]

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any


@register_pytree_node_class
@dataclasses.dataclass
class GridArray(np.lib.mixins.NDArrayOperatorsMixin):
  """Data with an alignment offset and an associated grid.

  Offset values in the range [0, 1] fall within a single grid cell.

  Examples:
    offset=(0, 0) means that each point is at the bottom-left corner.
    offset=(0.5, 0.5) is at the grid center.
    offset=(1, 0.5) is centered on the right-side edge.

  Attributes:
    data: array values.
    offset: alignment location of the data with respect to the grid.
    grid: the Grid associated with the array data.
    dtype: type of the array data.
    shape: lengths of the array dimensions.
  """
  # Don't (yet) enforce any explicit consistency requirements between data.ndim
  # and len(offset), e.g., so we can feel to add extra time/batch/channel
  # dimensions. But in most cases they should probably match.
  # Also don't enforce explicit consistency between data.shape and grid.shape,
  # but similarly they should probably match.
  data: Array
  offset: Tuple[float, ...]
  grid: Grid

  def tree_flatten(self):
    """Returns flattening recipe for GridArray JAX pytree."""
    children = (self.data,)
    aux_data = (self.offset, self.grid)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Returns unflattening recipe for GridArray JAX pytree."""
    return cls(*children, *aux_data)

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.data.shape

  _HANDLED_TYPES = (numbers.Number, np.ndarray, jnp.DeviceArray,
                    jax.ShapedArray, jax.core.Tracer)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """Define arithmetic on GridArrays using NumPy's mixin."""
    for x in inputs:
      if not isinstance(x, self._HANDLED_TYPES + (GridArray,)):
        return NotImplemented
    if method != '__call__':
      return NotImplemented
    try:
      # get the corresponding jax.np function to the NumPy ufunc
      func = getattr(jnp, ufunc.__name__)
    except AttributeError:
      return NotImplemented
    arrays = [x.data if isinstance(x, GridArray) else x for x in inputs]
    result = func(*arrays)
    offset = consistent_offset(*[x for x in inputs if isinstance(x, GridArray)])
    grid = consistent_grid(*[x for x in inputs if isinstance(x, GridArray)])
    if isinstance(result, tuple):
      return tuple(GridArray(r, offset, grid) for r in result)
    else:
      return GridArray(result, offset, grid)


GridArrayVector = Tuple[GridArray, ...]


class GridArrayTensor(np.ndarray):
  """A numpy array of GridArrays, representing a physical tensor field.

  Packing tensor coordinates into a numpy array of dtype object is useful
  because pointwise matrix operations like trace, transpose, and matrix
  multiplications of physical tensor quantities is meaningful.

  Example usage:
    grad = fd.gradient_tensor(uv)                    # a rank 2 Tensor
    strain_rate = (grad + grad.T) / 2.
    nu_smag = np.sqrt(np.trace(strain_rate.dot(strain_rate)))
    nu_smag = Tensor(nu_smag)                        # a rank 0 Tensor
    subgrid_stress = -2 * nu_smag * strain_rate      # a rank 2 Tensor
  """

  def __new__(cls, arrays):
    return np.asarray(arrays).view(cls)


jax.tree_util.register_pytree_node(
    GridArrayTensor,
    lambda tensor: (tensor.ravel().tolist(), tensor.shape),
    lambda shape, arrays: GridArrayTensor(np.asarray(arrays).reshape(shape)),
)


@dataclasses.dataclass(init=False, frozen=True)
class BoundaryConditions:
  """Base class for boundary conditions on a PDE variable.

  Attributes:
    types: `types[i]` is a tuple specifying the lower and upper BC types for
      dimension `i`.
  """
  types: Tuple[Tuple[str, str], ...]

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
    raise NotImplementedError(
        'shift() not implemented in BoundaryConditions base class.')

  def values(self, axis: int, grid: Grid, offset: Optional[Tuple[float, ...]],
             time: Optional[float]) -> Tuple[Optional[Array], Optional[Array]]:
    """Returns Arrays specifying boundary values on the grid along axis.

    Args:
      axis: axis along which to return boundary values.
      grid: a `Grid` object on which to evaluate boundary conditions.
      offset: a Tuple of offsets that specifies (along with grid) where to
        evaluate boundary conditions in space.
      time: a float used as an input to boundary function.

    Returns:
      A tuple of arrays of grid.ndim - 1 dimensions that specify values on the
      boundary. In case of periodic boundaries, returns a tuple(None,None).
    """
    raise NotImplementedError(
        'values() not implemented in BoundaryConditions base class.')

  def pad(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """Returns Arrays padded according to boundary condition.

    Args:
      u: a `GridArray` object.
      width: number of elements to pad along axis. Use negative value for lower
        boundary or positive value for upper boundary.
      axis: axis to pad along.

    Returns:
      A GridArray that is elongated along axis with padded values.
    """
    raise NotImplementedError(
        'pad() not implemented in BoundaryConditions base class.')

  def trim_boundary(self, u: GridArray) -> GridArray:
    """Returns GridArray without the grid points on the boundary.

    Some grid points of GridArray might coincide with boundary. This trims those
    values.

    Args:
      u: a `GridArray` object.

    Returns:
      A GridArray shrunk along certain dimensions.
    """
    raise NotImplementedError(
        'trim_boundary() not implemented in BoundaryConditions base class.')

  def pad_and_impose_bc(
      self,
      u: GridArray,
      offset_to_pad_to: Optional[Tuple[float, ...]] = None) -> GridVariable:
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
    raise NotImplementedError(
        'pad_and_impose_bc() not implemented in BoundaryConditions base class.')

  def impose_bc(self, u: GridArray) -> GridVariable:
    """Returns GridVariable with correct boundary condition.

    Some grid points of GridArray might coincide with boundary. This ensures
    that the GridVariable.array agrees with GridVariable.bc.
    Args:
      u: a `GridArray` object.

    Returns:
      A GridVariable that has correct boundary.
    """
    raise NotImplementedError(
        'impose_bc() not implemented in BoundaryConditions base class.')


@register_pytree_node_class
@dataclasses.dataclass
class GridVariable:
  """Associates a GridArray with BoundaryConditions.

  Performing pad and shift operations, e.g. for finite difference calculations,
  requires boundary condition (BC) information. Since different variables in a
  PDE system can have different BCs, this class associates a specific variable's
  data with its BCs.

  Array operations on GridVariables act like array operations on the
  encapsulated GridArray.

  Attributes:
    array: GridArray with the array data, offset, and associated grid.
    bc: boundary conditions for this variable.
    grid: the Grid associated with the array data.
    dtype: type of the array data.
    shape: lengths of the array dimensions.
    data: array values.
    offset: alignment location of the data with respect to the grid.
    grid: the Grid associated with the array data.
  """
  array: GridArray
  bc: BoundaryConditions

  def __post_init__(self):
    if not isinstance(self.array, GridArray):  # frequently missed by pytype
      raise ValueError(
          f'Expected array type to be GridArray, got {type(self.array)}')
    if len(self.bc.types) != self.grid.ndim:
      raise ValueError(
          'Incompatible dimension between grid and bc, grid dimension = '
          f'{self.grid.ndim}, bc dimension = {len(self.bc.types)}')

  def tree_flatten(self):
    """Returns flattening recipe for GridVariable JAX pytree."""
    children = (self.array,)
    aux_data = (self.bc,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Returns unflattening recipe for GridVariable JAX pytree."""
    return cls(*children, *aux_data)

  @property
  def dtype(self):
    return self.array.dtype

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.array.shape

  @property
  def data(self) -> Array:
    return self.array.data

  @property
  def offset(self) -> Tuple[float, ...]:
    return self.array.offset

  @property
  def grid(self) -> Grid:
    return self.array.grid

  def shift(
      self,
      offset: int,
      axis: int,
  ) -> GridArray:
    """Shift this GridVariable by `offset`.

    Args:
      offset: positive or negative integer offset to shift.
      axis: axis to shift along.

    Returns:
      A copy of the encapsulated GridArray, shifted by `offset`. The returned
      GridArray has offset `u.offset + offset`.
    """
    return self.bc.shift(self.array, offset, axis)

  def _interior_grid(self) -> Grid:
    """Returns only the interior grid points."""
    grid = self.array.grid
    domain = list(grid.domain)
    shape = list(grid.shape)
    for axis in range(self.grid.ndim):
      # nothing happens in periodic case
      if self.bc.types[axis][1] == 'periodic':
        continue
      # nothing happens if the offset is not 0.0 or 1.0
      # this will automatically set the grid to interior.
      if np.isclose(self.array.offset[axis], 1.0):
        shape[axis] -= 1
        domain[axis] = (domain[axis][0], domain[axis][1] - grid.step[axis])
      elif np.isclose(self.array.offset[axis], 0.0):
        shape[axis] -= 1
        domain[axis] = (domain[axis][0] + grid.step[axis], domain[axis][1])
    return Grid(shape, domain=tuple(domain))

  def trim_boundary(self) -> GridArray:
    """Returns a GridArray associated only with interior points.

     Interior is defined as the following:
       for d in range(u.grid.ndim):
        points = u.grid.axes(offset=u.offset[d])
        interior_points =
          all points where grid.domain[d][0] < points < grid.domain[d][1]

    The exception is when the boundary conditions are periodic,
    in which case all points are included in the interior.

    In case of dirichlet with edge offset, the grid and array size is reduced,
    since one scalar lies exactly on the boundary. In all other cases,
    self.grid and self.array are returned.
    """
    return self.bc.trim_boundary(self.array)

  def impose_bc(self) -> GridVariable:
    """Returns the GridVariable with edge BC enforced, if applicable.

    For GridVariables having nonperiodic BC and offset 0 or 1, there are values
    in the array data that are dependent on the boundary condition.
    impose_bc() changes these boundary values to match the prescribed BC.
    """
    return self.bc.impose_bc(self.array)


GridVariableVector = Tuple[GridVariable, ...]


def applied(func):
  """Convert an array function into one defined on GridArrays.

  Since `func` can only act on `data` attribute of GridArray, it implicitly
  enforces that `func` cannot modify the other attributes such as offset.

  Args:
    func: function being wrapped.

  Returns:
    A wrapped version of `func` that takes GridArray instead of Array args.
  """

  def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
    for arg in args + tuple(kwargs.values()):
      if isinstance(arg, GridVariable):
        raise ValueError('grids.applied() cannot be used with GridVariable')

    offset = consistent_offset(*[
        arg for arg in args + tuple(kwargs.values())
        if isinstance(arg, GridArray)
    ])
    grid = consistent_grid(*[
        arg for arg in args + tuple(kwargs.values())
        if isinstance(arg, GridArray)
    ])
    raw_args = [arg.data if isinstance(arg, GridArray) else arg for arg in args]
    raw_kwargs = {
        k: v.data if isinstance(v, GridArray) else v for k, v in kwargs.items()
    }
    data = func(*raw_args, **raw_kwargs)
    return GridArray(data, offset, grid)

  return wrapper


# Aliases for often used `grids.applied` functions.
where = applied(jnp.where)


def averaged_offset(
    *arrays: Union[GridArray, GridVariable]) -> Tuple[float, ...]:
  """Returns the averaged offset of the given arrays."""
  offset = np.mean([array.offset for array in arrays], axis=0)
  return tuple(offset.tolist())


def control_volume_offsets(
    c: Union[GridArray, GridVariable]) -> Tuple[Tuple[float, ...], ...]:
  """Returns offsets for the faces of the control volume centered at `c`."""
  return tuple(
      tuple(o + .5 if i == j else o
            for i, o in enumerate(c.offset))
      for j in range(len(c.offset)))


class InconsistentOffsetError(Exception):
  """Raised for cases of inconsistent offset in GridArrays."""


def consistent_offset(
    *arrays: Union[GridArray, GridVariable]) -> Tuple[float, ...]:
  """Returns the unique offset, or raises InconsistentOffsetError."""
  offsets = {array.offset for array in arrays}
  if len(offsets) != 1:
    raise InconsistentOffsetError(
        f'arrays do not have a unique offset: {offsets}')
  offset, = offsets
  return offset


class InconsistentGridError(Exception):
  """Raised for cases of inconsistent grids between GridArrays."""


def consistent_grid(*arrays: Union[GridArray, GridVariable]) -> Grid:
  """Returns the unique grid, or raises InconsistentGridError."""
  grids = {array.grid for array in arrays}
  if len(grids) != 1:
    raise InconsistentGridError(f'arrays do not have a unique grid: {grids}')
  grid, = grids
  return grid


class InconsistentBoundaryConditionsError(Exception):
  """Raised for cases of inconsistent bc between GridVariables."""


def unique_boundary_conditions(*arrays: GridVariable) -> BoundaryConditions:
  """Returns the unique BCs, or raises InconsistentBoundaryConditionsError."""
  bcs = {array.bc for array in arrays}
  if len(bcs) != 1:
    raise InconsistentBoundaryConditionsError(
        f'arrays do not have a unique bc: {bcs}')
  bc, = bcs
  return bc


@dataclasses.dataclass(init=False, frozen=True)
class Grid:
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
    shape = tuple(operator.index(s) for s in shape)
    object.__setattr__(self, 'shape', shape)

    if step is not None and domain is not None:
      raise TypeError('cannot provide both step and domain')
    elif domain is not None:
      if isinstance(domain, (int, float)):
        domain = ((0, domain),) * len(shape)
      else:
        if len(domain) != self.ndim:
          raise ValueError('length of domain does not match ndim: '
                           f'{len(domain)} != {self.ndim}')
        for bounds in domain:
          if len(bounds) != 2:
            raise ValueError(
                f'domain is not sequence of pairs of numbers: {domain}')
      domain = tuple((float(lower), float(upper)) for lower, upper in domain)
    else:
      if step is None:
        step = 1
      if isinstance(step, numbers.Number):
        step = (step,) * self.ndim
      elif len(step) != self.ndim:
        raise ValueError('length of step does not match ndim: '
                         f'{len(step)} != {self.ndim}')
      domain = tuple(
          (0.0, float(step_ * size)) for step_, size in zip(step, shape))

    object.__setattr__(self, 'domain', domain)

    step = tuple(
        (upper - lower) / size for (lower, upper), size in zip(domain, shape))
    object.__setattr__(self, 'step', step)

  @property
  def ndim(self) -> int:
    """Returns the number of dimensions of this grid."""
    return len(self.shape)

  @property
  def cell_center(self) -> Tuple[float, ...]:
    """Offset at the center of each grid cell."""
    return self.ndim * (0.5,)

  @property
  def cell_faces(self) -> Tuple[Tuple[float, ...]]:
    """Returns the offsets at each of the 'forward' cell faces."""
    d = self.ndim
    offsets = (np.eye(d) + np.ones([d, d])) / 2.
    return tuple(tuple(float(o) for o in offset) for offset in offsets)

  def stagger(self, v: Tuple[Array, ...]) -> Tuple[GridArray, ...]:
    """Places the velocity components of `v` on the `Grid`'s cell faces."""
    offsets = self.cell_faces
    return tuple(GridArray(u, o, self) for u, o in zip(v, offsets))

  def center(self, v: PyTree) -> PyTree:
    """Places all arrays in the pytree `v` at the `Grid`'s cell center."""
    offset = self.cell_center
    return jax.tree_map(lambda u: GridArray(u, offset, self), v)

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
    return tuple(lower + (jnp.arange(length) + offset_i) * step
                 for (lower, _), offset_i, length, step in zip(
                     self.domain, offset, self.shape, self.step))

  def fft_axes(self) -> Tuple[Array, ...]:
    """Returns the ordinal frequencies corresponding to the axes.

    Transforms each axis into the *ordinal* frequencies for the Fast Fourier
    Transform (FFT). Multiply by `2 * jnp.pi` to get angular frequencies.

    Returns:
      A tuple of `self.ndim` arrays. The jth return value has shape
      `[self.shape[j]]`.
    """
    freq_axes = tuple(
        jnp.fft.fftfreq(n, d=s) for (n, s) in zip(self.shape, self.step))
    return freq_axes

  def rfft_axes(self) -> Tuple[Array, ...]:
    """Returns the ordinal frequencies corresponding to the axes.

    Transforms each axis into the *ordinal* frequencies for the Fast Fourier
    Transform (FFT). Most useful for doing computations for real-valued (not
    complex valued) signals.

    Multiply by `2 * jnp.pi` to get angular frequencies.

    Returns:
      A tuple of `self.ndim` arrays. The shape of each array matches the result
      of rfftfreqs. Specifically, rfft is applied to the last dimension
      resulting in an array of length `self.shape[-1] // 2`. Complex `fft` is
      applied to the other dimensions resulting in shapes of size
      `self.shape[j]`.
    """
    fft_axes = tuple(
        jnp.fft.fftfreq(n, d=s)
        for (n, s) in zip(self.shape[:-1], self.step[:-1]))
    rfft_axis = (jnp.fft.rfftfreq(self.shape[-1], d=self.step[-1]),)
    return fft_axes + rfft_axis

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
    axes = self.axes(offset)
    return tuple(jnp.meshgrid(*axes, indexing='ij'))

  def rfft_mesh(self) -> Tuple[Array, ...]:
    """Returns a tuple of arrays containing positions in rfft space."""
    rfft_axes = self.rfft_axes()
    return tuple(jnp.meshgrid(*rfft_axes, indexing='ij'))

  def eval_on_mesh(self,
                   fn: Callable[..., Array],
                   offset: Optional[Sequence[float]] = None) -> GridArray:
    """Evaluates the function on the grid mesh with the specified offset.

    Args:
      fn: A function that accepts the mesh arrays and returns an array.
      offset: an optional sequence of length `ndim`.  If not specified, uses the
        offset for the cell center.

    Returns:
      fn(x, y, ...) evaluated on the mesh, as a GridArray with specified offset.
    """
    if offset is None:
      offset = self.cell_center
    return GridArray(fn(*self.mesh(offset)), offset, self)


def domain_interior_masks(grid: Grid):
  """Returns cell face arrays with 1 on the interior, 0 on the boundary."""
  masks = []
  for offset in grid.cell_faces:
    mesh = grid.mesh(offset)
    mask = 1
    for i, x in enumerate(mesh):
      lower = (np.invert(np.isclose(x, grid.domain[i][0]))).astype('int')
      upper = (np.invert(np.isclose(x, grid.domain[i][1]))).astype('int')
      mask = mask * upper * lower
    masks.append(mask)
  return tuple(masks)
