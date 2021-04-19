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

"""Grid classes that contain discretization information and boundary conditions.
"""

import numbers
import operator
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import dataclasses
import jax
from jax import lax
from jax import tree_util
from jax.lib import xla_bridge
import jax.numpy as jnp
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


@dataclasses.dataclass
class AlignedArray(np.lib.mixins.NDArrayOperatorsMixin):
  """Data with an aligned offset on a grid.

  Offset values in the range [0, 1] fall within a single grid cell.

  Examples:
    offset=(0, 0) means that each point is at the bottom-left corner.
    offset=(0.5, 0.5) is at the grid center.
    offset=(1, 0.5) is centered on the right-side edge.
  """
  # Don't (yet) enforce any explicit consistency requirements between data.ndim
  # and len(offset), e.g., so we can feel to add extra time/batch/channel
  # dimensions. But in most cases they should probably match.
  data: Array
  offset: Tuple[float, ...]

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def shape(self):
    return self.data.shape

  @property
  def ndim(self):
    return self.data.ndim

  _HANDLED_TYPES = (numbers.Number, np.ndarray, jnp.DeviceArray,
                    jax.ShapedArray, jax.core.Tracer)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """Define arithmetic on AlignedArrays using NumPy's mixin."""
    for x in inputs:
      if not isinstance(x, self._HANDLED_TYPES + (AlignedArray,)):
        return NotImplemented
    if method != '__call__':
      return NotImplemented
    try:
      # get the corresponding jax.np function to the NumPy ufunc
      func = getattr(jnp, ufunc.__name__)
    except AttributeError:
      return NotImplemented
    arrays = [x.data if isinstance(x, AlignedArray) else x for x in inputs]
    result = func(*arrays)
    offset = aligned_offset(*[x for x in inputs if isinstance(x, AlignedArray)])
    if isinstance(result, tuple):
      return tuple(AlignedArray(r, offset) for r in result)
    else:
      return AlignedArray(result, offset)

  def __getitem__(self, key):
    return applied(operator.getitem)(self, key)


def applied(func):
  """Convert an unaligned function into one defined on aligned arrays."""
  def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
    offset = aligned_offset(*[arg for arg in args + tuple(kwargs.values())
                              if isinstance(arg, AlignedArray)])
    raw_args = [
        arg.data if isinstance(arg, AlignedArray) else arg for arg in args
    ]
    raw_kwargs = {
        k: v.data if isinstance(v, AlignedArray) else v
        for k, v in kwargs.items()
    }
    data = func(*raw_args, **raw_kwargs)
    return AlignedArray(data, offset)
  return wrapper


tree_util.register_pytree_node(
    AlignedArray,
    lambda aligned: ((aligned.data,), aligned.offset),
    lambda offset, data: AlignedArray(data[0], offset),
)


class AlignmentError(Exception):
  """Raised for cases of inconsistent alignement."""


def aligned_offset(*arrays: AlignedArray) -> Tuple[float, ...]:
  """Returns the single matching offset, or raises AlignmentError."""
  offsets = {array.offset for array in arrays}
  if len(offsets) != 1:
    raise AlignmentError(f'arrays do not have a unique offset: {offsets}')
  offset, = offsets
  return offset


def averaged_offset(*arrays: AlignedArray) -> Tuple[float, ...]:
  """Returns the averaged offset of the given arrays."""
  offset = np.mean([array.offset for array in arrays], axis=0)
  return tuple(offset.tolist())


def control_volume_offsets(c: AlignedArray) -> Tuple[Tuple[float, ...], ...]:
  """Returns offsets for the faces of the control volume centered at `c`."""
  return tuple(tuple(o + .5 if i == j else o for i, o in enumerate(c.offset))
               for j in range(len(c.offset)))


class Tensor(np.ndarray):
  """A numpy array of AlignedArrays, representing a physical tensor field.

  Packing tensor coordinates into a numpy array of dtype object is useful
  because pointwise matrix operations like trace, transpose, and matrix
  multiplications of physical tensor quantities is meaningful.

  Example usage:
    grad = fd.gradient_tensor(uv, grid)              # a rank 2 Tensor
    strain_rate = (grad + grad.T) / 2.
    nu_smag = np.sqrt(np.trace(strain_rate.dot(strain_rate)))
    nu_smag = Tensor(nu_smag)                        # a rank 0 Tensor
    subgrid_stress = -2 * nu_smag * strain_rate      # a rank 2 Tensor
  """

  def __new__(cls, arrays):
    return np.asarray(arrays).view(cls)

jax.tree_util.register_pytree_node(
    Tensor,
    lambda tensor: (tensor.ravel().tolist(), tensor.shape),
    lambda shape, arrays: Tensor(np.asarray(arrays).reshape(shape)),
)


PERIODIC = 'periodic'
DIRICHLET = 'dirichlet'
VALID_BOUNDARIES = {PERIODIC, DIRICHLET}


@dataclasses.dataclass(init=False, frozen=True)
class Grid:
  """Describes the size, shape and boundary conditions for an Arakawa C-Grid.

  See https://en.wikipedia.org/wiki/Arakawa_grids. Instances of this class are
  also responsible for boundary conditions, such as periodic boundaries.

  This class describes domains that can be written as an outer-product of 1D
  grids. Along each dimension `i`:
  - `shape[i]` gives the whole number of grid cells on a single device.
  - `step[i]` is the width of each grid cell.
  - `(lower, upper) = domain[i]` gives the locations of lower and upper
    boundaries. The identity `upper - lower = step[i] * shape[i]` is enforced.
  - `boundaries[i]` gives the boundary conditions in this direction, either
    periodic or dirichlet.
  - `device_layout[i]` gives the integer number of devices this dimension is
    parallelized across.

  NOTE: only periodic boundary conditions work correctly in the rest of the
  JAX-CFD code.
  """
  shape: Tuple[int, ...]
  step: Tuple[float, ...]
  domain: Tuple[Tuple[float, float], ...]
  boundaries: Tuple[str, ...]
  device_layout: Optional[Tuple[int, ...]]

  def __init__(
      self,
      shape: Sequence[int],
      step: Optional[Union[float, Sequence[float]]] = None,
      domain: Optional[Sequence[Tuple[float, float]]] = None,
      boundaries: Union[str, Sequence[str]] = 'periodic',
      device_layout: Optional[Sequence[int]] = None,
  ):
    """Construct a grid object."""
    shape = tuple(operator.index(s) for s in shape)
    object.__setattr__(self, 'shape', shape)

    if step is not None and domain is not None:
      raise TypeError('cannot provide both step and domain')
    elif domain is not None:
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

    step = tuple((upper - lower) / size
                 for (lower, upper), size in zip(domain, shape))
    object.__setattr__(self, 'step', step)

    if isinstance(boundaries, str):
      boundaries = (boundaries,) * self.ndim
    invalid_boundaries = [boundary for boundary in boundaries
                          if boundary not in {PERIODIC, DIRICHLET}]
    if invalid_boundaries:
      raise ValueError(f'invalid boundaries: {invalid_boundaries}')
    object.__setattr__(self, 'boundaries', tuple(boundaries))

    if device_layout is not None:
      device_layout = tuple(operator.index(s) for s in device_layout)
      if len(device_layout) != self.ndim:
        raise ValueError('length of device_layout does not match ndim: '
                         f'len({device_layout}) != {self.ndim}')
      device_count = xla_bridge.device_count()
      if np.prod(device_layout) != device_count:
        raise ValueError(f'device_layout={device_layout} does not match '
                         f'device_count={device_count}')
      if any(size % num_devices
             for size, num_devices in zip(self.shape, device_layout)):
        raise ValueError(f'device_layout={device_layout} does not divide '
                         f'shape={self.shape}')
    object.__setattr__(self, 'device_layout', device_layout)

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

  def stagger(self, v: Tuple[Array, ...]) -> Tuple[AlignedArray, ...]:
    """Places the velocity components of `v` on the `Grid`'s cell faces."""
    offsets = self.cell_faces
    return tuple(AlignedArray(u, o) for u, o in zip(v, offsets))

  def center(self, v: PyTree) -> PyTree:
    """Places all arrays in the pytree `v` at the `Grid`'s cell center."""
    offset = self.cell_center
    return jax.tree_map(lambda u: AlignedArray(u, offset), v)

  def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """Returns a tuple of arrays containing the grid points along each axis.

    Args:
      offset: an optional sequence of length `ndim`. The grid will be shifted
        by `offset * self.step`.

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
                 for (lower, _), offset_i, length, step
                 in zip(self.domain, offset, self.shape, self.step))

  def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """Returns an tuple of arrays containing positions in each grid cell.

    Args:
      offset: an optional sequence of length `ndim`. The grid will be shifted
        by `offset * self.step`.

    Returns:
      An tuple of `self.ndim` arrays, each of shape `self.shape`. In 3
      dimensions, entry `self.mesh[n][i, j, k]` is the location of point
      `i, j, k` in dimension `n`.
    """
    axes = self.axes(offset)
    return tuple(jnp.meshgrid(*axes, indexing='ij'))

  def shift(
      self,
      u: AlignedArray,
      offset: int,
      axis: int,
      pad_kwargs: Optional[Mapping[str, Any]] = None,
  ) -> AlignedArray:
    """Shift an AlignedArray by `offset`.

    Args:
      u: an `AlignedArray` object.
      offset: positive or negative integer offset to shift.
      axis: axis to shift along.
      pad_kwargs: optional keyword arguments passed into `jax.np.pad`. By
        default, uses appropriate padding for the Grid's boundary conditions
        (`mode='wrap'` for periodic, `mode='constant` with a fill value of zero
        for dirichlet).

    Returns:
      A copy of `u`, shifted by `offset`. The returned `AlignedArray` has offset
      `u.offset + offset`.
    """
    padding = (-offset, 0) if offset < 0 else (0, offset)
    padded = self.pad(u, padding, axis, pad_kwargs)
    trimmed = self.trim(padded, padding[::-1], axis)
    return trimmed

  def pad(
      self,
      u: AlignedArray,
      padding: Tuple[int, int],
      axis: int,
      pad_kwargs: Optional[Mapping[str, Any]] = None,
  ) -> AlignedArray:
    """Pad an AlignedArray by `padding`.

    Args:
      u: an `AlignedArray` object.
      padding: left and right padding along this axis.
      axis: axis to pad along.
      pad_kwargs: optional keyword arguments passed into `jax.np.pad`. By
        default, uses appropriate padding for the Grid's boundary conditions
        (`mode='wrap'` for periodic, `mode='constant'` with a fill value of zero
        for dirichlet).

    Returns:
      Padded array, elongated along the indicated axis.
    """
    if pad_kwargs is None:
      if self.boundaries[axis] == PERIODIC:
        pad_kwargs = dict(mode='wrap')
      else:
        assert self.boundaries[axis] == DIRICHLET
        pad_kwargs = dict(mode='constant')
    elif self.device_layout is not None:
      raise ValueError('only default pad_kwargs supported if device_layout set')

    offset = list(u.offset)
    offset[axis] -= padding[0]

    if self.device_layout is not None:
      left = self.device_shift(u.data, shift=+1, axis=axis)
      right = self.device_shift(u.data, shift=-1, axis=axis)
      left_start = u.data.shape[axis] - padding[0]
      data = jnp.concatenate([
          lax.dynamic_slice_in_dim(left, left_start, padding[0], axis),
          u.data,
          lax.dynamic_slice_in_dim(right, 0, padding[1], axis),
      ], axis=axis)
    else:
      full_padding = [(0, 0)] * u.data.ndim
      full_padding[axis] = padding
      data = jnp.pad(u.data, full_padding, **pad_kwargs)
    return AlignedArray(data, tuple(offset))

  def trim(
      self, u: AlignedArray, padding: Tuple[int, int], axis: int,
  ) -> AlignedArray:
    """Trim padding from an AlignedArray.

    Args:
      u: an `AlignedArray` object.
      padding: left and right padding along this axis.
      axis: axis to trim along.

    Returns:
      Trimmed array, shrunk along the indicated axis.
    """
    slice_size = u.data.shape[axis] - sum(padding)
    data = lax.dynamic_slice_in_dim(u.data, padding[0], slice_size, axis)
    offset = list(u.offset)
    offset[axis] += padding[0]
    return AlignedArray(data, tuple(offset))

  def device_shift(self, array: Array, shift: int, axis: int) -> Array:
    """Shift an array between devices."""
    if self.device_layout is None:
      raise ValueError('must specify device_layout to use device_shift')
    # TODO(shoyer): extract IDs matching actual device topology, i.e., from
    # metadata in jax.devices().
    ids = np.arange(np.prod(self.device_layout)).reshape(self.device_layout)
    perm = _device_permutation(ids, shift, axis, self.boundaries[axis])
    permuted = lax.ppermute(array, 'space', perm)
    return permuted


def _device_permutation(
    ids: np.ndarray,
    shift: int,
    axis: int,
    boundary: str = 'periodic',
) -> List[Tuple[Any, Any]]:
  """Determine the appropriate device ID permutation for the given shift."""
  ids = np.asarray(ids)
  shifted = np.roll(ids, -shift, axis)

  if boundary == DIRICHLET:
    index = [slice(None)] * ids.ndim
    index[axis] = slice(None, -shift) if shift >= 0 else slice(-shift, None)
    ids = ids[tuple(index)]
    shifted = shifted[tuple(index)]
  elif boundary != PERIODIC:
    raise ValueError(f'invalid boundary: {boundary}')

  return list(zip(ids.ravel().tolist(), shifted.ravel().tolist()))


# Aliases for often used `grids.applied` functions.
where = applied(jnp.where)
