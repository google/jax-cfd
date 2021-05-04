"""Utilities for spatial tiling of periodic convolutions into batch dimensions.

``layout`` tuple indicates how the corresponding spatial dimensions are layed
out in space. In 2D:
- `(1, 1)` indicates no tiling.
- `(4, 2)` indicates 4 x-tiles and 2 y-tiles
- `(16, 8)` indicates 16 x-tiles and 8 y-tiles

Tiling is helpful for getting the highest performance convolutions on TPU. Per
the TPU performance guide [1], batch dimensions on TPUs are tiled to multiples
of 8 or 128. Thus the product of all elements in `layout` should typically be
either 8 or 128.

[1] https://cloud.google.com/tpu/docs/performance-guide.
"""
import functools
from typing import Callable, Sequence, Tuple

import einops
import jax
from jax import lax
import jax.numpy as jnp


Array = jnp.ndarray


def _prod(xs):
  # backport of math.prod() from Python 3.8+
  result = 1
  for x in xs:
    result *= x
  return result


def _verify_layout(array, layout):
  if array.ndim != len(layout) + 2 or array.shape[0] != _prod(layout):
    raise ValueError(
        f"array shape does not match layout: {array.shape} vs {layout}")


def _layout_to_dict(layout):
  return dict(zip(["bx", "by", "bz"], layout))


def _tile_roll(array, layout, shift, axis):
  """Roll along the "tiled" dimension."""
  _verify_layout(array, layout)
  sizes = _layout_to_dict(layout)
  if len(layout) == 1:
    array = jnp.roll(array, shift, axis=axis)
  elif len(layout) == 2:
    array = einops.rearrange(array, "(bx by) ... -> bx by ...", **sizes)
    array = jnp.roll(array, shift, axis=axis)
    array = einops.rearrange(array, "bx by ... -> (bx by) ...", **sizes)
  elif len(layout) == 3:
    array = einops.rearrange(array, "(bx by bz) ... -> bx by bz ...", **sizes)
    array = jnp.roll(array, shift, axis=axis)
    array = einops.rearrange(array, "bx by bz ... -> (bx by bz) ...", **sizes)
  else:
    raise NotImplementedError
  return array


def _halo_pad_1d(array, layout, axis, padding=(1, 1)):
  """Pad for halo-exchange along a single array dimension."""
  pad_left, pad_right = padding
  spatial_axis = axis + 1
  pieces = []

  if pad_left:
    # Note: importantly, dynamic_slice_in_dim raises an error for out of bounds
    # access, which catches the edge case where a single array is insufficient
    # padding.
    start = array.shape[spatial_axis] - pad_left
    input_right = lax.dynamic_slice_in_dim(array, start, pad_left, spatial_axis)
    output_left = _tile_roll(input_right, layout, shift=+1, axis=axis)
    pieces.append(output_left)

  pieces.append(array)

  if pad_right:
    start = 0
    input_left = lax.dynamic_slice_in_dim(array, start, pad_right, spatial_axis)
    output_right = _tile_roll(input_left, layout, shift=-1, axis=axis)
    pieces.append(output_right)

  return jnp.concatenate(pieces, axis=spatial_axis)


@functools.partial(jax.jit, static_argnums=(1, 2,))
def _halo_exchange_pad(array: Array, layout: Tuple[int, ...],
                       padding: Tuple[Tuple[int, int]]) -> Array:
  """Pad with halo-exchange in N-dimensions."""
  _verify_layout(array, layout)
  if len(layout) != len(padding):
    raise ValueError(f"invalid padding: {padding}")
  out = array
  for axis, pad in enumerate(padding):
    out = _halo_pad_1d(out, layout, axis, pad)
  return out


def halo_exchange_pad(
    array: Array,
    layout: Tuple[int, ...],
    padding: Sequence[Tuple[int, int]],
) -> Array:
  """Pad with halo-exchange in N-dimensions."""
  return _halo_exchange_pad(
      array, layout,
      tuple(map(tuple, padding)))


@functools.partial(jax.jit, static_argnums=(1,))
def space_to_batch(array: Array, layout: Tuple[int, ...]) -> Array:
  """Rearrange from space to batch dimensions."""
  sizes = _layout_to_dict(layout)
  if len(layout) == 1:
    path = "(bx x) c -> (bx) x c"
  elif len(layout) == 2:
    path = "(bx x) (by y) c -> (bx by) x y c"
  elif len(layout) == 3:
    path = "(bx x) (by y) (bz z) c -> (bx by bz) x y z c"
  else:
    raise NotImplementedError
  return einops.rearrange(array, path, **sizes)


@functools.partial(jax.jit, static_argnums=(1,))
def batch_to_space(array: Array, layout: Tuple[int, ...]) -> Array:
  """Rearrange from batch to space dimensions."""
  sizes = _layout_to_dict(layout)
  if len(layout) == 1:
    path = "(bx) x c -> (bx x) c"
  elif len(layout) == 2:
    path = "(bx by) x y c -> (bx x) (by y) c"
  elif len(layout) == 3:
    path = "(bx by bz) x y z c-> (bx x) (by y) (bz z) c"
  else:
    raise NotImplementedError
  return einops.rearrange(array, path, **sizes)


def apply_convolution(
    conv: Callable[[Array], Array],
    inputs: Array,
    layout: Tuple[int, ...],
    padding: Sequence[Tuple[int, ...]],
) -> Array:
  """Apply a valid convolution with tiling and periodic boundary conditions.

  Args:
    conv: function that calculates a convolution with valid boundary conditions
      when applied to an array of shape [batch, [spatial dims], channel].
    inputs: array of shape [[spatial dims], channel].
    layout: tiling layout for implementing the operation.
    padding: amount of periodic padding to add before and after each spatial
      dimension.

  Returns:
    Convolved array.
  """
  if layout is None:
    # TODO(shoyer): replace this with some sensible heuristic
    layout = (1,) * len(padding)
  tiled = space_to_batch(inputs, layout)
  padded = halo_exchange_pad(tiled, layout, padding)
  convolved = conv(padded)
  output = batch_to_space(convolved, layout)
  return output
