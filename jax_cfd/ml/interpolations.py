"""Interpolation modules."""

import collections
import functools
from typing import Callable, Tuple

import gin
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.ml import layers
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers
import numpy as np


AlignedArray = grids.AlignedArray
AlignedField = Tuple[AlignedArray, ...]
Grid = grids.Grid
InterpolationFn = interpolation.InterpolationFn
InterpolationModule = Callable[..., InterpolationFn]
InterpolationTransform = Callable[..., InterpolationFn]
FluxLimiter = interpolation.FluxLimiter


@gin.configurable
class FusedLearnedInterpolation:
  """Learned interpolator that computes interpolation coefficients in 1 pass.

  Interpolation function that has pre-computed interpolation
  coefficients for a given velocity field `v`. It uses a collection of
  `SpatialDerivativeFromLogits` modules and a single neural network that
  produces logits for all expected interpolations. Interpolations are keyed by
  `input_offset`, `target_offset` and an optional `tag`. The `tag` allows us to
  perform multiple interpolations between the same `offset` and `target_offset`
  with different weights.
  """

  def __init__(
      self,
      grid: grids.Grid,
      dt: float,
      physics_specs: physics_specifications.BasePhysicsSpecs,
      v,
      tags=(None,),
      stencil_size=4,
      tower_factory=towers.forward_tower_factory,
      name='fused_learned_interpolation',
      extract_patch_method='roll',
      fuse_constraints=False,
      fuse_patches=False,
      tile_layout=None,
  ):
    """Constructs object and performed necessary pre-computate."""
    del dt, physics_specs  # unused.

    stencil_sizes = (stencil_size,) * grid.ndim
    derivative_orders = (0,) * grid.ndim
    derivatives = collections.OrderedDict()

    for u in v:
      for target_offset in grids.control_volume_offsets(u):
        for tag in tags:
          key = (u.offset, target_offset, tag)
          derivatives[key] = layers.SpatialDerivativeFromLogits(
              stencil_sizes,
              u.offset,
              target_offset,
              derivative_orders=derivative_orders,
              steps=grid.step,
              extract_patch_method=extract_patch_method,
              tile_layout=tile_layout)

    output_sizes = [deriv.subspace_size for deriv in derivatives.values()]
    cnn_network = tower_factory(sum(output_sizes), grid.ndim, name=name)
    inputs = jnp.stack([u.data for u in v], axis=-1)
    all_logits = cnn_network(inputs)

    if fuse_constraints:
      self._interpolators = layers.fuse_spatial_derivative_layers(
          derivatives, all_logits, fuse_patches)
    else:
      split_logits = jnp.split(all_logits, np.cumsum(output_sizes), axis=-1)
      self._interpolators = {
          k: functools.partial(derivative, logits=logits)
          for (k, derivative), logits in zip(derivatives.items(), split_logits)
      }

  def __call__(self, c, offset, grid, v, dt, tag=None):
    del grid, dt  # not used.
    # TODO(dkochkov) Add decorator to expand/squeeze channel dim.
    c = grids.AlignedArray(jnp.expand_dims(c.data, -1), c.offset)
    # TODO(jamieas): Try removing the following line.
    if c.offset == offset: return c
    key = (c.offset, offset, tag)
    interpolator = self._interpolators.get(key)
    if interpolator is None:
      raise KeyError(f'No interpolator for key {key}. '
                     f'Available keys: {list(self._interpolators.keys())}')
    return grids.AlignedArray(interpolator(c.data)[..., 0], offset)


@gin.configurable
class IndividualLearnedInterpolation:
  """Trainable interpolation module.

  This module uses a collection of SpatialDerivative modules that are applied
  to inputs based on the combination of initial and target offsets. Currently
  no symmetries are implemented and every new pair of offsets gets a separate
  network.
  """

  def __init__(
      self,
      grid: grids.Grid,
      dt: float,
      physics_specs: physics_specifications.BasePhysicsSpecs,
      v,
      stencil_size=4,
      tower_factory=towers.forward_tower_factory,
  ):
    del v, dt, physics_specs  # unused.
    self._ndim = grid.ndim
    self._tower_factory = functools.partial(tower_factory, ndim=grid.ndim)
    self._stencil_sizes = (stencil_size,) * self._ndim
    self._steps = grid.step
    self._modules = {}

  def _get_interpolation_module(self, offsets):
    """Constructs or retrieves a learned interpolation module."""
    if offsets in self._modules:
      return self._modules[offsets]
    inputs_offset, target_offset = offsets
    self._modules[offsets] = layers.SpatialDerivative(
        self._stencil_sizes, inputs_offset, target_offset,
        (0,) * self._ndim, self._tower_factory, self._steps)
    return self._modules[offsets]

  def __call__(self, c, offset, grid, v, dt):
    """Interpolates `c` to `offset`."""
    del dt  # not used.
    if c.offset == offset: return c
    offsets = (c.offset, offset)
    c_input = jnp.expand_dims(c.data, axis=-1)
    aux_inputs = [jnp.expand_dims(u.data, axis=-1) for u in v]
    res = self._get_interpolation_module(offsets)(c_input, *aux_inputs)
    return grids.AlignedArray(jnp.squeeze(res, axis=-1), offset)


@gin.configurable
def linear(*args, **kwargs):
  del args, kwargs
  return interpolation.linear


@gin.configurable
def upwind(*args, **kwargs):
  del args, kwargs
  return interpolation.upwind


@gin.configurable
def lax_wendroff(*args, **kwargs):
  del args, kwargs
  return interpolation.lax_wendroff


# TODO(dkochkov) make flux limiters configurable.
@gin.configurable
def tvd_limiter_transformation(
    interpolation_fn: InterpolationFn,
    limiter_fn: FluxLimiter = interpolation.van_leer_limiter,
) -> InterpolationFn:
  """Transformation function that applies flux limiter to `interpolation_fn`."""
  return interpolation.apply_tvd_limiter(interpolation_fn, limiter_fn)


@gin.configurable
def transformed(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    v: AlignedField,
    base_interpolation_module: InterpolationModule = lax_wendroff,
    transformation: InterpolationTransform = tvd_limiter_transformation,
) -> InterpolationFn:
  """Interpolation module that augments interpolation of the base module.

  This module generates interpolation method that consists of that generated
  by `base_interpolation_module` transformed by `transformation`. This allows
  implementation of additional constraints such as TVD, in which case
  `transformation` should apply a TVD limiter.

  Args:
    grid: grid on which the Navier-Stokes equation is discretized.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation module.
    v: input velocity field potentially used to pre-compute interpolations.
    base_interpolation_module: base interpolation module to use.
    transformation: transformation to apply to base interpolation function.

  Returns:
    Interpolation function.
  """
  interpolation_fn = base_interpolation_module(grid, dt, physics_specs, v=v)
  interpolation_fn = transformation(interpolation_fn)
  return interpolation_fn
