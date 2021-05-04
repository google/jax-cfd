"""Decoder modules that help interfacing model states with output data.

All decoder modules generate a function that given an specific model state
return the observable data of the same structure as provided to the Encoder.
Decoders can be either fixed functions, decorators, or learned modules.
"""

from typing import Any, Callable, Optional
import gin
from jax_cfd.base import array_utils
from jax_cfd.base import grids
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers


DecodeFn = Callable[[Any], Any]  # maps model state to data time slice.
DecoderModule = Callable[..., DecodeFn]  # generate DecodeFn closed over args.
TowerFactory = towers.TowerFactory


@gin.register
def identity_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Identity decoder module that returns model state as is."""
  del grid, dt, physics_specs  # unused.
  def decode_fn(inputs):
    return inputs

  return decode_fn


# TODO(dkochkov) generalize this to arbitrary pytrees.
@gin.register
def aligned_array_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Generates decoder that extracts data from AlignedArrays."""
  del grid, dt, physics_specs  # unused.
  def decode_fn(inputs):
    return tuple(x.data for x in inputs)

  return decode_fn


@gin.configurable
def channels_split_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Generates decoder that splits channels into data tuples."""
  del grid, dt, physics_specs  # unused.
  def decode_fn(inputs):
    return array_utils.split_axis(inputs, -1)

  return decode_fn


@gin.configurable
def latent_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: TowerFactory,
    num_components: Optional[int] = None,
):
  """Generates trainable decoder that maps latent representation to data tuple.

  Decoder first computes an array of outputs using network specified by a
  `tower_factory` and then splits the channels into `num_components` components.

  Args:
    grid: grid representing spatial discritization of the system.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation.
    tower_factory: factory that produces trainable tower network module.
    num_components: number of data tuples in the data representation of the
      state. If None, assumes num_components == grid.ndims. Default is None.

  Returns:
    decode function that maps latent state `inputs` at given time to a tuple of
    `num_components` data arrays representing the same state at the same time.
  """
  split_channels_fn = channels_split_decoder(grid, dt, physics_specs)

  def decode_fn(inputs):
    num_channels = num_components or grid.ndim
    decoder_tower = tower_factory(num_channels, grid.ndim, name='decoder')
    return split_channels_fn(decoder_tower(inputs))

  return decode_fn
