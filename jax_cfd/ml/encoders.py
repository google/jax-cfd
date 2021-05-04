"""Encoder modules that help interfacing input trajectories to model states.

All encoder modules generate a function that given an input trajectory infers
the final state of the physical system in the representation defined by the
Encoder. Encoders can be either fixed functions, decorators or learned modules.
The input state is expected to consist of arrays with `time` as a leading axis.
"""

from typing import Any, Callable, Optional, Tuple
import gin
import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import grids
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers


EncodeFn = Callable[[Any], Any]  # maps input trajectory to final model state.
EncoderModule = Callable[..., EncodeFn]  # generate EncodeFn closed over args.
TowerFactory = towers.TowerFactory


@gin.configurable
def aligned_array_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> EncodeFn:
  """Generates encoder that wraps last data slice as AlignedArrays."""
  del dt, physics_specs  # unused.
  data_offsets = data_offsets or grid.cell_faces
  slice_last_fn = lambda x: array_utils.slice_along_axis(x, 0, -1)

  def encode_fn(inputs):
    return tuple(grids.AlignedArray(slice_last_fn(x), offset)
                 for x, offset in zip(inputs, data_offsets))

  return encode_fn


@gin.configurable
def slice_last_state_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    time_axis=0,
) -> EncodeFn:
  """Generates encoder that returns last data slice along time axis."""
  del grid, dt, physics_specs  # unused.
  def encode_fn(inputs):
    return array_utils.slice_along_axis(inputs, time_axis, -1)
  return encode_fn


@gin.configurable
def slice_last_n_state_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    n: int = gin.REQUIRED,
    time_axis: int = 0,
) -> EncodeFn:
  """Generates encoder that returns last `n` data slices along last axis."""
  del grid, dt, physics_specs  # unused.
  def encode_fn(inputs):
    init_slice = array_utils.slice_along_axis(inputs, 0, slice(-n, None))
    return jax.tree_map(lambda x: jnp.moveaxis(x, time_axis, -1), init_slice)
  return encode_fn


@gin.configurable
def stack_last_n_state_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    n: int = gin.REQUIRED,
    time_axis: int = 0,
) -> EncodeFn:
  """Generates encoder that stacks last `n` inputs slices along last axis."""
  del grid, dt, physics_specs  # unused.
  def encode_fn(inputs):
    inputs = array_utils.slice_along_axis(inputs, 0, slice(-n, None))
    inputs = jax.tree_map(lambda x: jnp.moveaxis(x, time_axis, -1), inputs)
    return array_utils.concat_along_axis(jax.tree_leaves(inputs), axis=-1)

  return encode_fn


@gin.configurable
def latent_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: TowerFactory,
    num_latent_dims: int,
    n_frames: int,
    time_axis: int = 0,
):
  """Generates trainable encoder that maps inputs to a latent representation.

  Encoder first stacks last `n_frames` time slices in input trajectory along
  channels and then applies a network specified by a `tower_factory` to obtain
  a latent field representation with `num_latent_dims` channel dimensions.

  Args:
    grid: grid representing spatial discritization of the system.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation.
    tower_factory: factory that produces trainable tower network module.
    num_latent_dims: number of channels to have in latent representation.
    n_frames: number of last frames in input trajectory to use for encoding.
    time_axis: axis in input trajectory that correspond to time.

  Returns:
    encode function that maps input trajectory `inputs` to a latent field
    representation with `num_latent_dims`. Note that depending on the tower used
    the spatial dimension of the representation might differ from `inputs`.
  """

  stack_inputs_fn = stack_last_n_state_encoder(
      grid, dt, physics_specs, n_frames, time_axis)

  def encode_fn(inputs):
    inputs = stack_inputs_fn(inputs)
    encoder_tower = tower_factory(num_latent_dims, grid.ndim, name='encoder')
    return encoder_tower(inputs)

  return encode_fn
