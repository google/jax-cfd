"""Encoder modules that help interfacing input trajectories to model states.

All encoder modules generate a function that given an input trajectory infers
the final state of the physical system in the representation defined by the
Encoder. Encoders can be either fixed functions, decorators or learned modules.
The input state is expected to consist of arrays with `time` as a leading axis.
"""

from typing import Any, Callable, Optional, Tuple
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers


EncodeFn = Callable[[Any], Any]  # maps input trajectory to final model state.
EncoderModule = Callable[..., EncodeFn]  # generate EncodeFn closed over args.
TowerFactory = towers.TowerFactory


@gin.register
def aligned_array_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> EncodeFn:
  """Generates encoder that wraps last data slice as GridVariables."""
  del dt, physics_specs  # unused.
  data_offsets = data_offsets or grid.cell_faces
  slice_last_fn = lambda x: array_utils.slice_along_axis(x, 0, -1)

  # TODO(pnorgaard) Make the encoder/decoder/network register for BC
  def encode_fn(inputs):
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    return tuple(
        grids.GridVariable(grids.GridArray(slice_last_fn(x), offset, grid), bc)
        for x, offset in zip(inputs, data_offsets))

  return encode_fn


@gin.register
def collocated_to_staggered_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> EncodeFn:
  """Encoder that interpolates from collocated to staggered grids."""
  del dt, physics_specs  # unused.
  slice_last_fn = lambda x: array_utils.slice_along_axis(x, 0, -1)

  def encode_fn(inputs):
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    src_offset = grid.cell_center
    pre_interp = tuple(
        grids.GridVariable(
            grids.GridArray(slice_last_fn(x), src_offset, grid), bc)
        for x in inputs)
    return tuple(interpolation.linear(c, offset)
                 for c, offset in zip(pre_interp, grid.cell_faces))

  return encode_fn


@gin.register
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


@gin.register
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


@gin.register
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


@gin.register
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

  return hk.to_module(encode_fn)()


@gin.register
def aligned_latent_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: TowerFactory,
    num_latent_dims: int,
    n_frames: int,
    time_axis: int = 0,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
):
  """Latent encoder that decodes to GridVariables."""
  data_offsets = data_offsets or grid.cell_faces
  stack_inputs_fn = stack_last_n_state_encoder(
      grid, dt, physics_specs, n_frames, time_axis)

  def encode_fn(inputs):
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    inputs = stack_inputs_fn(inputs)
    encoder_tower = tower_factory(num_latent_dims, grid.ndim, name='encoder')
    raw_outputs = encoder_tower(inputs)
    split_outputs = [raw_outputs[..., i] for i in range(raw_outputs.shape[-1])]
    return tuple(
        grids.GridVariable(grids.GridArray(x, offset, grid), bc)
        for x, offset in zip(split_outputs, data_offsets))

  return hk.to_module(encode_fn)()


@gin.register
def vorticity_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> EncodeFn:
  """Maps velocity to vorticity."""
  del dt, physics_specs, data_offsets  # unused.
  slice_last_fn = lambda x: array_utils.slice_along_axis(x, 0, -1)

  def encode_fn(inputs):
    u, v = inputs
    u, v = slice_last_fn(u), slice_last_fn(v)
    uhat, vhat = jnp.fft.rfft2(u), jnp.fft.rfft2(v)
    kx, ky = grid.rfft_mesh()
    vorticity_hat = 2j * jnp.pi * (vhat * kx - uhat * ky)
    # TODO(dresdner) main difference is that the output is ifft'ed.
    # TODO(dresdner) and also that the output has a channel dim.
    return jnp.fft.irfft2(vorticity_hat)[..., jnp.newaxis]

  return encode_fn


@gin.register
def vorticity_velocity_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> EncodeFn:
  """Maps velocity to [velocity; vorticity]."""
  del dt, physics_specs, data_offsets  # unused.
  slice_last_fn = lambda x: array_utils.slice_along_axis(x, 0, -1)
  ifft = jnp.fft.irfft2

  def encode_fn(inputs):
    u, v = inputs
    u, v = slice_last_fn(u), slice_last_fn(v)
    uhat, vhat = jnp.fft.rfft2(u), jnp.fft.rfft2(v)
    kx, ky = grid.rfft_mesh()
    vorticity_hat = 2j * jnp.pi * (vhat * kx - uhat * ky)
    return jnp.stack([u, v, ifft(vorticity_hat)], axis=-1)

  return encode_fn


@gin.register
def spectral_vorticity_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> EncodeFn:
  """Generates encoder that wraps last data slice as GridVariables."""
  del dt, physics_specs, data_offsets  # unused.
  slice_last_fn = lambda x: array_utils.slice_along_axis(x, 0, -1)

  def encode_fn(inputs):
    u, v = inputs
    u, v = slice_last_fn(u), slice_last_fn(v)
    uhat, vhat = jnp.fft.rfft2(u), jnp.fft.rfft2(v)
    kx, ky = grid.rfft_mesh()
    vorticity = 2j * jnp.pi * (vhat * kx - uhat * ky)
    return vorticity

  return encode_fn
