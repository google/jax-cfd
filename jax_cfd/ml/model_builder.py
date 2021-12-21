"""Defines AbstractModel API, standard implementations and helper functions."""

import functools
from typing import Callable, Optional

import gin
import haiku as hk
from jax_cfd.base import grids
# Note: decoders, encoders and equations contain standard gin-configurables;
from jax_cfd.ml import decoders  # pylint: disable=unused-import
from jax_cfd.ml import encoders  # pylint: disable=unused-import
from jax_cfd.ml import equations  # pylint: disable=unused-import
from jax_cfd.ml import physics_specifications


# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic


def _identity(x):
  return x


class DynamicalSystem(hk.Module):
  """Abstract class for modeling dynamical systems."""

  def __init__(
      self,
      grid: grids.Grid,
      dt: float,
      physics_specs: physics_specifications.BasePhysicsSpecs,
      name: Optional[str] = None
  ):
    """Constructs an instance of a class."""
    super().__init__(name=name)
    self.grid = grid
    self.dt = dt
    self.physics_specs = physics_specs

  def encode(self, x):
    """Encodes input trajectory `x` to the model state."""
    raise NotImplementedError("Model subclass did not define encode")

  def decode(self, x):
    """Decodes a model state `x` to a data representation."""
    raise NotImplementedError("Model subclass did not define decode")

  def advance(self, x):
    """Returns a model state `x` advanced in time by `self.dt`."""
    raise NotImplementedError("Model subclass did not define advance")

  def trajectory(
      self,
      x,
      outer_steps: int,
      inner_steps: int = 1,
      *,
      start_with_input: bool = False,
      post_process_fn: Callable = _identity,
  ):
    """Returns a final model state and trajectory."""
    return trajectory_from_step(
        self.advance, outer_steps, inner_steps,
        start_with_input=start_with_input,
        post_process_fn=post_process_fn
    )(x)


@gin.register
class ModularStepModel(DynamicalSystem):
  """Dynamical model based on independent encoder/decoder/step components."""

  def __init__(
      self,
      grid: grids.Grid,
      dt: float,
      physics_specs: physics_specifications.BasePhysicsSpecs,
      advance_module=gin.REQUIRED,
      encoder_module=gin.REQUIRED,
      decoder_module=gin.REQUIRED,
      name: Optional[str] = None
  ):
    """Constructs an instance of a class."""
    super().__init__(grid=grid, dt=dt, physics_specs=physics_specs, name=name)
    self.advance_module = advance_module(grid, dt, physics_specs)
    self.encoder_module = encoder_module(grid, dt, physics_specs)
    self.decoder_module = decoder_module(grid, dt, physics_specs)

  def encode(self, x):
    return self.encoder_module(x)

  def decode(self, x):
    return self.decoder_module(x)

  def advance(self, x):
    return self.advance_module(x)


@gin.configurable
def get_model_cls(grid, dt, physics_specs, model_cls=gin.REQUIRED):
  """Returns a configured model class."""
  return functools.partial(model_cls, grid, dt, physics_specs)


def repeated(fn: Callable, steps: int) -> Callable:
  """Returns a repeatedly applied version of fn()."""
  def f_repeated(x_initial):
    g = lambda x, _: (fn(x), None)
    x_final, _ = hk.scan(g, x_initial, xs=None, length=steps)
    return x_final
  return f_repeated


@gin.configurable(allowlist=("set_checkpoint",))
def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool,
    post_process_fn: Callable,
    set_checkpoint: bool = False,
):
  """Returns a function that accumulates repeated applications of `step_fn`.

  Compute a trajectory by repeatedly calling `step_fn()`
  `outer_steps * inner_steps` times.

  Args:
    step_fn: function that takes a state and returns state after one time step.
    outer_steps: number of steps to save in the generated trajectory.
    inner_steps: number of repeated calls to step_fn() between saved steps.
    start_with_input: if True, output the trajectory at steps [0, ..., steps-1]
      instead of steps [1, ..., steps].
    post_process_fn: function to apply to trajectory outputs.
    set_checkpoint: whether to use `jax.checkpoint` on `step_fn`.

  Returns:
    A function that takes an initial state and returns a tuple consisting of:
      (1) the final frame of the trajectory.
      (2) trajectory of length `outer_steps` representing time evolution.
  """
  if set_checkpoint:
    step_fn = hk.remat(step_fn)

  if inner_steps != 1:
    step_fn = repeated(step_fn, inner_steps)

  def step(carry_in, _):
    carry_out = step_fn(carry_in)
    frame = carry_in if start_with_input else carry_out
    return carry_out, post_process_fn(frame)

  def multistep(x):
    return hk.scan(step, x, xs=None, length=outer_steps)

  return multistep
