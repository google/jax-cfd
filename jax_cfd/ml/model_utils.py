"""Helper methods for constructing trajectory functions in model_builder.py."""

import functools
from jax_cfd.base import array_utils


def with_preprocessing(fn, preprocess_fn):
  """Generates a function that computes `fn` on `preprocess_fn(x)`."""
  @functools.wraps(fn)
  def apply_fn(x, *args, **kwargs):
    return fn(preprocess_fn(x), *args, **kwargs)

  return apply_fn


def with_post_processing(fn, post_process_fn):
  """Generates a function that applies `post_process_fn` to outputs of `fn`."""
  @functools.wraps(fn)
  def apply_fn(*args, **kwargs):
    return post_process_fn(*fn(*args, **kwargs))

  return apply_fn


def with_split_input(fn, split_index, time_axis=0):
  """Decorates `fn` to be evaluated on first `split_index` time slices.

  The returned function is a generalization to pytrees of the function:
  `fn(x[:split_index], *args, **kwargs)`

  Args:
    fn: function to be transformed.
    split_index: number of input elements along the time axis to use.
    time_axis: axis corresponding to time dimension in `x` to decorated `fn`.

  Returns:
    decorated `fn` that is evaluated on only `split_index` first time slices of
    provided inputs.
  """
  @functools.wraps(fn)
  def apply_fn(x, *args, **kwargs):
    init, _ = array_utils.split_along_axis(x, split_index, axis=time_axis)
    return fn(init, *args, **kwargs)

  return apply_fn


def with_input_included(trajectory_fn, time_axis=0):
  """Returns a `trajectory_fn` that concatenates inputs `x` to trajectory."""
  @functools.wraps(trajectory_fn)
  def _trajectory(x, *args, **kwargs):
    final, unroll = trajectory_fn(x, *args, **kwargs)
    return final, array_utils.concat_along_axis([x, unroll], time_axis)

  return _trajectory


def decoded_trajectory_with_inputs(model, num_init_frames):
  """Returns trajectory_fn operating on decoded data.

  The returned function uses `num_init_frames` of the physics space trajectory
  provided as an input to initialize the model state, unrolls the trajectory of
  specified length that is decoded to the physics space using `model.decode_fn`.

  Args:
    model: model of a dynamical system used to obtain the trajectory.
    num_init_frames: number of time frames used from the physics trajectory to
      initialize the model state.

  Returns:
    Trajectory function that operates on physics space trajectories and returns
    unrolls in physics space.
  """
  def _trajectory_fn(x, steps, repeated_length=1):
    trajectory_fn = functools.partial(
        model.trajectory, post_process_fn=model.decode)
    # add preprocessing to convert data to model state.
    trajectory_fn = with_preprocessing(trajectory_fn, model.encode)
    # concatenate input trajectory to output trajectory for easier comparison.
    trajectory_fn = with_input_included(trajectory_fn)
    # make trajectories operate on full examples by splitting the init.
    trajectory_fn = with_split_input(trajectory_fn, num_init_frames)
    return trajectory_fn(x, steps, repeated_length)

  return _trajectory_fn
