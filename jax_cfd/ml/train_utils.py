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

"""Often used training utility funcutils."""

import collections

from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple, Union

import gin
import jax
from jax_cfd.base import array_utils
from jax_cfd.base import grids


# TODO(dkochkov): make utility functions agnostic of jax_cfd/other models;
Array = grids.Array
Field = Tuple[Array, ...]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector

# TODO(jamieas): Replace `Any` with well-defined types.
IntOrArray = Union[int, Array]
NavierStokesState = Tuple[GridArrayVector, GridArray, Optional[GridArray]]
Velocity = Field
OptimizerState = Any
ModelParams = Any
ModelGradients = ModelParams
EMAParams = ModelParams
StepAndOptimizerState = Tuple[IntOrArray, OptimizerState]
StepOptAndEMAState = Tuple[IntOrArray, OptimizerState, ModelParams]
LossValue = Array
LossFunction = Callable[[GridArrayVector, GridArrayVector], LossValue]
LossAndGradFunction = Callable[[ModelParams, Array],
                               Tuple[LossValue, ModelGradients]]
MetricFunction = Callable[[Velocity, Velocity], Array]
TrainStepFunction = Callable[[StepAndOptimizerState, Array],
                             Tuple[OptimizerState, Array]]
EvalStepFunction = Callable[[OptimizerState, Field],
                            Mapping[str, Array]]
TrajectoryFunction = Callable[[ModelParams, Velocity],
                              Tuple[Velocity, Velocity]]


#
#  Note that all functions below deal with *batched* inputs.
#


def loss_and_gradient(
    trajectory_fn: TrajectoryFunction,
    loss_fn: LossFunction
) -> LossAndGradFunction:
  """Returns a function that computes loss and the gradient of the loss.

  Args:
    trajectory_fn: a function that accepts `params` and `initial_velocity`
      and returns a trajectory of velocities.
    loss_fn: a function that accepts a predicted trajectory and a ground truth
      trajectory, returning a scalar loss value.

  Returns:
    A function that accepts `params, initial_velocity, target_trajectory` and
    returns the loss and the gradient of the loss.
  """
  def _loss(params: ModelParams, target_trajectory: Velocity) -> LossValue:
    """Returns loss value and gradient with respect to model parameters."""
    _, predicted_trajectory = trajectory_fn(params, target_trajectory)
    loss = loss_fn(predicted_trajectory, target_trajectory)  # type: ignore
    return loss
  return jax.value_and_grad(_loss)


def train_step(
    loss_and_grad_fn: LossAndGradFunction,
    update_fn: Callable[[int, ModelGradients, OptimizerState], OptimizerState],
    get_params_fn: Callable[[OptimizerState], ModelParams]
) -> TrainStepFunction:
  """Returns a function that performs a single training step.

  Args:
    loss_and_grad_fn: a function that accepts `params, initial_velocity,
      target_trajectory` and returns the loss and the gradient of the loss.
    update_fn: a function that accepts `step_num, gradients, optimizer_state`
      and returns an updated optimizer state.
    get_params_fn: a function that accepts `optimizer_state` and returns model
      params. If the state is encoded by params, this should be the identity.

  Returns:
    A function that performs a single training step.
  """
  def _train_step(
      step_and_state: StepAndOptimizerState,
      target_trajectory: Array,
  ) -> Tuple[StepAndOptimizerState, LossValue]:
    """A function that performs a single training step."""
    step, optimizer_state = step_and_state
    params = get_params_fn(optimizer_state)
    loss, grad = loss_and_grad_fn(params, target_trajectory)
    optimizer_state = update_fn(step, grad, optimizer_state)
    return (step + 1, optimizer_state), loss
  return _train_step


def eval_batch(
    trajectory_fn: TrajectoryFunction,
    metric_funcs: Mapping[str, MetricFunction],
) -> EvalStepFunction:
  """Returns a function that performs a single evaluation step.

  Args:
    trajectory_fn: a function that accepts `params` and `initial_velocity`
      and returns a trajectory of velocities.
    metric_funcs: a dictionary mapping strings to metric funcutils.

  Returns:
    A function that performs a single evaluation step.
  """
  def _eval_batch(
      params: ModelParams,
      target_trajectory: Velocity,
  ) -> Mapping[str, Array]:
    """A function that performs a single training step."""
    _, predicted_trajectory = trajectory_fn(params, target_trajectory)
    metric_values = {k: metric(predicted_trajectory, target_trajectory)
                     for k, metric in metric_funcs.items()}
    return metric_values
  return _eval_batch


def streaming_mean(
    batches: Iterable[Velocity],
    eval_fn: Callable[[Field], Mapping[str, Array]],
) -> Mapping[str, Array]:
  """Runs evaluation on `eval_data`.

  Args:
    batches: an iterable of batched velocity trajectories.
    eval_fn: a function that performs a single evaluation step.

  Returns:
    A dict mapping strings to metric values.

  Raises:
    RuntimeError: if there are no batches to iterate over.
  """
  # TODO(jamieas): update to accommodate non-scalar metrics.
  eval_metrics = collections.defaultdict(float)
  count = 0
  for batch in batches:
    batch_metrics = eval_fn(batch)
    for k, v in batch_metrics.items():
      eval_metrics[k] += v
    count += 1
  if not count:
    raise RuntimeError("no batches to iterate over")
  return {k: v / count for k, v in eval_metrics.items()}


@gin.register
def identity(batch: Tuple[Array, ...], rng: Array = None) -> Tuple[Array, ...]:
  """Identity preprocessing function that does not modify the `batch`."""
  del rng  # unused.
  return batch


# TODO(dkochkov) consider adding an option to perform pressure projection step.
@gin.configurable
def add_noise_to_input_frame(
    batch: Tuple[Array, ...],
    rng: Array,
    scale: float = 1e-2,
    **kwargs
) -> Tuple[Array, ...]:
  """Adds noise to the 0th time frame in the `batch`.

  Args:
    batch: original batch to which the noise will be added.
    rng: random number key to be used to generate noise.
    scale: scale of the normal noise to be added.
    **kwargs: other keyword arguments. Not used.

  Returns:
    batch with noise added along the 0th time slice.
  """
  del kwargs  # unused.
  time_zero_slice = array_utils.slice_along_axis(batch, 1, 0)
  shapes = jax.tree_map(lambda x: x.shape, time_zero_slice)
  rngs = jax.random.split(rng, len(jax.tree_leaves(time_zero_slice)))
  # TODO(dkochkov) add `split_like` method to `array_utils.py`.
  rngs = jax.tree_unflatten(jax.tree_structure(time_zero_slice), rngs)
  noise_fn = lambda key, s: scale * jax.random.truncated_normal(key, -2., 2., s)
  noise = jax.tree_map(noise_fn, rngs, shapes)
  add_noise_fn = lambda x, n: x.at[:, 0, ...].add(n)
  return jax.tree_map(add_noise_fn, batch, noise)


def preprocess(
    data_iterator: Iterator[Tuple[Array, ...]],
    rng_stream: Iterator[Array],
    preprocess_fn: Callable[..., Tuple[Array, ...]]
):
  """Generator that applies `preprocess_fn` to entries of the `data_iterator`.

  Args:
    data_iterator: numpy iterator holding the data.
    rng_stream: stream of random numbers to be used by `preprocess_fn`.
    preprocess_fn: preprocessing function to be applied to each batch of data.

  Yields:
    Batch of data from `data_iterator` preprocessed with `preprocess_fn`.
  """
  preprocess_fn = jax.jit(preprocess_fn)
  while True:
    rng = next(rng_stream)
    yield preprocess_fn(next(data_iterator), rng)
