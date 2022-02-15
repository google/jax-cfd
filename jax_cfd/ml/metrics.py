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

"""Metrics for whirl experiments."""

import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils as arr_utils
from jax_cfd.base import grids

Array = grids.Array
PyTree = Any


# TODO(shoyer): rewrite these metric functions in terms of vmap rather than
# explicit batch and time dimensions. Also consider go/clu-metrics.


def l2_loss_cumulative(trajectory: Tuple[Array, ...],
                       target: Tuple[Array, ...],
                       n: Optional[int] = None,
                       scale: float = 1,
                       time_axis=0) -> float:
  """Computes cumulative L2 loss on first `n` time slices."""
  trajectory = arr_utils.slice_along_axis(trajectory, time_axis, slice(None, n))
  target = arr_utils.slice_along_axis(target, time_axis, slice(None, n))
  return sum((scale * jnp.square(x - y)).sum()
             for x, y in zip(trajectory, target))


def l2_loss_single_step(trajectory: Tuple[Array, ...],
                        target: Tuple[Array, ...],
                        n: int,
                        scale: float = 1,
                        time_axis=0) -> float:
  """Computes L2 loss on `n`th time slice."""
  trajectory = arr_utils.slice_along_axis(trajectory, time_axis, n)
  target = arr_utils.slice_along_axis(target, time_axis, n)
  return sum((scale * jnp.square(x - y)).sum()
             for x, y in zip(trajectory, target))


def _normalize(array: Array, state_axes: Tuple[int, ...]) -> Array:
  l2_norm = (array ** 2).sum(axis=state_axes, keepdims=True) ** 0.5
  return array / l2_norm


def correlation_single_step(trajectory: Tuple[Array, ...],
                            target: Tuple[Array, ...],
                            n: int,
                            time_axis=0,
                            batch_axis=1) -> float:
  """Computes correlation on the `n`th time slice."""
  trajectory = jnp.stack(arr_utils.slice_along_axis(trajectory, time_axis, n))
  target = jnp.stack(arr_utils.slice_along_axis(target, time_axis, n))
  state_axes = tuple(axis if axis <= time_axis else axis - 1
                     for axis in range(trajectory.ndim)
                     if axis != time_axis and axis != batch_axis)
  trajectory_normalized = _normalize(trajectory, state_axes)
  target_normalized = _normalize(target, state_axes)
  return (trajectory_normalized * target_normalized).sum(axis=state_axes).mean()


def local_reduce(metric: Callable[..., Array],
                 reduction_function: Callable[[Array, int], Array],
                 batch_axis: int = 0) -> Callable[..., Array]:
  """Computes the mean of a metric over a local batch.

  Example usage:
  ```
  average_l2_loss = metrics.local_reduction(
      functools.partial(metrics.l2_loss_cumulative, n=10), jnp.mean)
  ```

  Args:
    metric: a callable that returns a single array.
    reduction_function: a callable that takes arguments `x, axis` and returns
      a single array. For example, `jnp.mean`.
    batch_axis: an integer indicating the batch dimension. Defaults to 0.

  Returns:
    A function that takes the same arguments as `metric` but computes the mean
    along the batch axis.
  """
  def reduced_metric(*args, **kwargs):
    metric_value = jax.vmap(metric, batch_axis)(*args, **kwargs)
    return reduction_function(metric_value, batch_axis)
  return reduced_metric


def distributed_reduce(metric: Callable[..., Array],
                       reduction_function: Callable[[Array, str], Array],
                       axis_name: str = 'batch') -> Callable[..., Array]:
  """Computes the mean of a metric over a distributed batch.

  Note that the functions returned are only suitable for use inside another
  function that is pmapped along the same axis.

  Example usage:
  ```
  average_l2_loss = metrics.distributed_reduce(
    functools.partial(metrics.l2_loss_cumulative, n=10),
    reduction_function=jax.lax.pmean)

  @functools.partial(jax.pmap, axis_name='batch')
  def distributed_train_step(...):
    prediction = ...
    loss = average_l2_loss(prediction, target)
  ```

  Args:
    metric: a callable that takes locally batched arrays and returns a single
      array.
    reduction_function: a callable that takes arguments `x, axis_name` and
      returns a single array. For example, `jax.lax.pmean`.
    axis_name: the name that will be used for distributing metric computation
      and combining results

  Returns:
    A function that takes the same arguments as `metric` but computes the mean
    along the axis specified by `axis_name`.
  """

  def reduced_metric(*args, **kwargs):
    metric_value = metric(*args, **kwargs)
    return reduction_function(metric_value, axis_name)
  return reduced_metric


local_mean = functools.partial(local_reduce, reduction_function=jnp.mean)
local_sum = functools.partial(local_reduce, reduction_function=jnp.sum)
distributed_mean = functools.partial(
    distributed_reduce, reduction_function=jax.lax.pmean)
distributed_sum = functools.partial(
    distributed_reduce, reduction_function=jax.lax.psum)
