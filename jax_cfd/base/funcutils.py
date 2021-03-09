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

"""JAX utility functions for JAX-CFD."""

import contextlib
from typing import Any, Callable, Sequence

import jax
from jax import tree_util
import jax.numpy as jnp


# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic
# Not accurate for contextmanager
# pylint: disable=g-doc-return-or-yield

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any


_INITIALIZING = 0


@contextlib.contextmanager
def init_context():
  """Creates a context in which scan() only evaluates f() once.

  This is useful for initializing a neural net with Haiku that involves modules
  that are applied inside scan(). Within init_context(), these modules are only
  called once. This allows us to preserve the pre-omnistaging behavior of JAX,
  e.g., so we can initialize a neural net module pass directly into a scanned
  function.
  """
  global _INITIALIZING
  _INITIALIZING += 1
  try:
    yield
  finally:
    _INITIALIZING -= 1


def _tree_stack(trees: Sequence[PyTree]) -> PyTree:
  if trees:
    return tree_util.tree_multimap(lambda *xs: jnp.stack(xs), *trees)
  else:
    return trees


def scan(f, init, xs, length=None):
  """A version of jax.lax.scan that supports init_context()."""
  # Note: we use our own version of scan rather than haiku.scan() because
  # haiku.scan() only support use inside haiku modules, but we want to be able
  # to use the same scan function even when not using haiku.
  if _INITIALIZING:
    xs_flat, treedef = tree_util.tree_flatten(xs)
    if length is None:
      length, = {x.shape[0] for x in xs_flat}
    x0 = tree_util.tree_unflatten(treedef, [x[0, ...] for x in xs_flat])
    carry, y0 = f(init, x0)
    # Create a dummy-output of the right shape while only calling f() once.
    ys = _tree_stack(length * [y0])
    return carry, ys
  return jax.lax.scan(f, init, xs, length)


def repeated(f: Callable, steps: int) -> Callable:
  """Returns a repeatedly applied version of f()."""
  def f_repeated(x_initial):
    g = lambda x, _: (f(x), None)
    x_final, _ = scan(g, x_initial, xs=None, length=steps)
    return x_final
  return f_repeated


def _identity(x):
  return x


def trajectory(
    step_fn: Callable,
    steps: int,
    post_process: Callable = _identity,
    *,
    start_with_input: bool = False,
) -> Callable:
  """Returns a function that accumulates repeated applications of `step_fn`.

  Args:
    step_fn: function that takes a state and returns state after one time step.
    steps: number of steps to take when generating the trajectory.
    post_process: transformation to be applied to each frame of the trajectory.
    start_with_input: if True, output the trajectory at steps [0, ..., steps-1]
      instead of steps [1, ..., steps].

  Returns:
    A function that takes an initial state and returns a tuple consisting of:
      (1) the final frame of the trajectory _before_ `post_process` is applied.
      (2) trajectory of length `steps` representing time evolution.
  """
  # TODO(shoyer): change the default to start_with_input=True, once we're sure
  # it works for training.
  def step(carry_in, _):
    carry_out = step_fn(carry_in)
    frame = post_process(carry_in if start_with_input else carry_out)
    return carry_out, frame

  def multistep(values):
    return scan(step, values, xs=None, length=steps)

  return multistep
