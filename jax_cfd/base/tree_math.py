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
"""Support math with JAX pytrees.

To be removed in favor of a more comprehensive solution like
https://github.com/google/jax/pull/8504 when available.
"""

import operator
from jax import tree_util


@tree_util.register_pytree_node_class
class Vector:
  """A pytree wrapper to support basic infix arithmetic."""

  def __init__(self, pytree):
    self.pytree = pytree

  def tree_flatten(self):
    return (self.pytree,), None

  @classmethod
  def tree_unflatten(cls, _, args):
    return cls(*args)

  def __add__(self, other):
    if isinstance(other, Vector):
      result = tree_util.tree_map(operator.add, self.pytree, other.pytree)
    else:
      result = tree_util.tree_map(lambda x: x + other, self.pytree)
    return Vector(result)

  __radd__ = __add__

  def __mul__(self, other):
    if isinstance(other, Vector):
      result = tree_util.tree_map(operator.mul, self.pytree, other.pytree)
    else:
      result = tree_util.tree_map(lambda x: x * other, self.pytree)
    return Vector(result)

  __rmul__ = __mul__

  def __truediv__(self, other):
    if isinstance(other, Vector):
      result = tree_util.tree_map(operator.truediv, self.pytree, other.pytree)
    else:
      result = tree_util.tree_map(lambda x: x / other, self.pytree)
    return Vector(result)


def vector_to_pytree_fun(func):
  """Make a pytree -> pytree function from a vector -> vector function."""
  def wrapper(state):
    return func(Vector(state)).pytree
  return wrapper


def pytree_to_vector_fun(func):
  """Make a vector -> vector function from a pytree -> pytree function."""
  def wrapper(state, *aux_args):
    return Vector(func(state.pytree, *aux_args))
  return wrapper
