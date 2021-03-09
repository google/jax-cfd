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

"""Tests for jax_cfd.grids."""

# TODO(jamieas): Consider updating these tests using the `hypothesis` framework.
import functools
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np


class GridTest(test_util.TestCase):

  def test_constructor(self):
    grid = grids.Grid((10,))
    self.assertEqual(grid.shape, (10,))
    self.assertEqual(grid.step, (1.0,))
    self.assertEqual(grid.domain, ((0, 10.0),))
    self.assertEqual(grid.boundaries, ('periodic',))
    axis, = grid.axes()
    self.assertAllClose(axis, 0.5 + np.arange(10))

    grid = grids.Grid((10,), step=0.1, boundaries='periodic')
    self.assertEqual(grid.step, (0.1,))
    self.assertEqual(grid.domain, ((0, 1.0),))
    self.assertEqual(grid.boundaries, ('periodic',))
    axis, = grid.axes(offset=(0,))
    self.assertAllClose(axis, 0.1 * np.arange(10))

    grid = grids.Grid((10,), domain=[(-2, 2)], boundaries='dirichlet')
    self.assertEqual(grid.step, (2/5,))
    self.assertEqual(grid.domain, ((-2, 2),))
    self.assertEqual(grid.boundaries, ('dirichlet',))
    axis, = grid.axes()
    self.assertAllClose(axis, np.linspace(-2 + 1/5, 2 - 1/5, num=10), atol=1e-6)

    with self.assertRaisesRegex(TypeError, 'cannot provide both'):
      grids.Grid((2,), step=(1.0,), domain=[(0, 2.0)])
    with self.assertRaisesRegex(ValueError, 'length of domain'):
      grids.Grid((2, 3), domain=[(0, 1)])
    with self.assertRaisesRegex(ValueError, 'pairs of numbers'):
      grids.Grid((2,), domain=[(0, 1, 2)])
    with self.assertRaisesRegex(ValueError, 'length of step'):
      grids.Grid((2, 3), step=(1.0,))
    with self.assertRaisesRegex(ValueError, 'invalid boundaries'):
      grids.Grid((2, 3), boundaries='not_valid')

  @mock.patch.object(xla_bridge, 'device_count', new=lambda: 8)
  def test_constructor_device_layout(self):
    shape = (100, 100, 100)
    grid = grids.Grid(shape, device_layout=(1, 2, 4))
    self.assertEqual(grid.device_layout, (1, 2, 4))

    with self.assertRaisesRegex(ValueError, 'length of device_layout'):
      grids.Grid(shape, device_layout=(2, 4))
    with self.assertRaisesRegex(ValueError, 'does not match device_count'):
      grids.Grid(shape, device_layout=(2, 2, 1))
    with self.assertRaisesRegex(ValueError, 'does not divide'):
      grids.Grid(shape, device_layout=(1, 1, 8))

  @parameterized.parameters(
      dict(backend=np,
           shape=(11,),
           initial_offset=(0.0,),
           step=1,
           offset=(0,)),
      dict(backend=np,
           shape=(11,),
           initial_offset=(0.0,),
           step=1,
           offset=(1,)),
      dict(backend=np,
           shape=(11,),
           initial_offset=(0.0,),
           step=1,
           offset=(-1,)),
      dict(backend=np,
           shape=(11,),
           initial_offset=(0.0,),
           step=1,
           offset=(5,)),
      dict(backend=np,
           shape=(11,),
           initial_offset=(0.0,),
           step=1,
           offset=(13,)),
      dict(backend=np,
           shape=(11,),
           initial_offset=(0.0,),
           step=1,
           offset=(31,)),
      dict(backend=np,
           shape=(11, 12, 17),
           initial_offset=(-0.5, -1.0, 1.0),
           step=0.1,
           offset=(-236, 10001, 3)),
      dict(backend=np,
           shape=(121,),
           initial_offset=(-0.5,),
           step=1,
           offset=(31,)),
      dict(backend=np,
           shape=(11, 12, 17),
           initial_offset=(0.5, 0.0, 1.0),
           step=0.1,
           offset=(-236, 10001, 3)),
  )
  def test_shift(self, backend, shape, initial_offset, step, offset):
    """Test that `shift` returns the expected values."""
    grid = grids.Grid(shape, step)
    data = backend.arange(np.prod(shape)).reshape(shape)
    u = grids.AlignedArray(data, initial_offset)
    shifted_u = u
    for axis, o in enumerate(offset):
      shifted_u = grid.shift(shifted_u, o, axis=axis)

    shifted_indices = [(jnp.arange(s) + o) % s for s, o in zip(shape, offset)]
    shifted_mesh = jnp.meshgrid(*shifted_indices, indexing='ij')
    expected_offset = tuple(i + o for i, o in zip(initial_offset, offset))
    expected = grids.AlignedArray(data[tuple(shifted_mesh)], expected_offset)

    self.assertArrayEqual(shifted_u, expected)

  @parameterized.parameters(
      dict(
          grid=grids.Grid((3,), boundaries='periodic'),
          inputs=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
          padding=(0, 0),
          expected=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
      ),
      dict(
          grid=grids.Grid((3,), boundaries='periodic'),
          inputs=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
          padding=(0, 1),
          expected=grids.AlignedArray(np.array([1, 2, 3, 1]), (0,)),
      ),
      dict(
          grid=grids.Grid((3,), boundaries='periodic'),
          inputs=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
          padding=(1, 1),
          expected=grids.AlignedArray(np.array([3, 1, 2, 3, 1]), (-1,)),
      ),
      dict(
          grid=grids.Grid((3,), boundaries='dirichlet'),
          inputs=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
          padding=(1, 1),
          expected=grids.AlignedArray(np.array([0, 1, 2, 3, 0]), (-1,)),
      ),
      dict(
          grid=grids.Grid((3,)),
          inputs=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
          padding=(2, 1),
          pad_kwargs=dict(mode='constant', constant_values=(-1, -2)),
          expected=grids.AlignedArray(np.array([-1, -1, 1, 2, 3, -2]), (-2,)),
      ),
  )
  def test_pad(self, grid, inputs, padding, expected, pad_kwargs=None):
    actual = grid.pad(inputs, padding, axis=0, pad_kwargs=pad_kwargs)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      dict(
          inputs=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
          padding=(0, 0),
          expected=grids.AlignedArray(np.array([1, 2, 3]), (0,)),
      ),
      dict(
          inputs=grids.AlignedArray(np.array([1, 2, 3, 4]), (0,)),
          padding=(1, 1),
          expected=grids.AlignedArray(np.array([2, 3]), (1,)),
      ),
      dict(
          inputs=grids.AlignedArray(np.arange(10), (0,)),
          padding=(2, 3),
          expected=grids.AlignedArray(np.arange(2, 7), (2,)),
      ),
  )
  def test_trim(self, inputs, padding, expected):
    grid = grids.Grid(inputs.data.shape)
    actual = grid.trim(inputs, padding, axis=0)
    self.assertArrayEqual(actual, expected)

  def test_pad_with_devices(self):
    raise unittest.SkipTest("won't pass on CPU until b/144248774 is fixed")
    # TODO(shoyer): add a TPU test

    # pylint: disable=unreachable
    data = grids.AlignedArray(jnp.array([[1, 2, 3]]), offset=(0,))

    grid = grids.Grid((3,), boundaries='periodic', device_layout=(1,))
    expected = grids.AlignedArray(jnp.array([[3, 1, 2, 3, 1]]), offset=(-1,))
    actual = jax.pmap(
        functools.partial(grid.pad, padding=(1, 1), axis=0), axis_name=0)(data)
    self.assertArrayEqual(actual, expected)

    grid = grids.Grid((3,), boundaries='dirichlet', device_layout=(1,))
    expected = grids.AlignedArray(jnp.array([[0, 1, 2, 3, 0]]), offset=(-1,))
    actual = jax.pmap(
        functools.partial(grid.pad, padding=(1, 1), axis=0), axis_name=0)(data)
    self.assertArrayEqual(actual, expected)

  def test_device_permutation(self):
    actual = set(grids._device_permutation([0, 1, 2], shift=+1, axis=0))
    expected = {(0, 1), (1, 2), (2, 0)}
    self.assertEqual(actual, expected)

    actual = set(grids._device_permutation([0, 1, 2], shift=-1, axis=0))
    expected = {(1, 0), (2, 1), (0, 2)}
    self.assertEqual(actual, expected)

    actual = set(grids._device_permutation([0, 1, 2], shift=+1, axis=0,
                                           boundary='dirichlet'))
    expected = {(0, 1), (1, 2)}
    self.assertEqual(actual, expected)

    actual = set(grids._device_permutation([0, 1, 2], shift=-1, axis=0,
                                           boundary='dirichlet'))
    expected = {(1, 0), (2, 1)}
    self.assertEqual(actual, expected)

    actual = set(grids._device_permutation([[0, 1, 2]], shift=0, axis=0))
    expected = {(0, 0), (1, 1), (2, 2)}
    self.assertEqual(actual, expected)

    actual = set(grids._device_permutation([[0, 1, 2]], shift=+1, axis=0))
    expected = {(0, 0), (1, 1), (2, 2)}
    self.assertEqual(actual, expected)

    actual = set(grids._device_permutation([[0, 1, 2]], shift=+1, axis=0,
                                           boundary='dirichlet'))
    expected = set()
    self.assertEqual(actual, expected)

    actual = set(grids._device_permutation([[0, 1, 2], [3, 4, 5]], shift=+1,
                                           axis=1, boundary='dirichlet'))
    expected = {(0, 1), (1, 2), (3, 4), (4, 5)}
    self.assertEqual(actual, expected)


class AlignedArrayTest(test_util.TestCase):

  def test_tree_util(self):
    array = grids.AlignedArray(jnp.arange(3), offset=(0,))
    flat, treedef = jax.tree_flatten(array)
    roundtripped = jax.tree_unflatten(treedef, flat)
    self.assertArrayEqual(array, roundtripped)

  def test_aligned_offset(self):
    data = jnp.arange(3)
    array_at_zero = grids.AlignedArray(data, offset=(0,))
    array_at_one = grids.AlignedArray(data, offset=(1,))

    offset = grids.aligned_offset(array_at_zero, array_at_zero)
    self.assertEqual(offset, (0,))

    with self.assertRaises(grids.AlignmentError):
      grids.aligned_offset(array_at_zero, array_at_one)

  def test_add_sub_correctness(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    offsets = (0.5, 0.5)
    input_array_1 = grids.AlignedArray(values_1, offset=offsets)
    input_array_2 = grids.AlignedArray(values_2, offset=offsets)
    actual_sum = input_array_1 + input_array_2
    actual_sub = input_array_1 - input_array_2
    expected_sum = grids.AlignedArray(values_1 + values_2, offset=offsets)
    expected_sub = grids.AlignedArray(values_1 - values_2, offset=offsets)
    self.assertAllClose(actual_sum, expected_sum, atol=1e-7)
    self.assertAllClose(actual_sub, expected_sub, atol=1e-7)

  def test_add_sub_offset_raise(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    offset_1 = (0.5, 0.5)
    offset_2 = (0.5, 0.0)
    input_array_1 = grids.AlignedArray(values_1, offset=offset_1)
    input_array_2 = grids.AlignedArray(values_2, offset=offset_2)
    with self.assertRaises(grids.AlignmentError):
      _ = input_array_1 + input_array_2
    with self.assertRaises(grids.AlignmentError):
      _ = input_array_1 - input_array_2

  def test_mul_div_correctness(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    scalar = 3.1415
    offsets = (0.5, 0.5)
    input_array_1 = grids.AlignedArray(values_1, offset=offsets)
    input_array_2 = grids.AlignedArray(values_2, offset=offsets)
    actual_mul = input_array_1 * input_array_2
    array_1_times_scalar = input_array_1 * scalar
    expected_1_times_scalar = grids.AlignedArray(values_1 * scalar, offsets)
    actual_div = input_array_1 / 2.5
    expected_div = grids.AlignedArray(values_1 / 2.5, offset=offsets)
    expected_mul = grids.AlignedArray(values_1 * values_2, offset=offsets)
    self.assertAllClose(actual_mul, expected_mul, atol=1e-7)
    self.assertAllClose(
        array_1_times_scalar, expected_1_times_scalar, atol=1e-7)
    self.assertAllClose(actual_div, expected_div, atol=1e-7)

  def test_add_inplace(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    offsets = (0.5, 0.5)
    array = grids.AlignedArray(values_1, offsets)
    array += values_2
    expected = grids.AlignedArray(values_1 + values_2, offsets)
    self.assertAllClose(array, expected, atol=1e-7)

  def test_jit(self):
    u = grids.AlignedArray(jnp.ones([10, 10]), (.5, .5))

    def f(u):
      return u.data < 2.

    self.assertAllClose(f(u), jax.jit(f)(u))

  def test_applied(self):
    u = grids.AlignedArray(jnp.ones([10, 10]), (.5, .5))
    expected = grids.AlignedArray(-jnp.ones([10, 10]), (.5, .5))
    actual = grids.applied(jnp.negative)(u)
    self.assertAllClose(expected, actual)

  def test_indexing(self):
    u = grids.AlignedArray(jnp.arange(10), (.5,))
    expected = grids.AlignedArray(jnp.arange(5), (.5,))
    actual = u[:5]
    self.assertArrayEqual(expected, actual)


if __name__ == '__main__':
  absltest.main()
