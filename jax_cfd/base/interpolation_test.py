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

"""Tests for jax_cfd.interpolation."""

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.base import test_util
import numpy as np
import scipy.interpolate as spi


class LinearInterpolationTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_offset_too_short',
           shape=(10, 10, 10),
           step=(1., 1., 1.),
           offset=(2., 3.)),
      dict(testcase_name='_offset_too_long',
           shape=(10, 10),
           step=(1., 1.),
           offset=(2., 3., 4.))
      )
  def testRaisesForInvalidOffset(self, shape, step, offset):
    """Test that incompatible offsets raise an exception."""
    grid = grids.Grid(shape, step)
    u = grids.AlignedArray(jnp.ones(shape), jnp.zeros(shape))
    with self.assertRaises(ValueError):
      interpolation.linear(u, offset, grid)

  @parameterized.named_parameters(
      dict(testcase_name='_1D',
           shape=(100,),
           step=(.1,),
           f=(lambda x: np.random.RandomState(123).randn(*x[0].shape)),
           initial_offset=(-.5,),
           final_offset=(.5,)),
      dict(testcase_name='_2D',
           shape=(100, 100),
           step=(1., 1.),
           f=(lambda xy: np.random.RandomState(231).randn(*xy[0].shape)),
           initial_offset=(1., 0.),
           final_offset=(0., 0.)),
      dict(testcase_name='_3D',
           shape=(100, 100, 100),
           step=(.3, .4, .5),
           f=(lambda xyz: np.random.RandomState(312).randn(*xyz[0].shape)),
           initial_offset=(0., 1., 0.),
           final_offset=(.5, .5, .5)),
      )
  def testEquivalenceWithScipy(
      self, shape, step, f, initial_offset, final_offset):
    """Tests that interpolation is close to results of `scipy.interpolate`."""
    grid = grids.Grid(shape, step)

    initial_mesh = grid.mesh(offset=initial_offset)
    initial_axes = grid.axes(offset=initial_offset)
    initial_u = grids.AlignedArray(f(initial_mesh), initial_offset)

    final_mesh = grid.mesh(offset=final_offset)
    final_u = interpolation.linear(initial_u, final_offset, grid)

    expected_data = spi.interpn(initial_axes,
                                initial_u.data,
                                jnp.stack(final_mesh, -1),
                                method='linear',
                                bounds_error=False)

    # Scipy does not support periodic boundaries so we compare only valid
    # values.
    valid = np.where(~jnp.isnan(expected_data))
    self.assertAllClose(
        expected_data[valid], final_u.data[valid], atol=1e-4)
    self.assertAllClose(final_offset, final_u.offset)


class UpwindInterpolationTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_2D',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_offset=(1., .5),
           u_offset=(.5, 1.)),
      dict(testcase_name='_3D',
           grid_shape=(10, 10, 10),
           grid_step=(1., 1., 1.),
           c_offset=(.5, 1., .5),
           u_offset=(.5, .5, 1.)),
      )
  def testRaisesForInvalidOffset(
      self, grid_shape, grid_step, c_offset, u_offset):
    """Test that incompatible offsets raise an exception."""
    grid = grids.Grid(grid_shape, grid_step)
    c = grids.AlignedArray(jnp.ones(grid_shape), offset=c_offset)
    with self.assertRaises(grids.AlignmentError):
      interpolation.upwind(c, u_offset, grid, None)

  @parameterized.named_parameters(
      dict(testcase_name='_2D_positive',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
           c_offset=(.5, .5),
           u_data=lambda: jnp.ones((10, 10)),
           u_offset=(.5, 1.),
           u_axis=1,
           expected_data=lambda: jnp.arange(10. * 10.).reshape((10, 10))),
      dict(testcase_name='_2D_negative',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
           c_offset=(.5, .5),
           u_data=lambda: -1. * jnp.ones((10, 10)),
           u_offset=(1., .5),
           u_axis=0,
           expected_data=lambda: jnp.roll(  # pylint: disable=g-long-lambda
               jnp.arange(10. * 10.).reshape((10, 10)), shift=-1, axis=0)),
      dict(testcase_name='_2D_negative_large_offset',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
           c_offset=(.5, .5),
           u_data=lambda: -1. * jnp.ones((10, 10)),
           u_offset=(2., .5),
           u_axis=0,
           expected_data=lambda: jnp.roll(  # pylint: disable=g-long-lambda
               jnp.arange(10. * 10.).reshape((10, 10)), shift=-2, axis=0)),
      )
  def testCorrectness(self, grid_shape, grid_step, c_data, c_offset, u_data,
                      u_offset, u_axis, expected_data):
    grid = grids.Grid(grid_shape, grid_step)
    initial_c = grids.AlignedArray(c_data(), c_offset)
    u = grids.AlignedArray(u_data(), u_offset)
    v = tuple(
        u if axis == u_axis else None for axis, _ in enumerate(u_offset)
    )
    final_c = interpolation.upwind(initial_c, u_offset, grid, v)
    self.assertAllClose(expected_data(), final_c.data)
    self.assertAllClose(u_offset, final_c.offset)


if __name__ == '__main__':
  absltest.main()
