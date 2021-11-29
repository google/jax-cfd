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

import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.base import test_util
import numpy as np
import scipy.interpolate as spi


def periodic_grid_variable(data, offset, grid):
  return grids.GridVariable(
      array=grids.GridArray(data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


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
    u = periodic_grid_variable(jnp.ones(shape), jnp.zeros(shape), grid)
    with self.assertRaises(ValueError):
      interpolation.linear(u, offset)

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
    initial_u = periodic_grid_variable(f(initial_mesh), initial_offset, grid)

    final_mesh = grid.mesh(offset=final_offset)
    final_u = interpolation.linear(initial_u, final_offset)

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
    c = periodic_grid_variable(jnp.ones(grid_shape), c_offset, grid)
    with self.assertRaises(grids.InconsistentOffsetError):
      interpolation.upwind(c, u_offset, None)

  @parameterized.named_parameters(
      dict(testcase_name='_2D_positive_velocity',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
           c_offset=(.5, .5),
           u_data=lambda: jnp.ones((10, 10)),
           u_offset=(.5, 1.),
           u_axis=1,
           expected_data=lambda: jnp.arange(10. * 10.).reshape((10, 10))),
      dict(testcase_name='_2D_negative_velocity',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
           c_offset=(.5, .5),
           u_data=lambda: -1. * jnp.ones((10, 10)),
           u_offset=(1., .5),
           u_axis=0,
           expected_data=lambda: jnp.roll(  # pylint: disable=g-long-lambda
               jnp.arange(10. * 10.).reshape((10, 10)), shift=-1, axis=0)),
      dict(testcase_name='_2D_negative_velocity_large_offset',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
           c_offset=(.5, .5),
           u_data=lambda: -1. * jnp.ones((10, 10)),
           u_offset=(2., .5),
           u_axis=0,
           expected_data=lambda: jnp.roll(  # pylint: disable=g-long-lambda
               jnp.arange(10. * 10.).reshape((10, 10)), shift=-2, axis=0)),
      dict(testcase_name='_2D_integer_offset',
           grid_shape=(10, 10),
           grid_step=(1., 1.),
           c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
           c_offset=(0.5, 1),
           u_data=lambda: -1. * jnp.ones((10, 10)),
           u_offset=(0.5, 0),
           u_axis=1,
           expected_data=lambda: jnp.roll(  # pylint: disable=g-long-lambda
               jnp.arange(10. * 10.).reshape((10, 10)), shift=1, axis=1)),
      )
  def testCorrectness(self, grid_shape, grid_step, c_data, c_offset, u_data,
                      u_offset, u_axis, expected_data):
    grid = grids.Grid(grid_shape, grid_step)
    initial_c = periodic_grid_variable(c_data(), c_offset, grid)
    u = periodic_grid_variable(u_data(), u_offset, grid)
    v = tuple(
        u if axis == u_axis else None for axis, _ in enumerate(u_offset)
    )
    final_c = interpolation.upwind(initial_c, u_offset, v)
    self.assertAllClose(expected_data(), final_c.data)
    self.assertAllClose(u_offset, final_c.offset)


class LaxWendroffInterpolationTest(test_util.TestCase):

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
    c = periodic_grid_variable(jnp.ones(grid_shape), c_offset, grid)
    with self.assertRaises(grids.InconsistentOffsetError):
      interpolation.lax_wendroff(c, u_offset, v=None, dt=0.)

  @parameterized.named_parameters(
      dict(
          testcase_name='_2D_positive_velocity_courant=1',
          grid_shape=(10, 10),
          grid_step=(1., 1.),
          c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
          c_offset=(.5, .5),
          u_data=lambda: jnp.ones((10, 10)),
          u_offset=(.5, 1.),
          u_axis=1,
          dt=1.,  # courant = u * dt / grid_step = 1
          expected_data=lambda: jnp.arange(10. * 10.).reshape((10, 10))),
      dict(
          testcase_name='_2D_negative_velocity_courant=-1',
          grid_shape=(10, 10),
          grid_step=(1., 1.),
          c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
          c_offset=(.5, .5),
          u_data=lambda: -1. * jnp.ones((10, 10)),
          u_offset=(.5, 1.),
          u_axis=1,
          dt=1.,  # courant = u * dt / grid_step = -1
          expected_data=lambda: jnp.roll(  # pylint: disable=g-long-lambda
              jnp.arange(10. * 10.).reshape((10, 10)), shift=-1, axis=1)),
      dict(
          testcase_name='_2D_positive_velocity_courant=0',
          grid_shape=(10, 10),
          grid_step=(1., 1.),
          c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
          c_offset=(.5, .5),
          u_data=lambda: jnp.ones((10, 10)),
          u_offset=(.5, 1.),
          u_axis=1,
          dt=0.,  # courant = u * dt / grid_step = 0
          # for courant = 1, result is the average of cell and upwind
          expected_data=lambda: 0.5 * (  # pylint: disable=g-long-lambda
              jnp.arange(10. * 10.).reshape((10, 10)) + jnp.roll(
                  jnp.arange(10. * 10.).reshape((10, 10)), shift=-1, axis=1))),
      dict(
          testcase_name='_2D_negative_velocity_courant=0',
          grid_shape=(10, 10),
          grid_step=(1., 1.),
          c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
          c_offset=(.5, .5),
          u_data=lambda: -1. * jnp.ones((10, 10)),
          u_offset=(.5, 1.),
          u_axis=1,
          dt=0.,  # courant = u * dt / grid_step = 0
          # for courant = 1, result is the average of cell and upwind
          expected_data=lambda: 0.5 * (  # pylint: disable=g-long-lambda
              jnp.arange(10. * 10.).reshape((10, 10)) + jnp.roll(
                  jnp.arange(10. * 10.).reshape((10, 10)), shift=-1, axis=1))),
  )
  def testCorrectness(self, grid_shape, grid_step, c_data, c_offset, u_data,
                      u_offset, u_axis, dt, expected_data):
    grid = grids.Grid(grid_shape, grid_step)
    initial_c = periodic_grid_variable(c_data(), c_offset, grid)
    u = periodic_grid_variable(u_data(), u_offset, grid)
    v = tuple(
        u if axis == u_axis else None for axis, _ in enumerate(u_offset)
    )
    final_c = interpolation.lax_wendroff(initial_c, u_offset, v, dt)
    self.assertAllClose(expected_data(), final_c.data)
    self.assertAllClose(u_offset, final_c.offset)


class ApplyTvdLimiterTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_case_where_lax_wendroff_is_same_as_upwind',
          interpolation_fn=interpolation.lax_wendroff,
          limiter=interpolation.van_leer_limiter,
          grid_shape=(10, 10),
          grid_step=(1., 1.),
          c_data=lambda: jnp.arange(10. * 10.).reshape((10, 10)),
          c_offset=(.5, .5),
          u_data=lambda: jnp.ones((10, 10)),
          u_offset=(.5, 1.),
          u_axis=1,
          dt=1.,
          expected_fn=interpolation.upwind),
  )
  def testCorrectness(self, interpolation_fn, limiter, grid_shape, grid_step,
                      c_data, c_offset, u_data, u_offset, u_axis, dt,
                      expected_fn):
    c_interpolation_fn = interpolation.apply_tvd_limiter(
        interpolation_fn, limiter)
    grid = grids.Grid(grid_shape, grid_step)
    initial_c = periodic_grid_variable(c_data(), c_offset, grid)
    u = periodic_grid_variable(u_data(), u_offset, grid)
    v = tuple(
        u if axis == u_axis else None for axis, _ in enumerate(u_offset)
    )
    final_c = c_interpolation_fn(initial_c, u_offset, v, dt)
    expected = expected_fn(initial_c, u_offset, v, dt)
    self.assertAllClose(expected, final_c)
    self.assertAllClose(u_offset, final_c.offset)


class PointInterpolationTest(test_util.TestCase):

  def test_eval_of_2d_function(self):
    grid_shape = (50, 111)
    offset = (0.5, 0.8)

    grid = grids.Grid(grid_shape, domain=((0., jnp.pi),) * 2)
    xy_grid_pts = jnp.stack(grid.mesh(offset), axis=-1)

    func = lambda xy: jnp.sin(xy[..., 0]) * xy[..., 1]

    c = grids.GridArray(data=func(xy_grid_pts), offset=offset, grid=grid)

    vec_interp = jax.vmap(
        interpolation.point_interpolation, in_axes=(0, None), out_axes=0)

    # At the grid points, accuracy should be excellent...almost perfect up to
    # epsilon.
    xy_grid_pts = jnp.reshape(xy_grid_pts, (-1, 2))  # Reshape for vmap.
    self.assertAllClose(
        vec_interp(xy_grid_pts, c), func(xy_grid_pts), atol=1e-6)

    # At random points, tol is guided by standard first order method heuristics.
    atol = 1 / min(*grid_shape)
    xy_random = np.random.RandomState(0).rand(100, 2) * np.pi
    self.assertAllClose(
        vec_interp(xy_random, c), func(xy_random), atol=atol)

  def test_order_equals_0_is_piecewise_constant(self):
    grid_shape = (3,)
    offset = (0.5,)

    grid = grids.Grid(grid_shape, domain=((0., 1.),))
    x_grid_pts, = grid.mesh(offset=offset)

    func = lambda x: 2 * x**2

    c = grids.GridArray(data=func(x_grid_pts), offset=offset, grid=grid)

    def _nearby_points(value):
      eps = grid.step[0] / 3
      return [value - eps, value, value + eps]

    def _interp(x):
      return interpolation.point_interpolation(x, c, order=0)

    for x in x_grid_pts:
      for near_x in _nearby_points(x):
        np.testing.assert_allclose(func(x), _interp(near_x))

  def test_mode_and_cval_args_are_used(self):
    # Just test that the mode arg is used, in the simplest way.
    # We don't have to check the correctnesss of these args, since
    # jax.scipy.ndimage.map_coordinates is tested separately.
    grid_shape = (3,)
    offset = (0.5,)

    grid = grids.Grid(grid_shape, domain=((0., 1.),))
    x_grid_pts, = grid.mesh(offset=offset)

    c = grids.GridArray(
        # Just use random points. Won't affect anything.
        data=x_grid_pts * 0.1, offset=offset, grid=grid)

    # Outside the domain, the passing of mode='constant' and cval=1234 results
    # in this value being used.
    outside_domain_point = 10.
    self.assertAllClose(
        1234,
        interpolation.point_interpolation(
            outside_domain_point, c, mode='constant', cval=1234))


if __name__ == '__main__':
  absltest.main()
