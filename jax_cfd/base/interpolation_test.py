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


class Interp1DTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='FillWithExtrapolate', fill_value='extrapolate'),
      dict(testcase_name='FillWithScalar', fill_value=123),
  )
  def test_same_as_scipy_on_scalars_and_check_grads(self, fill_value):
    rng = np.random.RandomState(45)

    n = 10
    y = rng.randn(n)

    x_low = 5.
    x_high = 9.
    x = np.linspace(x_low, x_high, num=n)

    sp_func = spi.interp1d(
        x,
        y,
        kind='linear',
        fill_value=fill_value,
        bounds_error=False)
    cfd_func = interpolation.interp1d(x, y, fill_value=fill_value)

    # Check x_new at the table definition points `x`, points outside, and points
    # in between.
    x_to_check = np.concatenate(([x_low - 1], x, x * 1.051, [x_high + 1]))

    for x_new in x_to_check:
      sp_y_new = sp_func(x_new).astype(np.float32)
      cfd_y_new = cfd_func(x_new)
      self.assertAllClose(sp_y_new, cfd_y_new, rtol=1e-5, atol=1e-6)

      grad_cfd_y_new = jax.grad(cfd_func)(x_new)

      # Gradients should be nonzero except when outside the range of `x` and
      # filling with a constant.
      # Why check this? Because, some indexing methods result in gradients == 0
      # at the interpolation table points.
      if fill_value == 'extrapolate' or x_low <= x_new <= x_high:
        self.assertLess(0, np.abs(grad_cfd_y_new))
      else:
        self.assertTrue(np.isfinite(grad_cfd_y_new))

  @parameterized.named_parameters(
      dict(
          testcase_name='TableIs1D_XNewIs1D_Axis0_FillWithScalar',
          table_ndim=1,
          x_new_ndim=1,
          axis=0,
          fill_value=12345,
      ),
      dict(
          testcase_name='TableIs1D_XNewIs1D_Axis0_FillWithExtrapolate',
          table_ndim=1,
          x_new_ndim=1,
          axis=0,
          fill_value='extrapolate',
      ),
      dict(
          testcase_name='TableIs2D_XNewIs1D_Axis0_FillWithExtrapolate',
          table_ndim=2,
          x_new_ndim=1,
          axis=0,
          fill_value='extrapolate',
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axis1_FillWithScalar',
          table_ndim=3,
          x_new_ndim=2,
          axis=1,
          fill_value=12345,
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn1_FillWithScalar',
          table_ndim=3,
          x_new_ndim=2,
          axis=-1,
          fill_value=12345,
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn1_FillWithExtrapolate',
          table_ndim=3,
          x_new_ndim=2,
          axis=-1,
          fill_value='extrapolate',
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn3_FillWithScalar',
          table_ndim=3,
          x_new_ndim=2,
          axis=-3,
          fill_value=1234,
      ),
      dict(
          testcase_name='TableIs3D_XNewIs2D_Axisn3_FillWithConstantExtension',
          table_ndim=3,
          x_new_ndim=2,
          axis=-3,
          fill_value='constant_extension',
      ),
  )
  def test_same_as_scipy_on_arrays(
      self,
      table_ndim,
      x_new_ndim,
      axis,
      fill_value,
  ):
    """Test results are the same as scipy.interpolate.interp1d."""
    rng = np.random.RandomState(45)

    # Arbitrary shape that ensures all dims are different to prevent
    # broadcasting from hiding bugs.
    y_shape = tuple(range(5, 5 + table_ndim))
    y = rng.randn(*y_shape)
    n = y_shape[axis]

    x_low = 5
    x_high = 9
    x = np.linspace(x_low, x_high, num=n)**2  # Arbitrary non-linearly spaced x

    # Scipy doesn't have 'constant_extension', so treat it special.
    # Here use np.nan as the fill value, which will be easy to spot if we handle
    # it wrong.
    if fill_value == 'constant_extension':
      sp_fill_value = np.nan
    else:
      sp_fill_value = fill_value

    sp_func = spi.interp1d(
        x,
        y,
        kind='linear',
        axis=axis,
        fill_value=sp_fill_value,
        bounds_error=False)
    cfd_func = interpolation.interp1d(x, y, axis=axis, fill_value=fill_value)

    # Make n_x_new > n, so we can selectively fill values as below.
    n_x_new = max(2 * n, 20)

    # Make x_new over the same range as x.
    x_new_shape = tuple(range(2, x_new_ndim + 1)) + (n_x_new,)
    x_new = (x_low + rng.rand(*x_new_shape) * (x_high - x_low))**2

    x_new[..., 0] = np.min(x) - 1  # Out of bounds low
    x_new[..., -1] = np.max(x) + 1  # Out of bounds high
    x_new[..., 1:n + 1] = x  # All the grid points

    # Scipy doesn't have the 'constant_extension' feature, but
    # constant_extension is achieved by clipping the input.
    if fill_value == 'constant_extension':
      sp_x_new = np.clip(x_new, np.min(x), np.max(x))
    else:
      sp_x_new = x_new

    sp_y_new = sp_func(sp_x_new).astype(np.float32)
    cfd_y_new = cfd_func(x_new)
    self.assertAllClose(sp_y_new, cfd_y_new, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
