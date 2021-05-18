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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np


def _trim_boundary(array):
  return array[(slice(1, -1),) * array.ndim]


class FiniteDifferenceTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_central_difference_periodic',
           method=fd.central_difference,
           grid_type=grids.Grid,
           shape=(3, 3, 3),
           step=(1., 1., 1.),
           expected_offset=0,
           negative_shift=-1,
           positive_shift=1),
      dict(testcase_name='_backward_difference_periodic',
           method=fd.backward_difference,
           grid_type=grids.Grid,
           shape=(2, 3, 4),
           step=(.1, .3, 1.),
           expected_offset=-0.5,
           negative_shift=-1,
           positive_shift=0),
      dict(testcase_name='_forward_difference_periodic',
           method=fd.forward_difference,
           grid_type=grids.Grid,
           shape=(23, 32, 1),
           step=(.01, 2., .1),
           expected_offset=+0.5,
           negative_shift=0,
           positive_shift=1),
      )
  def test_finite_difference_indexing(
      self, method, grid_type, shape, step, expected_offset, negative_shift,
      positive_shift):
    """Tests finite difference code using explicit indices."""
    grid = grid_type(shape, step)
    u = grids.AlignedArray(
        jnp.arange(np.prod(shape)).reshape(shape), (0, 0, 0))
    actual_x, actual_y, actual_z = method(u, grid)

    x, y, z = jnp.meshgrid(*[jnp.arange(s) for s in shape], indexing='ij')

    diff_x = (u.data[jnp.roll(x, -positive_shift, axis=0), y, z] -
              u.data[jnp.roll(x, -negative_shift, axis=0), y, z])
    expected_data_x = diff_x / step[0] / (positive_shift - negative_shift)
    expected_x = grids.AlignedArray(expected_data_x, (expected_offset, 0, 0))

    diff_y = (u.data[x, jnp.roll(y, -positive_shift, axis=1), z] -
              u.data[x, jnp.roll(y, -negative_shift, axis=1), z])
    expected_data_y = diff_y / step[1] / (positive_shift - negative_shift)
    expected_y = grids.AlignedArray(expected_data_y, (0, expected_offset, 0))

    diff_z = (u.data[x, y, jnp.roll(z, -positive_shift, axis=2)] -
              u.data[x, y, jnp.roll(z, -negative_shift, axis=2)])
    expected_data_z = diff_z / step[2] / (positive_shift - negative_shift)
    expected_z = grids.AlignedArray(expected_data_z, (0, 0, expected_offset))

    self.assertArrayEqual(expected_x, actual_x)
    self.assertArrayEqual(expected_y, actual_y)
    self.assertArrayEqual(expected_z, actual_z)

  @parameterized.named_parameters(
      dict(testcase_name='_central_difference_periodic',
           method=fd.central_difference,
           grid_type=grids.Grid,
           shape=(100, 100, 100),
           offset=(0, 0, 0),
           f=lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.sin(z),
           gradf=(lambda x, y, z: -jnp.sin(x) * jnp.cos(y) * jnp.sin(z),
                  lambda x, y, z: -jnp.cos(x) * jnp.sin(y) * jnp.sin(z),
                  lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.cos(z)),
           atol=1e-3),
      dict(testcase_name='_backward_difference_periodic',
           method=fd.backward_difference,
           grid_type=grids.Grid,
           shape=(100, 100, 100),
           offset=(0, 0, 0),
           f=lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.sin(z),
           gradf=(lambda x, y, z: -jnp.sin(x) * jnp.cos(y) * jnp.sin(z),
                  lambda x, y, z: -jnp.cos(x) * jnp.sin(y) * jnp.sin(z),
                  lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.cos(z)),
           atol=5e-2),
      dict(testcase_name='_forward_difference_periodic',
           method=fd.forward_difference,
           grid_type=grids.Grid,
           shape=(200, 200, 200),
           offset=(0, 0, 0),
           f=lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.sin(z),
           gradf=(lambda x, y, z: -jnp.sin(x) * jnp.cos(y) * jnp.sin(z),
                  lambda x, y, z: -jnp.cos(x) * jnp.sin(y) * jnp.sin(z),
                  lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.cos(z)),
           atol=5e-2),
      )
  def test_finite_difference_analytic(
      self, method, grid_type, shape, offset, f, gradf, atol):
    """Tests finite difference code comparing to the analytice solution."""
    step = tuple([2. * jnp.pi / s for s in shape])
    grid = grid_type(shape, step)
    mesh = grid.mesh()
    u = grids.AlignedArray(f(*mesh), offset)
    expected_grad = jnp.stack([df(*mesh) for df in gradf])
    actual_grad = [array.data for array in method(u, grid)]
    self.assertAllClose(expected_grad, actual_grad, atol=atol)

  @parameterized.named_parameters(
      dict(testcase_name='_2D_constant',
           grid_type=grids.Grid,
           shape=(20, 20),
           f=lambda x, y: np.ones_like(x),
           g=lambda x, y: np.zeros_like(x),
           atol=1e-3),
      dict(testcase_name='_2D_quadratic',
           grid_type=grids.Grid,
           shape=(21, 21),
           f=lambda x, y: x * (x - 1.) + y * (y - 1.),
           g=lambda x, y: 4 * np.ones_like(x),
           atol=1e-3),
      dict(testcase_name='_3D_quadratic',
           grid_type=grids.Grid,
           shape=(13, 13, 13),
           f=lambda x, y, z: x * (x - 1.) + y * (y - 1.) + z * (z - 1.),
           g=lambda x, y, z: 6 * np.ones_like(x),
           atol=1e-3),
      )
  def test_laplacian(self, grid_type, shape, f, g, atol):
    step = tuple([1. / s for s in shape])
    grid = grid_type(shape, step)
    offset = (0,) * len(shape)
    mesh = grid.mesh(offset)
    u = grids.AlignedArray(f(*mesh), offset)
    expected_laplacian = _trim_boundary(grids.AlignedArray(g(*mesh), offset))
    actual_laplacian = _trim_boundary(fd.laplacian(u, grid))
    self.assertAllClose(expected_laplacian, actual_laplacian, atol=atol)

  @parameterized.named_parameters(
      dict(testcase_name='_2D_constant',
           grid_type=grids.Grid,
           shape=(20, 20),
           offsets=((0.5, 0), (0, 0.5)),
           f=lambda x, y: (np.ones_like(x), np.ones_like(y)),
           g=lambda x, y: jnp.zeros_like(x),
           atol=1e-3),
      dict(testcase_name='_2D_quadratic',
           grid_type=grids.Grid,
           shape=(21, 21),
           offsets=((0.5, 0), (0, 0.5)),
           f=lambda x, y: (x * (x - 1.), y * (y - 1.)),
           g=lambda x, y: 2 * x + 2 * y - 2,
           atol=0.1),
      dict(testcase_name='_3D_quadratic',
           grid_type=grids.Grid,
           shape=(13, 13, 13),
           offsets=((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),
           f=lambda x, y, z: (x * (x - 1.), y * (y - 1.), z * (z - 1.)),
           g=lambda x, y, z: 2 * x + 2 * y + 2 * z - 3,
           atol=0.25),
      )
  def test_divergence(self, grid_type, shape, offsets, f, g, atol):
    step = tuple([1. / s for s in shape])
    grid = grid_type(shape, step)
    v = [grids.AlignedArray(f(*grid.mesh(offset))[axis], offset)
         for axis, offset in enumerate(offsets)]
    expected_divergence = _trim_boundary(
        grids.AlignedArray(g(*grid.mesh()), (0,) * grid.ndim))
    actual_divergence = _trim_boundary(fd.divergence(v, grid))
    self.assertAllClose(expected_divergence, actual_divergence, atol=atol)

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      dict(
          testcase_name='_2D_constant',
          shape=(20, 20),
          f=lambda x, y: (np.ones_like(x), np.ones_like(y)),
          g=lambda x, y: np.array([[jnp.zeros_like(x)] * 2] * 2),
          atol=0),
      dict(
          testcase_name='_2D_quadratic',
          shape=(21, 21),
          f=lambda x, y: (x * (y - 1.), y * (x - 2.)),
          g=lambda x, y: np.array([[y - 1., y], [x, x - 2.]]),
          atol=2e-6),
      dict(
          testcase_name='_2D_quartic',
          shape=(21, 21),
          f=lambda x, y: (x**2 * y**2, (x - 1.)**3 * (y - 2.)),
          g=lambda x, y: np.array([[2 * x * y**2, 3 * (x - 1.)**2 *
                                    (y - 2.)], [2 * x**2 * y, (x - 1.)**3]]),
          atol=1e-2),
      dict(
          testcase_name='_3D_quadratic',
          shape=(13, 13, 13),
          f=lambda x, y, z: (x * (y - 1.), y * (z - 2.), z * (x - 3.)),
          g=lambda x, y, z: np.array([[y - 1., jnp.zeros_like(x), z],
                                      [x, z - 2., jnp.zeros_like(x)],
                                      [jnp.zeros_like(x), y, x - 3.]]),
          atol=4e-6),
  )
  # pylint: enable=g-long-lambda
  def test_cell_centered_gradient(self, shape, f, g, atol):
    step = tuple([1. / s for s in shape])
    grid = grids.Grid(shape, step)
    offsets = grid.cell_faces
    v = [
        grids.AlignedArray(f(*grid.mesh(offset))[axis], offset)
        for axis, offset in enumerate(offsets)
    ]
    expected_gradient = g(*grid.mesh())
    actual_gradient = fd.gradient_tensor(v, grid)
    for i in range(grid.ndim):
      for j in range(len(v)):
        print('i and j are', i, j)
        expected = _trim_boundary(expected_gradient[i, j])
        actual = _trim_boundary(actual_gradient[i, j])
        self.assertAllClose(expected, actual.data, atol=atol)

  @parameterized.named_parameters(
      # https://en.wikipedia.org/wiki/Curl_(mathematics)#Examples
      dict(testcase_name='_wikipedia_example_1',
           grid_type=grids.Grid,
           shape=(20, 20),
           offsets=((0.5, 0), (0, 0.5)),
           f=lambda x, y: (y, -x),
           g=lambda x, y: -2 * np.ones_like(x),
           atol=1e-3),
      dict(testcase_name='_wikipedia_example_2',
           grid_type=grids.Grid,
           shape=(21, 21),
           offsets=((0.5, 0), (0, 0.5)),
           f=lambda x, y: (np.ones_like(x), -x**2),
           g=lambda x, y: -2 * x,
           atol=1e-3),
      )
  def test_curl_2d(self, grid_type, shape, offsets, f, g, atol):
    step = tuple([1. / s for s in shape])
    grid = grid_type(shape, step)
    v = [grids.AlignedArray(f(*grid.mesh(offset))[axis], offset)
         for axis, offset in enumerate(offsets)]
    result_offset = (0.5, 0.5)
    expected_curl = _trim_boundary(
        grids.AlignedArray(g(*grid.mesh(result_offset)), result_offset))
    actual_curl = _trim_boundary(fd.curl_2d(v, grid))
    self.assertAllClose(expected_curl, actual_curl, atol=atol)

  @parameterized.named_parameters(
      # https://www.web-formulas.com/Math_Formulas/Linear_Algebra_Curl_of_a_Vector_Field.aspx
      dict(testcase_name='_web_formulas_example_3',
           grid_type=grids.Grid,
           shape=(13, 13, 13),
           offsets=((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),
           expected_offsets=((0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)),
           f=lambda x, y, z: (x + y + z, x - y - z, x**2 + y**2 + z**2),
           g=lambda x, y, z: (2 * y + 1, 1 - 2 * x, np.zeros_like(x)),
           atol=1e-3),
      )
  def test_curl_3d(
      self, grid_type, shape, offsets, expected_offsets, f, g, atol):
    step = tuple([1. / s for s in shape])
    grid = grid_type(shape, step)
    v = [grids.AlignedArray(f(*grid.mesh(offset))[axis], offset)
         for axis, offset in enumerate(offsets)]
    expected_curl = [
        _trim_boundary(grids.AlignedArray(g(*grid.mesh(offset))[axis], offset))
        for axis, offset in enumerate(expected_offsets)]
    actual_curl = list(map(_trim_boundary, fd.curl_3d(v, grid)))
    self.assertEqual(len(actual_curl), 3)
    self.assertAllClose(expected_curl[0], actual_curl[0], atol=atol)
    self.assertAllClose(expected_curl[1], actual_curl[1], atol=atol)
    self.assertAllClose(expected_curl[2], actual_curl[2], atol=atol)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
