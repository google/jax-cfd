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

"""Tests for jax_cfd.base.resize."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences
from jax_cfd.base import grids
from jax_cfd.base import initial_conditions
from jax_cfd.base import resize
from jax_cfd.base import test_util
import numpy as np

BCType = boundaries.BCType


def periodic_grid_variable(data, offset, grid):
  return grids.GridVariable(
      array=grids.GridArray(data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


class ResizeTest(test_util.TestCase):

  @parameterized.parameters(
      dict(u=np.array([[0, 1, 2, 3],
                       [4, 5, 6, 7],
                       [8, 9, 10, 11],
                       [12, 13, 14, 15]]),
           direction=0,
           factor=2,
           expected=np.array([[4.5, 6.5],
                              [12.5, 14.5]])),
      dict(u=np.array([[0, 1, 2, 3],
                       [4, 5, 6, 7],
                       [8, 9, 10, 11],
                       [12, 13, 14, 15]]),
           direction=1,
           factor=2,
           expected=np.array([[3., 5.],
                              [11., 13.]])),
  )
  def testDownsampleVelocityComponent(self, u, direction, factor, expected):
    """Test `downsample_array` produces the expected results."""
    actual = resize.downsample_staggered_velocity_component(
        u, direction, factor)
    self.assertAllClose(expected, actual)

  def testDownsampleVelocity(self):
    source_grid = grids.Grid((4, 4), domain=[(0, 1), (0, 1)])
    destination_grid = grids.Grid((2, 2), domain=[(0, 1), (0, 1)])
    u = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])
    expected = (np.array([[4.5, 6.5],
                          [12.5, 14.5]]), np.array([[3., 5.], [11., 13.]]))

    with self.subTest('ArrayField'):
      velocity = (u, u)
      actual = resize.downsample_staggered_velocity(source_grid,
                                                    destination_grid, velocity)
      self.assertAllClose(expected, actual)

    with self.subTest('GridArrayVector'):
      velocity = (grids.GridArray(u, offset=(1, 0), grid=source_grid),
                  grids.GridArray(u, offset=(0, 1), grid=source_grid))
      actual = resize.downsample_staggered_velocity(source_grid,
                                                    destination_grid, velocity)
      expected_combined = (
          grids.GridArray(expected[0], offset=(1, 0), grid=destination_grid),
          grids.GridArray(expected[1], offset=(0, 1), grid=destination_grid))
      self.assertAllClose(expected_combined[0], actual[0])
      self.assertAllClose(expected_combined[1], actual[1])

    with self.subTest('GridArrayVector: Inconsistent Grids'):
      with self.assertRaisesRegex(grids.InconsistentGridError,
                                  'source_grid for downsampling'):
        different_grid = grids.Grid((4, 4), domain=[(-2, 2), (0, 1)])
        velocity = (grids.GridArray(u, offset=(1, 0), grid=different_grid),
                    grids.GridArray(u, offset=(0, 1), grid=different_grid))
        resize.downsample_staggered_velocity(source_grid,
                                             destination_grid, velocity)

    with self.subTest('GridVariableVector'):
      velocity = (periodic_grid_variable(u, (1, 0), source_grid),
                  periodic_grid_variable(u, (0, 1), source_grid))
      actual = resize.downsample_staggered_velocity(source_grid,
                                                    destination_grid, velocity)
      expected_combined = (
          periodic_grid_variable(expected[0], (1, 0), destination_grid),
          periodic_grid_variable(expected[1], (0, 1), destination_grid),
      )
      self.assertAllClose(expected_combined[0], actual[0])
      self.assertAllClose(expected_combined[1], actual[1])

    with self.subTest('GridVariableVector: Inconsistent Grids'):
      with self.assertRaisesRegex(grids.InconsistentGridError,
                                  'source_grid for downsampling'):
        different_grid = grids.Grid((4, 4), domain=[(-2, 2), (0, 1)])
        velocity = (
            periodic_grid_variable(u, (1, 0), different_grid),
            periodic_grid_variable(u, (0, 1), different_grid))
        resize.downsample_staggered_velocity(source_grid,
                                             destination_grid, velocity)

  def testDownsampleFourierVorticity(self):

    with self.subTest('Space2D'):
      domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
      fine = grids.Grid(((256, 256)), domain=domain)
      medium = grids.Grid(((128, 128)), domain=domain)
      coarse = grids.Grid(((64, 64)), domain=domain)

      v0 = initial_conditions.filtered_velocity_field(
          jax.random.PRNGKey(42), fine, maximum_velocity=7, peak_wavenumber=1)
      fine_signal = finite_differences.curl_2d(v0).data

      # test that fine -> medium -> coarse == fine -> coarse
      fine_signal_hat = jnp.fft.rfftn(fine_signal)
      fine_to_medium = resize.downsample_spectral(
          None, medium, fine_signal_hat)
      medium_to_coarse = resize.downsample_spectral(
          None, coarse, fine_to_medium)
      fine_to_coarse = resize.downsample_spectral(
          None, coarse, fine_signal_hat)
      self.assertAllClose(fine_to_coarse, medium_to_coarse)

      # test that grid -> grid does nothing
      self.assertAllClose(
          fine_signal_hat,
          resize.downsample_spectral(None, fine, fine_signal_hat))

      self.assertAllClose(
          fine_to_medium,
          resize.downsample_spectral(None, medium, fine_to_medium))

    with self.subTest('DownsampleWavenumbers'):
      fine = grids.Grid((256, 256), domain=((0, 2 * jnp.pi),) * 2)
      coarse = grids.Grid((64, 64), domain=((0, 2 * jnp.pi),) * 2)

      kx_fine, ky_fine = fine.rfft_mesh()
      kx_coarse, ky_coarse = coarse.rfft_mesh()

      # (256/64)^2 = 16
      kx_down = 16 * resize.downsample_spectral(None, coarse, kx_fine)
      ky_down = 16 * resize.downsample_spectral(None, coarse, ky_fine)

      self.assertArrayEqual(kx_down, kx_coarse)
      self.assertArrayEqual(ky_down, ky_coarse)


class ResizeTopHatTest(test_util.TestCase):

  @parameterized.parameters(
      # Periodic BC
      dict(
          input_data=np.array([0, 1, 2, 3]),
          input_offset=(.5,),
          expected_data=np.array([0.5, 2.5]),
          filter_size=None,
      ),
      dict(
          input_data=np.array([-1, 1, 2, 3]),
          input_offset=(1.,),
          expected_data=np.array([.75, 1.75]),
          filter_size=None,
      ),
      dict(
          input_data=np.array([-1, 1, 2, 3]),
          input_offset=(0.,),
          expected_data=np.array([.5, 2.,]),
          filter_size=None,
      ),
      dict(
          input_data=np.array([0, 1, 2, 3, 4, 5]),
          input_offset=(1.,),
          expected_data=np.array([1.75, 3., 2.75]),
          filter_size=4,
      ),
      dict(
          input_data=np.array([0, 1, 2, 3, 4, 5]),
          input_offset=(.5,),
          expected_data=np.array([2., 2.5, 3.]),
          filter_size=4,
      ),
      dict(
          input_data=np.array([0, 1, 2, 3, 4, 5]),
          input_offset=(0.,),
          expected_data=np.array([2.25, 2, 3.25]),
          filter_size=4,
      ),
      dict(
          input_data=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
          input_offset=(1.,),
          expected_data=np.array([3., 4.]),
          filter_size=4,
      ),
      dict(
          input_data=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
          input_offset=(.5,),
          expected_data=np.array([1.5, 5.5]),
          filter_size=4,
      ),
      dict(
          input_data=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
          input_offset=(0.,),
          expected_data=np.array([3., 4.]),
          filter_size=4,
      ),
  )
  def test_downsample_velocity_1d_2x_grid(self, input_data, input_offset,
                                          expected_data, filter_size):
    source_grid = grids.Grid(
        input_data.shape, domain=[
            (0, 1),
        ])
    destination_grid = grids.Grid(
        np.array(expected_data.shape), domain=[
            (0, 1),
        ])
    v = grids.GridVariable(
        grids.GridArray(input_data, input_offset, source_grid),
        boundaries.periodic_boundary_conditions(1))
    (actual,) = resize.top_hat_downsample(source_grid, destination_grid, (v,),
                                          filter_size)
    expected = grids.GridVariable(
        grids.GridArray(expected_data, input_offset, grid=destination_grid),
        boundaries.periodic_boundary_conditions(1))
    self.assertAllClose(expected, actual)

  @parameterized.parameters(
      # Dirichlet BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.0, 0.0),)),
          input_data=np.array([1, 2, 3, 4, 5, 0]),
          input_offset=(1.,),
          expected_data=np.array([2, 3.25, 0]),
          filter_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.0, 0.0),)),
          input_data=np.array([0, 1, 2, 3, 4, 5]),
          input_offset=(.5,),
          expected_data=np.array([.75, 2.5, 1.75]),
          filter_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.0, 0.0),)),
          input_data=np.array([0, 1, 2, 3, 4, 5]),
          input_offset=(0.,),
          expected_data=np.array([0, 2, 3.25]),
          filter_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.0, 1.0),)),
          input_data=np.array([1, 2, 3, 4, 5, 1]),
          input_offset=(1.,),
          expected_data=np.array([2, 3.375, 1]),
          filter_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.0, 1.0),)),
          input_data=np.array([0, 1, 2, 3, 4, 5]),
          input_offset=(.5,),
          expected_data=np.array([.75, 2.5, 2.25]),
          filter_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.0, 1.0),)),
          input_data=np.array([0, 1, 2, 3, 4, 5]),
          input_offset=(0.,),
          expected_data=np.array([0, 2, 3.375]),
          filter_size=4,
      ),
  )
  def test_downsample_velocity_1d_2x_grid_dirichlet(self, bc_types, input_data,
                                                    input_offset, expected_data,
                                                    filter_size):
    source_grid = grids.Grid(
        input_data.shape, domain=[
            (0, 1),
        ])
    destination_grid = grids.Grid(
        np.array(input_data.shape) // 2, domain=[
            (0, 1),
        ])
    v = grids.GridVariable(
        grids.GridArray(input_data, input_offset, source_grid),
        boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1]))
    actual = resize.top_hat_downsample(source_grid, destination_grid, (v,),
                                       filter_size)
    expected = (grids.GridVariable(
        grids.GridArray(expected_data, input_offset, grid=destination_grid),
        boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])),)
    self.assertAllClose(expected[0], actual[0])

  @parameterized.parameters(
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),
                     (BCType.PERIODIC, BCType.PERIODIC)), ((None, None),
                                                           (None, None))),
          input_data=np.array([
              [0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11],
              [12, 13, 14, 15],
          ]),
          input_offset=(0.5, 1.),
          expected_data=np.array([[3, 4], [11, 12]]),
          filter_size=None,
      ),
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),
                     (BCType.DIRICHLET, BCType.DIRICHLET)), ((None, None),
                                                             (0.0, 1.0))),
          input_data=np.array([
              [0, 2, 3, 1],
              [4, 5, 6, 1],
              [8, 9, 10, 1],
              [12, 13, 14, 1],
          ]),
          input_offset=(0.5, 1.),
          expected_data=np.array([[3.375, 1], [11, 1]]),
          filter_size=None,
      ),
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),
                     (BCType.DIRICHLET, BCType.DIRICHLET)), ((None, None),
                                                             (0.0, 1.0))),
          input_data=np.array([
              [0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11],
              [12, 13, 14, 15],
          ]),
          input_offset=(1.0, .5),
          expected_data=np.array([[4.5, 6.5], [8.5, 10.5]]),
          filter_size=None,
      ),
  )
  def test_downsample_velocity_2d_2x_grid(self, bc_types, input_data,
                                          input_offset, expected_data,
                                          filter_size):
    source_grid = grids.Grid(input_data.shape, domain=[(0, 1), (0, 1)])
    destination_grid = grids.Grid(
        np.array(input_data.shape) // 2, domain=[(0, 1), (0, 1)])
    v = grids.GridVariable(
        grids.GridArray(input_data, input_offset, source_grid),
        boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1]))
    actual = resize.top_hat_downsample(source_grid, destination_grid, (v,),
                                       filter_size)
    expected = (grids.GridVariable(
        grids.GridArray(expected_data, input_offset, grid=destination_grid),
        boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])),)
    self.assertAllClose(expected[0], actual[0])

if __name__ == '__main__':
  absltest.main()
