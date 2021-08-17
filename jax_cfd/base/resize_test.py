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

import functools

from absl.testing import absltest
from absl.testing import parameterized
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.base import pressure
from jax_cfd.base import resize
from jax_cfd.base import test_util
import numpy as np

from google3.research.simulation.whirl.experiments.convection import implicit_solvers
from google3.research.simulation.whirl.experiments.convection import nonperiodic_boundary_conditions as bc

SIZE_X = 8


class ResizeTest(test_util.TestCase):

  @parameterized.parameters(
      dict(
          u=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                      [12, 13, 14, 15]]),
          direction=0,
          factor=2,
          expected=np.array([[4.5, 6.5], [12.5, 14.5]])),
      dict(
          u=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                      [12, 13, 14, 15]]),
          direction=1,
          factor=2,
          expected=np.array([[3., 5.], [11., 13.]])),
  )
  def testDownsampleVelocityComponent(self, u, direction, factor, expected):
    """Test `downsample_array` produces the expected results."""
    actual = resize.downsample_staggered_velocity_component(
        u, direction, factor)
    self.assertAllClose(expected, actual, atol=1e-6)

  @parameterized.parameters(
      dict(
          u=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                      [12, 13, 14, 15]]),
          factor=2,
          expected=np.array([[5, 7], [13, 15]])),
  )
  def testSubsampleVelocityComponent(self, u, factor, expected):
    """Test `downsample_array` produces the expected results."""
    actual = resize.subsample_staggered_velocity_component(
        u, None, factor)
    self.assertAllClose(expected, actual, atol=1e-6)

  @parameterized.parameters(
      ((1, 2),),
      ((2, 1),),
      ((2, 2),),
      ((1, 1),),
  )
  def test_downsample_staggered_velocity_periodic_dirichlet(self, factors):

    def default_boundary_fn_velocity(grid):
      velocity_bc = functools.partial(bc.non_slip_bc_vector, grid=grid)
      offsets = [(1.0, 0.5), (0.5, 1.0)]

      def _apply(v):
        v = tuple([
            grids.AlignedArray(u.data, offset) for u, offset in zip(v, offsets)
        ])
        v = velocity_bc(v)
        return v

      return _apply

    rs = np.random.RandomState(0)
    grid = grids.Grid(
        (SIZE_X, SIZE_X + 2),
        domain=((0, 1), (-1 / float(SIZE_X), 1 + 1 / float(SIZE_X))),
        boundaries=('periodic', 'dirichlet'))
    b = grids.AlignedArray(
        rs.randn(SIZE_X, SIZE_X + 2).astype(np.float32), (0.5, 1.0))
    b1 = grids.AlignedArray(
        rs.randn(SIZE_X, SIZE_X + 2).astype(np.float32), (1.0, 0.5))
    velocity = (b1, b)
    velocity = bc.non_slip_bc_vector(velocity, grid)
    pressure_solve = implicit_solvers.pressure_solve_fast_diag(
        grid, boundary_type=(('periodic',), ('neumann',)))
    velocity = pressure.projection(velocity, grid, pressure_solve)
    velocity = bc.non_slip_bc_vector(velocity, grid)

    destination_grid = grids.Grid(
        (SIZE_X // factors[0], SIZE_X // factors[1] + 2),
        domain=((0, 1), (-1 / float(SIZE_X) * factors[1],
                         1 + 1 / float(SIZE_X) * factors[1])),
        boundaries=('periodic', 'dirichlet'))
    velocity_reduced = resize.downsample_staggered_velocity(
        grid, destination_grid, velocity, default_boundary_fn_velocity)
    # checks that zero divergence is preserved
    self.assertAllClose(
        bc.reduce_to_interior(
            fd.divergence(velocity_reduced, destination_grid),
            destination_grid),
        0,
        atol=1e-5)
    x_velocity = velocity_reduced[0]
    y_velocity = velocity_reduced[1]
    # checks that the final size of the reduced velocity equals destination_grid
    self.assertEqual(x_velocity.data.shape,
                     (SIZE_X // factors[0], SIZE_X // factors[1] + 2))
    self.assertEqual(y_velocity.data.shape,
                     (SIZE_X // factors[0], SIZE_X // factors[1] + 2))
    # checks that the dirichlet boundary is satisfied
    self.assertAllClose(x_velocity.data[:, 0] + x_velocity.data[:, 1], 0)
    self.assertAllClose(x_velocity.data[:, -1] + x_velocity.data[:, -2], 0)
    self.assertAllClose(y_velocity.data[:, 0], 0)
    self.assertAllClose(y_velocity.data[:, -2], 0)

    # checks that it does nothing if no downsampling
    if factors[0] == factors[1] == 1:
      self.assertAllClose(x_velocity.data, velocity[0].data)
      self.assertAllClose(y_velocity.data, velocity[1].data)

  @parameterized.parameters(
      ((1, 2),),
      ((2, 1),),
      ((2, 2),),
      ((1, 1),),
  )
  def test_downsample_staggered_velocity_periodic_periodic(self, factors):
    rs = np.random.RandomState(0)
    grid = grids.Grid((SIZE_X, SIZE_X),
                      domain=((0, 1), (0, 1)),
                      boundaries=('periodic', 'periodic'))
    b = grids.AlignedArray(
        rs.randn(SIZE_X, SIZE_X).astype(np.float32), (0.5, 1.0))
    b1 = grids.AlignedArray(
        rs.randn(SIZE_X, SIZE_X).astype(np.float32), (1.0, 0.5))
    velocity = (b1, b)
    velocity = bc.non_slip_bc_vector(velocity, grid)
    pressure_solve = implicit_solvers.pressure_solve_fast_diag(
        grid, boundary_type=(('periodic',), ('periodic',)))
    velocity = pressure.projection(velocity, grid, pressure_solve)
    velocity = bc.non_slip_bc_vector(velocity, grid)

    destination_grid = grids.Grid((SIZE_X // factors[0], SIZE_X // factors[1]),
                                  domain=((0, 1), (0, 1)),
                                  boundaries=('periodic', 'periodic'))
    velocity_reduced = resize.downsample_staggered_velocity(
        grid, destination_grid, velocity)
    # checks that zero divergence is preserved
    self.assertAllClose(
        bc.reduce_to_interior(
            fd.divergence(velocity_reduced, destination_grid),
            destination_grid),
        0,
        atol=1e-5)
    x_velocity = velocity_reduced[0]
    y_velocity = velocity_reduced[1]
    # checks that the final size of the reduced velocity equals destination_grid
    self.assertEqual(x_velocity.data.shape,
                     (SIZE_X // factors[0], SIZE_X // factors[1]))
    self.assertEqual(y_velocity.data.shape,
                     (SIZE_X // factors[0], SIZE_X // factors[1]))

    # checks that it does nothing if no downsampling
    if factors[0] == factors[1] == 1:
      self.assertAllClose(x_velocity.data, velocity[0].data)
      self.assertAllClose(y_velocity.data, velocity[1].data)

  @parameterized.parameters(
      ((1, 2), (1.0, 0.5)),
      ((2, 1), (1.0, 0.5)),
      ((2, 2), (1.0, 0.5)),
      ((1, 1), (1.0, 0.5)),
      ((1, 2), (0.5, 1.0)),
      ((2, 1), (0.5, 1.0)),
      ((2, 2), (0.5, 1.0)),
      ((1, 1), (0.5, 1.0)),
  )
  def test_downsample_staggered_velocity_periodic_dirichlet_scalar(
      self, factors, scalar_offset):

    def default_boundary_fn(grid):
      velocity_bc = functools.partial(bc.non_slip_bc_vector, grid=grid)
      temperture_bc = functools.partial(
          bc.dirichlet_bc_scalar, grid=grid, value_hi=0.0, value_lo=1.0)
      offsets = [(1.0, 0.5), (0.5, 1.0), scalar_offset]

      def _apply(v_c):
        v_c = tuple([
            grids.AlignedArray(u.data, offset)
            for u, offset in zip(v_c, offsets)
        ])
        v = v_c[:2]
        c = v_c[-1]
        v = velocity_bc(v)
        return (v[0], v[1], temperture_bc(c))

      return _apply

    rs = np.random.RandomState(0)
    grid = grids.Grid(
        (SIZE_X, SIZE_X + 2),
        domain=((0, 1), (-1 / float(SIZE_X), 1 + 1 / float(SIZE_X))),
        boundaries=('periodic', 'dirichlet'))
    b = grids.AlignedArray(
        rs.randn(SIZE_X, SIZE_X + 2).astype(np.float32), (0.5, 1.0))
    b1 = grids.AlignedArray(
        rs.randn(SIZE_X, SIZE_X + 2).astype(np.float32), (1.0, 0.5))
    scalar = grids.AlignedArray(
        rs.randn(SIZE_X, SIZE_X + 2).astype(np.float32), scalar_offset)
    velocity_scalar = (b1, b, scalar)
    velocity_scalar = default_boundary_fn(grid)(velocity_scalar)
    destination_grid = grids.Grid(
        (SIZE_X // factors[0], SIZE_X // factors[1] + 2),
        domain=((0, 1), (-1 / float(SIZE_X) * factors[1],
                         1 + 1 / float(SIZE_X) * factors[1])),
        boundaries=('periodic', 'dirichlet'))
    velocity_scalar_reduced = resize.downsample_staggered_velocity(
        grid, destination_grid, velocity_scalar, default_boundary_fn)
    x_velocity = velocity_scalar_reduced[0]
    y_velocity = velocity_scalar_reduced[1]
    scalar = velocity_scalar_reduced[2]
    # checks that the final size of the reduced velocity equals destination_grid
    self.assertEqual(x_velocity.data.shape,
                     (SIZE_X // factors[0], SIZE_X // factors[1] + 2))
    self.assertEqual(y_velocity.data.shape,
                     (SIZE_X // factors[0], SIZE_X // factors[1] + 2))
    # checks that the final size of the reduced scalar equals destination_grid
    self.assertEqual(scalar.data.shape,
                     (SIZE_X // factors[0], SIZE_X // factors[1] + 2))
    # checks that it does nothing if no downsampling
    if factors[0] == factors[1] == 1:
      self.assertAllClose(scalar.data, velocity_scalar[2].data)

    # checks that the dirichlet boundary is satisfied
    scalar_interp = interpolation.linear(
        scalar, offset=(0.5, 1.0), grid=destination_grid)
    self.assertAllClose(scalar_interp.data[:, 0], 1.0, atol=1e-6)
    self.assertAllClose(scalar_interp.data[:, -2], 0, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
