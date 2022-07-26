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
"""Tests for jax_cfd.boundaries."""

# TODO(jamieas): Consider updating these tests using the `hypothesis` framework.
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np

BCType = boundaries.BCType


class ConstantBoundaryConditionsTest(test_util.TestCase):

  def test_init_usage(self):

    with self.subTest('init 1d'):
      bc = boundaries.HomogeneousBoundaryConditions(
          ((BCType.PERIODIC, BCType.PERIODIC)))
      self.assertEqual(bc.types, (('periodic', 'periodic')))

    with self.subTest('init 2d'):
      bc = boundaries.HomogeneousBoundaryConditions([
          (BCType.PERIODIC, BCType.PERIODIC),
          (BCType.DIRICHLET, BCType.DIRICHLET)
      ])
      self.assertEqual(bc.types, (
          ('periodic', 'periodic'),
          ('dirichlet', 'dirichlet'),
      ))

    with self.subTest('periodic bc utility'):
      bc = boundaries.periodic_boundary_conditions(ndim=3)
      self.assertEqual(bc.types, (
          ('periodic', 'periodic'),
          ('periodic', 'periodic'),
          ('periodic', 'periodic'),
      ))

    with self.subTest('dirichlet bc utility'):
      bc = boundaries.dirichlet_boundary_conditions(ndim=3)
      self.assertEqual(bc.types, (
          ('dirichlet', 'dirichlet'),
          ('dirichlet', 'dirichlet'),
          ('dirichlet', 'dirichlet'),
      ))

    with self.subTest('neumann bc utility'):
      bc = boundaries.neumann_boundary_conditions(ndim=3)
      self.assertEqual(bc.types, (
          ('neumann', 'neumann'),
          ('neumann', 'neumann'),
          ('neumann', 'neumann'),
      ))

    with self.subTest('channel flow 2d bc utility'):
      bc = boundaries.channel_flow_boundary_conditions(ndim=2)
      self.assertEqual(bc.types, (
          ('periodic', 'periodic'),
          ('dirichlet', 'dirichlet'),
      ))

    with self.subTest('channel flow 3d bc utility'):
      bc = boundaries.channel_flow_boundary_conditions(ndim=3)
      self.assertEqual(bc.types, (
          ('periodic', 'periodic'),
          ('dirichlet', 'dirichlet'),
          ('periodic', 'periodic'),
      ))

    with self.subTest('periodic and neumann bc utility'):
      bc = boundaries.periodic_and_neumann_boundary_conditions()
      self.assertEqual(bc.types, (
          ('periodic', 'periodic'),
          ('neumann', 'neumann'),
      ))

  @parameterized.parameters(
      dict(
          shape=(11,),
          initial_offset=(0.0,),
          step=1,
          offset=(0,),
      ),
      dict(
          shape=(11,),
          initial_offset=(0.0,),
          step=1,
          offset=(1,),
      ),
      dict(
          shape=(11,),
          initial_offset=(0.0,),
          step=1,
          offset=(-1,),
      ),
      dict(
          shape=(11,),
          initial_offset=(0.0,),
          step=1,
          offset=(5,),
      ),
      dict(
          shape=(11,),
          initial_offset=(0.0,),
          step=1,
          offset=(13,),
      ),
      dict(
          shape=(11,),
          initial_offset=(0.0,),
          step=1,
          offset=(31,),
      ),
      dict(
          shape=(11, 12, 17),
          initial_offset=(-0.5, -1.0, 1.0),
          step=0.1,
          offset=(-236, 10001, 3),
      ),
      dict(
          shape=(121,),
          initial_offset=(-0.5,),
          step=1,
          offset=(31,),
      ),
      dict(
          shape=(11, 12, 17),
          initial_offset=(0.5, 0.0, 1.0),
          step=0.1,
          offset=(-236, 10001, 3),
      ),
  )
  def test_shift_periodic(self, shape, initial_offset, step, offset):
    """Test that `shift` returns the expected values for periodic BC."""
    grid = grids.Grid(shape, step)
    data = np.arange(np.prod(shape)).reshape(shape)
    array = grids.GridArray(data, initial_offset, grid)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)

    shifted_array = array
    for axis, o in enumerate(offset):
      shifted_array = bc.shift(shifted_array, o, axis)

    shifted_indices = [(jnp.arange(s) + o) % s for s, o in zip(shape, offset)]
    shifted_mesh = jnp.meshgrid(*shifted_indices, indexing='ij')
    expected_offset = tuple(i + o for i, o in zip(initial_offset, offset))
    expected = grids.GridArray(data[tuple(shifted_mesh)], expected_offset, grid)

    self.assertArrayEqual(shifted_array, expected)

  @parameterized.parameters(
      # Periodic BC
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=-2,
          expected_data=np.array([13, 14, 11, 12]),
          expected_offset=(-2,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=-1,
          expected_data=np.array([14, 11, 12, 13]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=1,
          expected_data=np.array([12, 13, 14, 11]),
          expected_offset=(1,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=2,
          expected_data=np.array([13, 14, 11, 12]),
          expected_offset=(2,),
      ),
      # Dirichlet BC
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=-1,
          expected_data=np.array([-12, 11, 12, 13]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=1,
          expected_data=np.array([12, 13, 14, 0]),
          expected_offset=(1,),
      ),
      # Neumann BC
      dict(
          bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=-1,
          expected_data=np.array([11, 11, 12, 13]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=1,
          expected_data=np.array([12, 13, 14, 14]),
          expected_offset=(1,),
      ),
      # Dirichlet / Neumann BC
      dict(
          bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=-1,
          expected_data=np.array([-12, 11, 12, 13]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          shift_offset=1,
          expected_data=np.array([12, 13, 14, 14]),
          expected_offset=(1,),
      ),
  )
  def test_shift_1d(self, bc_types, input_data, input_offset, shift_offset,
                    expected_data, expected_offset):
    grid = grids.Grid(input_data.shape)
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.HomogeneousBoundaryConditions(bc_types)
    actual = bc.shift(array, shift_offset, axis=0)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      # Periodic BC
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-2,
          expected_data=np.array([13, 14, 11, 12, 13, 14]),
          expected_offset=(-2,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-1,
          expected_data=np.array([14, 11, 12, 13, 14]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=1,
          expected_data=np.array([11, 12, 13, 14, 11]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=2,
          expected_data=np.array([11, 12, 13, 14, 11, 12]),
          expected_offset=(0,),
      ),
      # Dirichlet BC
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-1,
          expected_data=np.array([-12, 11, 12, 13, 14]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=1,
          expected_data=np.array([11, 12, 13, 14, 0]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-3,
          expected_data=np.array([-14, -13, -12, 11, 12, 13, 14]),
          expected_offset=(-3,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=3,
          expected_data=np.array([11, 12, 13, 14, 0, -14, -13]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0.5,),
          width=3,
          expected_data=np.array([11, 12, 13, 14, -14, -13, -12]),
          expected_offset=(0.5,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(1.,),
          width=3,
          expected_data=np.array([11, 12, 13, 14, -13, -12, -11]),
          expected_offset=(1.,),
      ),
      # Neumann BC
      dict(
          bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-1,
          expected_data=np.array([11, 11, 12, 13, 14]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=1,
          expected_data=np.array([11, 12, 13, 14, 14]),
          expected_offset=(0,),
      ),
      # Dirichlet / Neumann BC
      dict(
          bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-1,
          expected_data=np.array([-12, 11, 12, 13, 14]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=1,
          expected_data=np.array([11, 12, 13, 14, 14]),
          expected_offset=(0,),
      ),
  )
  def test_pad_1d(self, bc_types, input_data, input_offset, width,
                  expected_data, expected_offset):
    grid = grids.Grid(input_data.shape)
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.HomogeneousBoundaryConditions(bc_types)
    actual = bc._pad(array, width, axis=0)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      # Dirichlet BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=-3,
          input_data=np.array([-14, -13, -12, 11, 12, 13, 14]),
          input_offset=(-3,),
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=0,
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=3,
          input_data=np.array([11, 12, 13, 14, 2, -12, -11]),
          input_offset=(0,),
      ),
      # Neumann BC
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=-1,
          input_data=np.array([10, 11, 12, 13, 14]),
          input_offset=(-1,),
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=0,
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=1,
          input_data=np.array([11, 12, 13, 14, 12]),
          input_offset=(0,),
      ),
      # Dirichlet / Neumann BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=-1,
          input_data=np.array([-12, 11, 12, 13, 14]),
          input_offset=(-1,),
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=1,
          input_data=np.array([11, 12, 13, 14, 12]),
          input_offset=(0,),
      ),
      # Periodic BC
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=-1,
          input_data=np.array([-12, 11, 12, 13, 14]),
          input_offset=(-1,),
      ),
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          width=1,
          input_data=np.array([11, 12, 13, 14, 12]),
          input_offset=(0,),
      ),
  )
  def test_trim_padding_1d(self, bc_types, expected_data, expected_offset,
                           width, input_data, input_offset):
    grid = grids.Grid(expected_data.shape)
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.HomogeneousBoundaryConditions(bc_types)
    actual, _ = bc._trim_padding(array, axis=0)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      # Dirichlet BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([12, 13, 14]),
          expected_offset=(1,),
          input_data=np.array([-14, -13, -12, 11, 12, 13, 14]),
          input_offset=(-3,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([12, 13, 14]),
          expected_offset=(1,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([12, 13, 14]),
          expected_offset=(1,),
          input_data=np.array([11, 12, 13, 14, 2, -12, -11]),
          input_offset=(0,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(1,),
          input_data=np.array([11, 12, 13, 14, 2, -12, -11]),
          input_offset=(1,),
          grid_size=5,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(.5,),
          input_data=np.array([11, 12, 13, 14, -12, -11]),
          input_offset=(.5,),
          grid_size=4,
      ),
      # Neumann BC
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([10, 11, 12, 13, 14]),
          input_offset=(-1,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14, 12]),
          input_offset=(0,),
          grid_size=4,
      ),
      # Periodic BC
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([-12, 11, 12, 13, 14]),
          input_offset=(-1,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14, 12]),
          input_offset=(0,),
          grid_size=4,
      ),
  )
  def test_trim_boundary_1d(self, bc_types, expected_data, expected_offset,
                            input_data, input_offset, grid_size):
    grid = grids.Grid((grid_size,))
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
    actual = bc.trim_boundary(array)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      # Dirichlet BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([1, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([12, 13, 14]),
          input_offset=(1,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14, 2]),
          expected_offset=(1,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(1,),
          grid_size=5,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(.5,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(.5,),
          grid_size=4,
      ),
      # Neumann BC
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
      # Periodic BC
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
  )
  def test_pad_and_impose_bc_1d(self, bc_types, expected_data, expected_offset,
                                input_data, input_offset, grid_size):
    grid = grids.Grid((grid_size,))
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
    actual = bc.pad_and_impose_bc(array, expected_offset).array
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      # Dirichlet BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([1, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([0, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14, 2]),
          expected_offset=(1,),
          input_data=np.array([11, 12, 13, 14, 11]),
          input_offset=(1,),
          grid_size=5,
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(.5,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(.5,),
          grid_size=4,
      ),
      # Neumann BC
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
      # Periodic BC
      dict(
          bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          grid_size=4,
      ),
  )
  def test_impose_bc_1d(self, bc_types, expected_data, expected_offset,
                        input_data, input_offset, grid_size):
    grid = grids.Grid((grid_size,))
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
    actual = bc.impose_bc(array).array
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      # Dirichlet BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-3,
          expected_data=np.array([-14, -13, -12, 11, 12, 13, 14]),
          expected_offset=(-3,),
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=3,
          expected_data=np.array([11, 12, 13, 14, 2, -12, -11]),
          expected_offset=(0,),
      ),
      # Neumann BC
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-1,
          expected_data=np.array([10, 11, 12, 13, 14]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=1,
          expected_data=np.array([11, 12, 13, 14, 12]),
          expected_offset=(0,),
      ),
      # Dirichlet / Neumann BC
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-1,
          expected_data=np.array([-12, 11, 12, 13, 14]),
          expected_offset=(-1,),
      ),
      dict(
          bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=1,
          expected_data=np.array([11, 12, 13, 14, 12]),
          expected_offset=(0,),
      ),
  )
  def test_pad_1d_inhomogeneous(self, bc_types, input_data, input_offset, width,
                                expected_data, expected_offset):
    grid = grids.Grid(input_data.shape)
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.ConstantBoundaryConditions(*bc_types)
    actual = bc._pad(array, width, axis=0)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 0.5),
          width=-2,
          axis=0,
          expected_data=np.array([
              [-21, -22, -23, -24],
              [-11, -12, -13, -14],
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          expected_offset=(-1.5, 0.5),
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 0.5),
          width=2,
          axis=1,
          expected_data=np.array([
              [11, 12, 13, 14, -14, -13],
              [21, 22, 23, 24, -24, -23],
              [31, 32, 33, 34, -34, -33],
          ]),
          expected_offset=(0.5, 0.5),
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 1),  # edge aligned offset
          width=-2,
          axis=1,
          expected_data=np.array([
              [-11, 0, 11, 12, 13, 14],
              [-21, 0, 21, 22, 23, 24],
              [-31, 0, 31, 32, 33, 34],
          ]),
          expected_offset=(0.5, -1),
      ),
  )
  def test_pad_dirichlet_cell_center(self, input_data, input_offset, width,
                                     axis, expected_data, expected_offset):
    grid = grids.Grid(input_data.shape)
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
    actual = bc._pad(array, width, axis)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 0.5),
          width=-2,
          axis=0,
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.0, 0.5),
          width=-2,
          axis=0,
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(1.0, 0.5),
          width=-2,
          axis=0,
      ))
  def test_pad_periodic_raises(self, input_data, input_offset, width, axis):
    grid = grids.Grid((4, 4))
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    error_msg = 'the GridArray shape does not match the grid.'
    with self.assertRaisesRegex(ValueError, error_msg):
      _ = bc._pad(array, width, axis)

  @parameterized.parameters(
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 0.5),
          width=-1,
          axis=0,
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.0, 0.5),
          width=-1,
          axis=0,
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(1.0, 0.5),
          width=-1,
          axis=0,
      ))
  def test_pad_neumann_raises(self, input_data, input_offset, width, axis):
    grid = grids.Grid((4, 4))
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.neumann_boundary_conditions(grid.ndim)
    error_msg = 'the GridArray shape does not match the grid.'
    with self.assertRaisesRegex(ValueError, error_msg):
      _ = bc._pad(array, width, axis)

  @parameterized.parameters(
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 0.5),
          width=-2,
          axis=0,
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.0, 0.5),
          width=-2,
          axis=0,
      ),
  )
  def test_pad_dirichlet_raises(self, input_data, input_offset, width, axis):
    grid = grids.Grid((4, 4))
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
    if input_offset[axis] == 0.5:
      error_msg = 'the GridArray shape does not match the grid.'
    else:
      error_msg = ('For a dirichlet cell-face boundary condition, the GridArray'
                   ' has more than 1 grid point missing.')
    with self.assertRaisesRegex(ValueError, error_msg):
      _ = bc._pad(array, width, axis)

  @parameterized.parameters(
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 0.5),
          values=((1.0, 2.0), (3.0, 4.0)),
          width=-2,
          axis=0,
          expected_data=np.array([
              [-19, -20, -21, -22],
              [-9, -10, -11, -12],
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          expected_offset=(-1.5, 0.5),
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 0.5),
          values=((1.0, 2.0), (3.0, 4.0)),
          width=2,
          axis=1,
          expected_data=np.array([
              [11, 12, 13, 14, -6, -5],
              [21, 22, 23, 24, -16, -15],
              [31, 32, 33, 34, -26, -25],
          ]),
          expected_offset=(0.5, 0.5),
      ),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          input_offset=(0.5, 1),  # edge aligned offset
          values=((1.0, 2.0), (3.0, 4.0)),
          width=-2,
          axis=1,
          expected_data=np.array([
              [-8, 3, 11, 12, 13, 14],
              [-18, 3, 21, 22, 23, 24],
              [-28, 3, 31, 32, 33, 34],
          ]),
          expected_offset=(0.5, -1),
      ),
  )
  def test_pad_dirichlet_cell_center_inhomogeneous(self, input_data,
                                                   input_offset, values, width,
                                                   axis, expected_data,
                                                   expected_offset):
    input_data = input_data.astype('float')
    expected_data = expected_data.astype('float')
    grid = grids.Grid(input_data.shape)
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim, values)
    actual = bc._pad(array, width, axis)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      dict(
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=-1,
          expected_data=np.array([12, 13, 14]),
          expected_offset=(1,),
      ),
      dict(
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=0,
          expected_data=np.array([11, 12, 13, 14]),
          expected_offset=(0,),
      ),
      dict(
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=1,
          expected_data=np.array([11, 12, 13]),
          expected_offset=(0,),
      ),
      dict(
          input_data=np.array([11, 12, 13, 14]),
          input_offset=(0,),
          width=2,
          expected_data=np.array([11, 12]),
          expected_offset=(0,),
      ),
  )
  def test_trim_1d(self, input_data, input_offset, width, expected_data,
                   expected_offset):
    grid = grids.Grid(input_data.shape)
    array = grids.GridArray(input_data, input_offset, grid)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    # Note: trim behavior does not depend on bc type
    actual = bc._trim(array, width, axis=0)
    expected = grids.GridArray(expected_data, expected_offset, grid)
    self.assertArrayEqual(actual, expected)

  @parameterized.parameters(
      dict(
          values=((1.0, 2.0),), axis=0, shape=(3,), expected_values=(1.0, 2.0)),
      dict(
          values=((1.0, 2.0), (3.0, 4.0)),
          axis=0,
          shape=(3, 4),
          expected_values=((1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0))),
      dict(
          values=((1.0, 2.0), (3.0, 4.0)),
          axis=1,
          shape=(3, 4),
          expected_values=((3.0, 3.0, 3.0), (4.0, 4.0, 4.0))),
  )
  def test_values_constant_boundary(self, values, axis, shape, expected_values):
    grid = grids.Grid(shape)
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim, values)
    actual = bc.values(axis, grid)
    self.assertArrayEqual(actual, expected_values)
    self.assertIsInstance(actual, tuple)
    for x in actual:
      self.assertIsInstance(x, jnp.ndarray)

  @parameterized.parameters(
      dict(axis=0, shape=(3,), expected_values=(0.0, 0.0)),
      dict(
          axis=0,
          shape=(3, 4),
          expected_values=((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))),
      dict(
          axis=1,
          shape=(3, 4),
          expected_values=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
  )
  def test_values_homogeneous_boundary(self, axis, shape, expected_values):
    grid = grids.Grid(shape)
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
    actual = bc.values(axis, grid)
    self.assertArrayEqual(actual, expected_values)
    self.assertIsInstance(actual, tuple)
    for x in actual:
      self.assertIsInstance(x, jnp.ndarray)

  @parameterized.parameters(
      dict(
          input_data=np.array([11, 12, 13, 14]),
          offset=(0.0,),
          values=((1.0, 2.0),),
          expected_data=np.array([1, 12, 13, 14])),
      dict(
          input_data=np.array([11, 12, 13, 14]),
          offset=(1.0,),
          values=((1.0, 2.0),),
          expected_data=np.array([11, 12, 13, 2])),
      dict(
          input_data=np.array([11, 12, 13, 14]),
          offset=(0.5,),
          values=((1.0, 2.0),),
          expected_data=np.array([11, 12, 13, 14])),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          offset=(1.0, 0.5),
          values=((1.0, 2.0), (3.0, 4.0)),
          expected_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [2, 2, 2, 2],
          ])),
      dict(
          input_data=np.array([
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
          ]),
          offset=(0.5, 0.0),
          values=((1.0, 2.0), (3.0, 4.0)),
          expected_data=np.array([
              [3, 12, 13, 14],
              [3, 22, 23, 24],
              [3, 32, 33, 34],
          ])),
  )
  def test_impose_bc_constant_boundary(
      self, input_data, offset, values, expected_data):
    grid = grids.Grid(input_data.shape)
    array = grids.GridArray(input_data, offset, grid)
    bc = boundaries.dirichlet_boundary_conditions(grid.ndim, values)
    variable = grids.GridVariable(array, bc)
    variable = variable.impose_bc()
    expected = grids.GridArray(expected_data, offset, grid)
    self.assertArrayEqual(variable.array, expected)

  def test_has_all_periodic_boundary_conditions(self):
    grid = grids.Grid((10, 10))
    array = grids.GridArray(np.zeros((10, 10)), (0.5, 0.5), grid)
    periodic_bc = boundaries.periodic_boundary_conditions(ndim=2)
    nonperiodic_bc = boundaries.periodic_and_neumann_boundary_conditions()

    with self.subTest('returns True'):
      c = grids.GridVariable(array, periodic_bc)
      v = (grids.GridVariable(array, periodic_bc),
           grids.GridVariable(array, periodic_bc))
      self.assertTrue(boundaries.has_all_periodic_boundary_conditions(c, *v))

    with self.subTest('returns False'):
      c = grids.GridVariable(array, periodic_bc)
      v = (grids.GridVariable(array, periodic_bc),
           grids.GridVariable(array, nonperiodic_bc))
      self.assertFalse(boundaries.has_all_periodic_boundary_conditions(c, *v))

  def test_get_pressure_bc_from_velocity_2d(self):
    grid = grids.Grid((10, 10))
    u_array = grids.GridArray(jnp.zeros(grid.shape), (1, 0.5), grid)
    v_array = grids.GridArray(jnp.zeros(grid.shape), (0.5, 1), grid)
    velocity_bc = boundaries.channel_flow_boundary_conditions(ndim=2)
    v = (grids.GridVariable(u_array, velocity_bc),
         grids.GridVariable(v_array, velocity_bc))
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
    self.assertEqual(pressure_bc.types, ((BCType.PERIODIC, BCType.PERIODIC),
                                         (BCType.NEUMANN, BCType.NEUMANN)))

  def test_get_pressure_bc_from_velocity_3d(self):
    grid = grids.Grid((10, 10, 10))
    u_array = grids.GridArray(jnp.zeros(grid.shape), (1, 0.5, 0.5), grid)
    v_array = grids.GridArray(jnp.zeros(grid.shape), (0.5, 1, 0.5), grid)
    w_array = grids.GridArray(jnp.zeros(grid.shape), (0.5, 0.5, 1), grid)
    velocity_bc = boundaries.channel_flow_boundary_conditions(ndim=3)
    v = (grids.GridVariable(u_array, velocity_bc),
         grids.GridVariable(v_array, velocity_bc),
         grids.GridVariable(w_array, velocity_bc))
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
    self.assertEqual(pressure_bc.types, ((BCType.PERIODIC, BCType.PERIODIC),
                                         (BCType.NEUMANN, BCType.NEUMANN),
                                         (BCType.PERIODIC, BCType.PERIODIC)))

if __name__ == '__main__':
  absltest.main()
