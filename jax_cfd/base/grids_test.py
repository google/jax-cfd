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
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.base import test_util
import numpy as np


class GridArrayTest(test_util.TestCase):

  def test_tree_util(self):
    array = grids.GridArray(jnp.arange(3), offset=(0,), grid=grids.Grid((3,)))
    flat, treedef = jax.tree_flatten(array)
    roundtripped = jax.tree_unflatten(treedef, flat)
    self.assertArrayEqual(array, roundtripped)

  def test_consistent_offset(self):
    data = jnp.arange(3)
    grid = grids.Grid((3,))
    array_offset_0 = grids.GridArray(data, offset=(0,), grid=grid)
    array_offset_1 = grids.GridArray(data, offset=(1,), grid=grid)

    offset = grids.consistent_offset(array_offset_0, array_offset_0)
    self.assertEqual(offset, (0,))

    with self.assertRaises(grids.InconsistentOffsetError):
      grids.consistent_offset(array_offset_0, array_offset_1)

  def test_averaged_offset(self):
    data = jnp.arange(3)
    grid = grids.Grid((3,))
    array_offset_0 = grids.GridArray(data, offset=(0,), grid=grid)
    array_offset_1 = grids.GridArray(data, offset=(1,), grid=grid)

    averaged_offset = grids.averaged_offset(array_offset_0, array_offset_1)
    self.assertEqual(averaged_offset, (0.5,))

  def test_control_volume_offsets(self):
    data = jnp.arange(5, 5)
    grid = grids.Grid((5, 5))
    array = grids.GridArray(data, offset=(0, 0), grid=grid)
    cv_offset = grids.control_volume_offsets(array)
    self.assertEqual(cv_offset, ((0.5, 0), (0, 0.5)))

  def test_consistent_grid(self):
    data = jnp.arange(3)
    offset = (0,)
    array_grid_3 = grids.GridArray(data, offset, grid=grids.Grid((3,)))
    array_grid_5 = grids.GridArray(data, offset, grid=grids.Grid((5,)))

    grid = grids.consistent_grid(array_grid_3, array_grid_3)
    self.assertEqual(grid, grids.Grid((3,)))

    with self.assertRaises(grids.InconsistentGridError):
      grids.consistent_grid(array_grid_3, array_grid_5)

  def test_add_sub_correctness(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    offsets = (0.5, 0.5)
    grid = grids.Grid((5, 5))
    input_array_1 = grids.GridArray(values_1, offsets, grid)
    input_array_2 = grids.GridArray(values_2, offsets, grid)
    actual_sum = input_array_1 + input_array_2
    actual_sub = input_array_1 - input_array_2
    expected_sum = grids.GridArray(values_1 + values_2, offsets, grid)
    expected_sub = grids.GridArray(values_1 - values_2, offsets, grid)
    self.assertAllClose(actual_sum, expected_sum, atol=1e-7)
    self.assertAllClose(actual_sub, expected_sub, atol=1e-7)

  def test_add_sub_offset_raise(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    offset_1 = (0.5, 0.5)
    offset_2 = (0.5, 0.0)
    grid = grids.Grid((5, 5))
    input_array_1 = grids.GridArray(values_1, offset_1, grid)
    input_array_2 = grids.GridArray(values_2, offset_2, grid)
    with self.assertRaises(grids.InconsistentOffsetError):
      _ = input_array_1 + input_array_2
    with self.assertRaises(grids.InconsistentOffsetError):
      _ = input_array_1 - input_array_2

  def test_add_sub_grid_raise(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    offset = (0.5, 0.5)
    grid_1 = grids.Grid((5, 5), domain=((0, 1), (0, 1)))
    grid_2 = grids.Grid((5, 5), domain=((-2, 2), (-2, 2)))
    input_array_1 = grids.GridArray(values_1, offset, grid_1)
    input_array_2 = grids.GridArray(values_2, offset, grid_2)
    with self.assertRaises(grids.InconsistentGridError):
      _ = input_array_1 + input_array_2
    with self.assertRaises(grids.InconsistentGridError):
      _ = input_array_1 - input_array_2

  def test_mul_div_correctness(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    scalar = 3.1415
    offset = (0.5, 0.5)
    grid = grids.Grid((5, 5))
    input_array_1 = grids.GridArray(values_1, offset, grid)
    input_array_2 = grids.GridArray(values_2, offset, grid)
    actual_mul = input_array_1 * input_array_2
    array_1_times_scalar = input_array_1 * scalar
    expected_1_times_scalar = grids.GridArray(values_1 * scalar, offset, grid)
    actual_div = input_array_1 / 2.5
    expected_div = grids.GridArray(values_1 / 2.5, offset, grid)
    expected_mul = grids.GridArray(values_1 * values_2, offset, grid)
    self.assertAllClose(actual_mul, expected_mul, atol=1e-7)
    self.assertAllClose(
        array_1_times_scalar, expected_1_times_scalar, atol=1e-7)
    self.assertAllClose(actual_div, expected_div, atol=1e-7)

  def test_add_inplace(self):
    values_1 = np.random.uniform(size=(5, 5))
    values_2 = np.random.uniform(size=(5, 5))
    offsets = (0.5, 0.5)
    grid = grids.Grid((5, 5))
    array = grids.GridArray(values_1, offsets, grid)
    array += values_2
    expected = grids.GridArray(values_1 + values_2, offsets, grid)
    self.assertAllClose(array, expected, atol=1e-7)

  def test_jit(self):
    u = grids.GridArray(jnp.ones([10, 10]), (.5, .5), grids.Grid((10, 10)))

    def f(u):
      return u.data < 2.

    self.assertAllClose(f(u), jax.jit(f)(u))

  def test_applied(self):
    grid = grids.Grid((10, 10))
    offset = (0.5, 0.5)
    u = grids.GridArray(jnp.ones([10, 10]), offset, grid)
    expected = grids.GridArray(-jnp.ones([10, 10]), offset, grid)
    actual = grids.applied(jnp.negative)(u)
    self.assertAllClose(expected, actual)


class GridVariableTest(test_util.TestCase):

  def test_constructor_and_attributes(self):
    with self.subTest('1d'):
      grid = grids.Grid((10,))
      data = np.zeros((10,), dtype=np.float32)
      array = grids.GridArray(data, offset=(0.5,), grid=grid)
      bc = boundaries.periodic_boundary_conditions(grid.ndim)
      variable = grids.GridVariable(array, bc)
      self.assertEqual(variable.array, array)
      self.assertEqual(variable.bc, bc)
      self.assertEqual(variable.dtype, np.float32)
      self.assertEqual(variable.shape, (10,))
      self.assertArrayEqual(variable.data, data)
      self.assertEqual(variable.offset, (0.5,))
      self.assertEqual(variable.grid, grid)

    with self.subTest('2d'):
      grid = grids.Grid((10, 10))
      data = np.zeros((10, 10), dtype=np.float32)
      array = grids.GridArray(data, offset=(0.5, 0.5), grid=grid)
      bc = boundaries.periodic_boundary_conditions(grid.ndim)
      variable = grids.GridVariable(array, bc)
      self.assertEqual(variable.array, array)
      self.assertEqual(variable.bc, bc)
      self.assertEqual(variable.dtype, np.float32)
      self.assertEqual(variable.shape, (10, 10))
      self.assertArrayEqual(variable.data, data)
      self.assertEqual(variable.offset, (0.5, 0.5))
      self.assertEqual(variable.grid, grid)

    with self.subTest('batch dim data'):
      grid = grids.Grid((10, 10))
      data = np.zeros((5, 10, 10), dtype=np.float32)
      array = grids.GridArray(data, offset=(0.5, 0.5), grid=grid)
      bc = boundaries.periodic_boundary_conditions(grid.ndim)
      variable = grids.GridVariable(array, bc)
      self.assertEqual(variable.array, array)
      self.assertEqual(variable.bc, bc)
      self.assertEqual(variable.dtype, np.float32)
      self.assertEqual(variable.shape, (5, 10, 10))
      self.assertArrayEqual(variable.data, data)
      self.assertEqual(variable.offset, (0.5, 0.5))
      self.assertEqual(variable.grid, grid)

    with self.subTest('raises exception'):
      with self.assertRaisesRegex(ValueError,
                                  'Incompatible dimension between grid and bc'):
        grid = grids.Grid((10,))
        data = np.zeros((10,))
        array = grids.GridArray(data, offset=(0.5,), grid=grid)  # 1D
        bc = boundaries.periodic_boundary_conditions(ndim=2)  # 2D
        grids.GridVariable(array, bc)

  @parameterized.parameters(
      dict(
          shape=(10,),
          offset=(0.0,),
      ),
      dict(
          shape=(10,),
          offset=(0.5,),
      ),
      dict(
          shape=(10,),
          offset=(1.0,),
      ),
      dict(
          shape=(10, 10),
          offset=(1.0, 0.0),
      ),
      dict(
          shape=(10, 10, 10),
          offset=(1.0, 0.0, 0.5),
      ),
  )
  def test_interior_consistency_periodic(self, shape, offset):
    grid = grids.Grid(shape)
    data = np.random.randint(0, 10, shape)
    array = grids.GridArray(data, offset=offset, grid=grid)
    bc = boundaries.periodic_boundary_conditions(ndim=len(shape))
    u = grids.GridVariable(array, bc)
    u_interior = u.trim_boundary()
    self.assertEqual(u_interior, u.array)

  @parameterized.parameters(
      dict(
          shape=(10,),
          bc=boundaries.dirichlet_boundary_conditions(ndim=1),
      ),
      dict(
          shape=(10,),
          bc=boundaries.neumann_boundary_conditions(ndim=1),
      ),
      dict(
          shape=(10, 10),
          bc=boundaries.dirichlet_boundary_conditions(ndim=2),
      ),
      dict(
          shape=(10, 10),
          bc=boundaries.neumann_boundary_conditions(ndim=2),
      ),
      dict(
          shape=(10, 10, 10),
          bc=boundaries.dirichlet_boundary_conditions(ndim=3),
      ),
      dict(
          shape=(10, 10, 10),
          bc=boundaries.neumann_boundary_conditions(ndim=3),
      ),
  )
  def test_interior_consistency_no_edge_offsets(self, bc, shape):
    grid = grids.Grid(shape)
    data = np.random.randint(0, 10, shape)
    array = grids.GridArray(data, offset=(0.5,) * len(shape), grid=grid)
    u = grids.GridVariable(array, bc)
    u_interior = u.trim_boundary()
    self.assertEqual(u_interior, u.array)

  @parameterized.parameters(
      dict(
          shape=(10,),
          bc=boundaries.neumann_boundary_conditions(ndim=1),
          offset=(0.0,)),
      dict(
          shape=(10, 10),
          bc=boundaries.neumann_boundary_conditions(ndim=2),
          offset=(0.0, 0.0)),
      dict(
          shape=(10, 10, 10),
          bc=boundaries.neumann_boundary_conditions(ndim=3),
          offset=(0.0, 0.0, 0.0)),
  )
  def test_interior_consistency_edge_offsets_neumann(self, shape, bc, offset):
    grid = grids.Grid(shape)
    data = np.random.randint(0, 10, shape)
    array = grids.GridArray(data, offset=offset, grid=grid)
    u = grids.GridVariable(array, bc)
    u_interior = u.trim_boundary()
    self.assertEqual(u_interior.offset, u.array.offset)
    self.assertEqual(u_interior.grid.ndim, u.array.grid.ndim)
    self.assertEqual(u_interior.grid.step, u.array.grid.step)

  @parameterized.parameters(
      dict(
          shape=(10,),
          bc=boundaries.dirichlet_boundary_conditions(ndim=1),
          offset=(0.0,)),
      dict(
          shape=(10, 10),
          bc=boundaries.dirichlet_boundary_conditions(ndim=2),
          offset=(0.0, 0.0)),
      dict(
          shape=(10, 10, 10),
          bc=boundaries.dirichlet_boundary_conditions(ndim=3),
          offset=(0.0, 0.0, 0.0)),
  )
  def test_interior_consistency_edge_offsets_dirichlet(self, shape, bc, offset):
    grid = grids.Grid(shape)
    data = np.random.randint(0, 10, shape)
    array = grids.GridArray(data, offset=offset, grid=grid)
    u = grids.GridVariable(array, bc)
    u_interior = u.trim_boundary()
    self.assertEqual(u_interior.offset,
                     tuple(offset + 1 for offset in u.array.offset))
    self.assertEqual(u_interior.grid.ndim, u.array.grid.ndim)
    self.assertEqual(u_interior.grid.step, u.array.grid.step)

  def test_interior_dirichlet(self):
    data = np.array([
        [11, 12, 13, 14, 15],
        [21, 22, 23, 24, 25],
        [31, 32, 33, 34, 35],
        [41, 42, 43, 44, 45],
    ])

    grid = grids.Grid(shape=(4, 5), domain=((0, 1), (0, 1)))
    bc = boundaries.dirichlet_boundary_conditions(ndim=2)

    with self.subTest('offset=(1, 0.5)'):
      offset = (1., 0.5)
      array = grids.GridArray(data, offset, grid)
      u = grids.GridVariable(array, bc)
      u_interior = u.trim_boundary()
      answer = np.array([[11, 12, 13, 14, 15], [21, 22, 23, 24, 25],
                         [31, 32, 33, 34, 35]])
      self.assertArrayEqual(u_interior.data, answer)
      self.assertEqual(u_interior.offset, offset)
      self.assertEqual(u.grid, grid)

    with self.subTest('offset=(1, 1)'):
      offset = (1., 1.)
      array = grids.GridArray(data, offset, grid)
      u = grids.GridVariable(array, bc)
      u_interior = u.trim_boundary()
      answer = np.array([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
      self.assertArrayEqual(u_interior.data, answer)
      self.assertEqual(u_interior.grid, grid)

    with self.subTest('offset=(0.0, 0.5)'):
      offset = (0., 0.5)
      array = grids.GridArray(data, offset, grid)
      u = grids.GridVariable(array, bc)
      u_interior = u.trim_boundary()
      answer = np.array([[21, 22, 23, 24, 25], [31, 32, 33, 34, 35],
                         [41, 42, 43, 44, 45]])
      self.assertArrayEqual(u_interior.data, answer)
      self.assertEqual(u_interior.grid, grid)

    with self.subTest('offset=(0.0, 0.0)'):
      offset = (0.0, 0.0)
      array = grids.GridArray(data, offset, grid)
      u = grids.GridVariable(array, bc)
      u_interior = u.trim_boundary()
      answer = np.array([[22, 23, 24, 25], [32, 33, 34, 35], [42, 43, 44, 45]])
      self.assertArrayEqual(u_interior.data, answer)
      self.assertEqual(u_interior.grid, grid)

    with self.subTest('offset=(0.5, 0.0)'):
      offset = (0.5, 0.0)
      array = grids.GridArray(data, offset, grid)
      u = grids.GridVariable(array, bc)
      u_interior = u.trim_boundary()
      answer = np.array([[12, 13, 14, 15], [22, 23, 24, 25], [32, 33, 34, 35],
                         [42, 43, 44, 45]])
      self.assertArrayEqual(u_interior.data, answer)
      self.assertEqual(u_interior.grid, grid)

    # this is consistent for all offsets, not just edge and center.
    with self.subTest('offset=(0.25, 0.75)'):
      offset = (0.25, 0.75)
      array = grids.GridArray(data, offset, grid)
      u = grids.GridVariable(array, bc)
      u_interior = u.trim_boundary()
      self.assertArrayEqual(u_interior.data, data)
      self.assertEqual(u_interior.grid, grid)

  @parameterized.parameters(
      dict(
          shape=(10,),
          bc=boundaries.periodic_boundary_conditions(ndim=1),
          padding=(1, 1),
          axis=0,
      ),
      dict(
          shape=(10, 10),
          bc=boundaries.dirichlet_boundary_conditions(ndim=2),
          padding=(2, 1),
          axis=1,
      ),
      dict(
          shape=(10, 10, 10),
          bc=boundaries.neumann_boundary_conditions(ndim=3),
          padding=(0, 2),
          axis=2,
      ),
  )
  def test_shift_pad_trim(self, shape, bc, padding, axis):
    grid = grids.Grid(shape)
    data = np.random.randint(0, 10, shape)
    array = grids.GridArray(data, offset=(0.5,) * len(shape), grid=grid)
    u = grids.GridVariable(array, bc)

    with self.subTest('shift'):
      self.assertArrayEqual(
          u.shift(offset=1, axis=axis), bc.shift(array, 1, axis))

    with self.subTest('raises exception'):
      with self.assertRaisesRegex(ValueError,
                                  'Incompatible dimension between grid and bc'):
        grid = grids.Grid((10,))
        data = np.zeros((10,))
        array = grids.GridArray(data, offset=(0.5,), grid=grid)  # 1D
        bc = boundaries.periodic_boundary_conditions(ndim=2)  # 2D
        grids.GridVariable(array, bc)

  def test_unique_boundary_conditions(self):
    grid = grids.Grid((5,))
    array = grids.GridArray(np.arange(5), offset=(0.5,), grid=grid)
    bc1 = boundaries.periodic_boundary_conditions(grid.ndim)
    bc2 = boundaries.dirichlet_boundary_conditions(grid.ndim)
    x_bc1 = grids.GridVariable(array, bc1)
    y_bc1 = grids.GridVariable(array, bc1)
    z_bc2 = grids.GridVariable(array, bc2)

    bc = grids.unique_boundary_conditions(x_bc1, y_bc1)
    self.assertEqual(bc, bc1)

    with self.assertRaises(grids.InconsistentBoundaryConditionsError):
      grids.unique_boundary_conditions(x_bc1, y_bc1, z_bc2)


class GridArrayTensorTest(test_util.TestCase):

  def test_tensor_transpose(self):
    grid = grids.Grid((5, 5))
    offset = (0.5, 0.5)
    a = grids.GridArray(1 * jnp.ones([5, 5]), offset, grid)
    b = grids.GridArray(2 * jnp.ones([5, 5]), offset, grid)
    c = grids.GridArray(3 * jnp.ones([5, 5]), offset, grid)
    d = grids.GridArray(4 * jnp.ones([5, 5]), offset, grid)
    tensor = grids.GridArrayTensor([[a, b], [c, d]])
    self.assertIsInstance(tensor, np.ndarray)
    transposed_tensor = np.transpose(tensor)
    self.assertAllClose(tensor[0, 1], transposed_tensor[1, 0])


class GridTest(test_util.TestCase):

  def test_constructor_and_attributes(self):
    with self.subTest('1d'):
      grid = grids.Grid((10,))
      self.assertEqual(grid.shape, (10,))
      self.assertEqual(grid.step, (1.0,))
      self.assertEqual(grid.domain, ((0, 10.),))
      self.assertEqual(grid.ndim, 1)
      self.assertEqual(grid.cell_center, (0.5,))
      self.assertEqual(grid.cell_faces, ((1.0,),))

    with self.subTest('1d domain scalar size'):
      grid = grids.Grid((10,), domain=10)
      self.assertEqual(grid.domain, ((0.0, 10.0),))

    with self.subTest('2d'):
      grid = grids.Grid(
          (10, 10),
          step=0.1,
      )
      self.assertEqual(grid.step, (0.1, 0.1))
      self.assertEqual(grid.domain, ((0, 1.0), (0, 1.0)))
      self.assertEqual(grid.ndim, 2)
      self.assertEqual(grid.cell_center, (0.5, 0.5))
      self.assertEqual(grid.cell_faces, ((1.0, 0.5), (0.5, 1.0)))

    with self.subTest('3d'):
      grid = grids.Grid((10, 10, 10), step=(0.1, 0.2, 0.5))
      self.assertEqual(grid.step, (0.1, 0.2, 0.5))
      self.assertEqual(grid.domain, ((0, 1.0), (0, 2.0), (0, 5.0)))
      self.assertEqual(grid.ndim, 3)
      self.assertEqual(grid.cell_center, (0.5, 0.5, 0.5))
      self.assertEqual(grid.cell_faces,
                       ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)))

    with self.subTest('1d domain'):
      grid = grids.Grid((10,), domain=[(-2, 2)])
      self.assertEqual(grid.step, (2 / 5,))
      self.assertEqual(grid.domain, ((-2., 2.),))
      self.assertEqual(grid.ndim, 1)
      self.assertEqual(grid.cell_center, (0.5,))
      self.assertEqual(grid.cell_faces, ((1.0,),))

    with self.subTest('2d domain'):
      grid = grids.Grid((10, 20), domain=[(-2, 2), (0, 3)])
      self.assertEqual(grid.step, (4 / 10, 3 / 20))
      self.assertEqual(grid.domain, ((-2., 2.), (0., 3.)))
      self.assertEqual(grid.ndim, 2)
      self.assertEqual(grid.cell_center, (0.5, 0.5))
      self.assertEqual(grid.cell_faces, ((1.0, 0.5), (0.5, 1.0)))

    with self.subTest('2d periodic'):
      grid = grids.Grid((10, 20), domain=2 * np.pi)
      self.assertEqual(grid.step, (2 * np.pi / 10, 2 * np.pi / 20))
      self.assertEqual(grid.domain, ((0., 2 * np.pi), (0., 2 * np.pi)))
      self.assertEqual(grid.ndim, 2)

    with self.assertRaisesRegex(TypeError, 'cannot provide both'):
      grids.Grid((2,), step=(1.0,), domain=[(0, 2.0)])
    with self.assertRaisesRegex(ValueError, 'length of domain'):
      grids.Grid((2, 3), domain=[(0, 1)])
    with self.assertRaisesRegex(ValueError, 'pairs of numbers'):
      grids.Grid((2,), domain=[(0, 1, 2)])
    with self.assertRaisesRegex(ValueError, 'length of step'):
      grids.Grid((2, 3), step=(1.0,))

  def test_stagger(self):
    grid = grids.Grid((10, 10))
    array_1 = jnp.zeros((10, 10))
    array_2 = jnp.ones((10, 10))
    u, v = grid.stagger((array_1, array_2))
    self.assertEqual(u.offset, (1.0, 0.5))
    self.assertEqual(v.offset, (0.5, 1.0))

  def test_center(self):
    grid = grids.Grid((10, 10))

    with self.subTest('array ndim same as grid'):
      array_1 = jnp.zeros((10, 10))
      array_2 = jnp.zeros((20, 30))
      v = (array_1, array_2)  # tuple is a simple pytree
      v_centered = grid.center(v)
      self.assertLen(v_centered, 2)
      self.assertIsInstance(v_centered[0], grids.GridArray)
      self.assertIsInstance(v_centered[1], grids.GridArray)
      self.assertEqual(v_centered[0].shape, (10, 10))
      self.assertEqual(v_centered[1].shape, (20, 30))
      self.assertEqual(v_centered[0].offset, (0.5, 0.5))
      self.assertEqual(v_centered[1].offset, (0.5, 0.5))

    with self.subTest('array ndim different than grid'):
      # Assigns offset dimension based on grid.ndim
      array_1 = jnp.zeros((10,))
      array_2 = jnp.ones((10, 10, 10))
      v = (array_1, array_2)  # tuple is a simple pytree
      v_centered = grid.center(v)
      self.assertLen(v_centered, 2)
      self.assertIsInstance(v_centered[0], grids.GridArray)
      self.assertIsInstance(v_centered[1], grids.GridArray)
      self.assertEqual(v_centered[0].shape, (10,))
      self.assertEqual(v_centered[1].shape, (10, 10, 10))
      self.assertEqual(v_centered[0].offset, (0.5, 0.5))
      self.assertEqual(v_centered[1].offset, (0.5, 0.5))

  def test_axes_and_mesh(self):
    with self.subTest('1d'):
      grid = grids.Grid((5,), step=0.1)
      axes = grid.axes()
      self.assertLen(axes, 1)
      self.assertAllClose(axes[0], [0.05, 0.15, 0.25, 0.35, 0.45])
      mesh = grid.mesh()
      self.assertLen(mesh, 1)
      self.assertAllClose(axes[0], mesh[0])  # in 1d, mesh matches array

    with self.subTest('1d with offset'):
      grid = grids.Grid((5,), step=0.1)
      axes = grid.axes(offset=(0,))
      self.assertLen(axes, 1)
      self.assertAllClose(axes[0], [0.0, 0.1, 0.2, 0.3, 0.4])
      mesh = grid.mesh(offset=(0,))
      self.assertLen(mesh, 1)
      self.assertAllClose(axes[0], mesh[0])  # in 1d, mesh matches array

    with self.subTest('2d'):
      grid = grids.Grid((4, 6), domain=[(-2, 2), (0, 3)])
      axes = grid.axes()
      self.assertLen(axes, 2)
      self.assertAllClose(axes[0], [-1.5, -0.5, 0.5, 1.5])
      self.assertAllClose(axes[1], [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
      mesh = grid.mesh()
      self.assertLen(mesh, 2)
      self.assertEqual(mesh[0].shape, (4, 6))
      self.assertEqual(mesh[1].shape, (4, 6))
      self.assertAllClose(mesh[0][:, 0], axes[0])
      self.assertAllClose(mesh[1][0, :], axes[1])

    with self.subTest('2d with offset'):
      grid = grids.Grid((4, 6), domain=[(-2, 2), (0, 3)])
      axes = grid.axes(offset=(0, 1))
      self.assertLen(axes, 2)
      self.assertAllClose(axes[0], [-2.0, -1.0, 0.0, 1.0])
      self.assertAllClose(axes[1], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
      mesh = grid.mesh(offset=(0, 1))
      self.assertLen(mesh, 2)
      self.assertEqual(mesh[0].shape, (4, 6))
      self.assertEqual(mesh[1].shape, (4, 6))
      self.assertAllClose(mesh[0][:, 0], axes[0])
      self.assertAllClose(mesh[1][0, :], axes[1])

  @parameterized.parameters(
      dict(
          shape=(10,),
          fn=lambda x: 2 * np.ones_like(x),
          offset=None,
          expected_array=2 * np.ones((10,)),
          expected_offset=(0.5,)),
      dict(
          shape=(10, 10),
          fn=lambda x, y: np.ones_like(x) + np.ones_like(y),
          offset=(1, 0.5),
          expected_array=2 * np.ones((10, 10)),
          expected_offset=(1, 0.5)),
      dict(
          shape=(10, 10, 10),
          fn=lambda x, y, z: np.ones_like(z),
          offset=None,
          expected_array=np.ones((10, 10, 10)),
          expected_offset=(0.5, 0.5, 0.5)),
  )
  def test_eval_on_mesh_default_offset(self, shape, fn, offset, expected_array,
                                       expected_offset):
    grid = grids.Grid(shape, step=0.1)
    expected = grids.GridArray(expected_array, expected_offset, grid)
    actual = grid.eval_on_mesh(fn, offset)
    self.assertArrayEqual(expected, actual)

  def test_spectral_axes(self):
    length = 42.
    shape = (64,)
    grid = grids.Grid(shape, domain=((0, length),))

    xs, = grid.axes()
    fft_xs, = grid.fft_axes()
    fft_xs *= 2 * jnp.pi  # convert ordinal to angular frequencies

    # compare the derivative of the sine function (i.e. cosine) with its
    # derivative computed in frequency-space. Note that this derivative involves
    # the computed frequencies so it can serve as a test.
    angular_freq = 2 * jnp.pi / length
    ys = jnp.sin(angular_freq * xs)
    expected = angular_freq * jnp.cos(angular_freq * xs)
    actual = jnp.fft.ifft(1j * fft_xs * jnp.fft.fft(ys))
    self.assertAllClose(expected, actual, atol=1e-4)

  def test_real_spectral_axes_1d(self):
    length = 42.
    shape = (64,)
    grid = grids.Grid(shape, domain=((0, length),))

    xs, = grid.axes()
    fft_xs, = grid.rfft_axes()
    fft_xs *= 2 * jnp.pi  # convert ordinal to angular frequencies

    # compare the derivative of the sine function (i.e. cosine) with its
    # derivative computed in frequency-space. Note that this derivative involves
    # the computed frequencies so it can serve as a test.
    angular_freq = 2 * jnp.pi / length
    ys = jnp.sin(angular_freq * xs)
    expected = angular_freq * jnp.cos(angular_freq * xs)
    actual = jnp.fft.irfft(1j * fft_xs * jnp.fft.rfft(ys))
    self.assertAllClose(expected, actual, atol=1e-4)

  def test_real_spectral_axes_nd_shape(self):
    dim = 3
    grid_size = 64
    shape = (grid_size,) * dim
    domain = ((0, 2 * jnp.pi),) * dim
    grid = grids.Grid(shape, domain=(domain))

    xs1, xs2, xs3 = grid.rfft_axes()
    self.assertEqual(len(xs1), grid_size)
    self.assertEqual(len(xs2), grid_size)
    self.assertEqual(len(xs3), grid_size // 2 + 1)

  def test_domain_interior_masks(self):
    with self.subTest('1d'):
      grid = grids.Grid((5,))
      expected = [[1, 1, 1, 1, 0]]
      self.assertAllClose(grids.domain_interior_masks(grid), expected)

    with self.subTest('2d'):
      grid = grids.Grid((3, 3))
      expected = ([[1, 1, 1], [1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0],
                                                      [1, 1, 0]])
      self.assertAllClose(grids.domain_interior_masks(grid), expected)

    with self.subTest('3d'):
      grid = grids.Grid((3, 4, 5))
      actual = grids.domain_interior_masks(grid)
      self.assertLen(actual, 3)
      # masks are zero on the outer edge, 1 on the interior
      self.assertAllClose(actual[0][:-1, :, :], 1)
      self.assertAllClose(actual[0][-1, :, :], 0)
      self.assertAllClose(actual[1][:, :-1, :], 1)
      self.assertAllClose(actual[1][:, -1, :], 0)
      self.assertAllClose(actual[2][:, :, :-1], 1)
      self.assertAllClose(actual[2][:, :, -1], 0)


if __name__ == '__main__':
  absltest.main()
