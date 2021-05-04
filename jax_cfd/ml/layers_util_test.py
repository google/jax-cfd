"""Tests for google3.research.simulation.whirl.layers_util."""
import itertools
from absl.testing import absltest
from absl.testing import parameterized
from jax_cfd.base import test_util
from jax_cfd.ml import layers_util
import numpy as np


FINITE_DIFF = layers_util.Method.FINITE_DIFFERENCE
FINITE_VOL = layers_util.Method.FINITE_VOLUME


def _stencil_id(stencil_coordinates, stencil_sizes):
  """Computes id of in the stencil basis given stencil_coordinates."""
  axes_shifts = [1]
  for stencil_size in stencil_sizes:
    axes_shifts.append(axes_shifts[-1] * stencil_size)
  axes_shifts = np.array(axes_shifts[:-1])
  return np.sum(np.array(stencil_coordinates[::-1]) * axes_shifts)


class HelperFunctionsTest(test_util.TestCase):
  """Tests helper functions in layers_util."""

  def test_exponents_up_to_degree(self):
    exponents_iterator = layers_util._exponents_up_to_degree(2, 2)
    expected_values = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
    actual_values = list(exponents_iterator)
    self.assertEqual(expected_values, actual_values)


class PolynomialAccuracyConstraintsTest(test_util.TestCase):
  """Tests polynomial accuracy constraints utils."""

  @parameterized.parameters(
      dict(
          accuracy_order=1,
          method=FINITE_DIFF,
          expected_a=[[1, 1]],
          expected_b=[1]),
      dict(
          accuracy_order=1,
          method=FINITE_VOL,
          expected_a=[[1, 1]],
          expected_b=[1]),
      dict(
          accuracy_order=2,
          method=FINITE_DIFF,
          expected_a=[[-1 / 2, 1 / 2], [1, 1]],
          expected_b=[0, 1]),
      dict(
          accuracy_order=2,
          method=FINITE_VOL,
          expected_a=[[-1 / 2, 1 / 2], [1, 1]],
          expected_b=[0, 1]),
  )
  def test_constraints_1d(self, accuracy_order, method, expected_a, expected_b):
    a, b = layers_util.polynomial_accuracy_constraints(
        [np.array([-0.5, 0.5])], method, derivative_orders=[0],
        accuracy_order=accuracy_order, grid_step=1.0)
    np.testing.assert_allclose(a, expected_a)
    np.testing.assert_allclose(b, expected_b)

  def test_constraints_2d_second_order_zeroth_derivative(self):
    # these constraints should be under-determined.
    stencils = [np.array([-0.5, 0.5])] * 2
    a, b = layers_util.polynomial_accuracy_constraints(
        stencils,
        FINITE_DIFF,
        derivative_orders=[0, 0],
        accuracy_order=2)
    # three constraints, for each term in [1, x, y]
    self.assertEqual(a.shape, (3, 4))
    self.assertEqual(b.shape, (3,))
    # explicitly test two valid solutions
    np.testing.assert_allclose(a.dot([1 / 4, 1 / 4, 1 / 4, 1 / 4]), b)
    np.testing.assert_allclose(a.dot([4 / 10, 1 / 10, 1 / 10, 4 / 10]), b)

  def test_constraints_2d_first_order_first_derivative(self):
    stencils = [np.array([-1, 0, 1])] * 2
    a, b = layers_util.polynomial_accuracy_constraints(
        stencils,
        FINITE_DIFF,
        derivative_orders=[1, 0],
        accuracy_order=1)
    # three constraints, for each term in [1, x, y]
    self.assertEqual(a.shape, (3, 9))
    self.assertEqual(b.shape, (3,))
    # explicitly test a valid solution
    solution = np.array([-1, 0, -1, 0, 0, 0, 1, 0, 1]) / 4
    np.testing.assert_allclose(a.dot(solution), b)
    # explicitly test an invalid solution.
    # this solution is invalid because the stencil is a linear combination
    # of derivatives in both the x and y directions.
    non_solution = np.array([-1, 0, -1, 1, 0, -1, 1, 0, 1]) / 4
    self.assertGreater(np.linalg.norm(a.dot(non_solution) - b), 0.1)

  def test_constraints_2d_first_order_second_derivative(self):
    stencils = [np.array([-1, 0, 1])] * 2
    a, b = layers_util.polynomial_accuracy_constraints(
        stencils,
        FINITE_DIFF,
        derivative_orders=[1, 1],
        accuracy_order=1)
    # six constraints, for each term in [1, x, y, x^2, xy, y^2]
    self.assertEqual(a.shape, (6, 9))
    self.assertEqual(b.shape, (6,))
    # explicitly test a valid solution
    solution = np.array([1, 0, -1, 0, 0, 0, -1, 0, 1]) / 4
    np.testing.assert_allclose(a.dot(solution), b)

  def test_constraints_3d_second_order_zeroth_derivative(self):
    stencils = [np.array([-1, 0, 1])] * 3
    a, b = layers_util.polynomial_accuracy_constraints(
        stencils,
        FINITE_DIFF,
        derivative_orders=[0, 0, 0],
        accuracy_order=2)
    # four constraints, for each term in [1, x, y, z]
    self.assertEqual(a.shape, (4, 27))
    self.assertEqual(b.shape, (4,))
    # explicitly test a valid solution
    stencil = list(itertools.product(*stencils))
    solution_a = np.zeros(27)
    solution_a[stencil.index((0, 0, 0))] = 1.
    solution_b = np.zeros(27)
    solution_b[stencil.index((0, -1, 0))] = 0.25
    solution_b[stencil.index((0, 0, 0))] = 0.5
    solution_b[stencil.index((0, 1, 0))] = 0.25
    np.testing.assert_allclose(a.dot(solution_a), b)
    np.testing.assert_allclose(a.dot(solution_b), b)

  def test_constraints_3d_second_order_first_derivative(self):
    stencils = [np.array([-1, 0, 1])] * 3
    a, b = layers_util.polynomial_accuracy_constraints(
        stencils,
        FINITE_DIFF,
        derivative_orders=[0, 0, 1],
        accuracy_order=2)
    # ten constraints, for each term in [1, (3 choose 1), (3 choose 2)]
    self.assertEqual(a.shape, (10, 27))
    self.assertEqual(b.shape, (10,))
    # explicitly test a few valid solutions
    solution = np.zeros(27)
    solution[_stencil_id([1, 1, 0], [3] * 3)] = -0.5
    solution[_stencil_id([1, 1, 2], [3] * 3)] = 0.5
    np.testing.assert_allclose(a.dot(solution), b)


class PolynomialAccuracyCoefficientsTests(test_util.TestCase):

  # For test-cases, see
  # https://en.wikipedia.org/wiki/Finite_difference_coefficient
  @parameterized.parameters(
      dict(stencil=[-1, 0, 1], derivative_order=1, expected=[-1 / 2, 0, 1 / 2]),
      dict(stencil=[-1, 0, 1], derivative_order=2, expected=[1, -2, 1]),
      dict(
          stencil=[-2, -1, 0, 1, 2],
          derivative_order=2,
          expected=[-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
      dict(
          stencil=[-2, -1, 0, 1, 2],
          derivative_order=2,
          accuracy_order=1,
          expected=[0, 1, -2, 1, 0]),
      dict(stencil=[0, 1], derivative_order=1, expected=[-1, 1]),
      dict(stencil=[0, 2], derivative_order=1, expected=[-0.5, 0.5]),
      dict(stencil=[0, 0.5], derivative_order=1, expected=[-2, 2]),
      dict(
          stencil=[0, 1, 2, 3, 4],
          derivative_order=4,
          expected=[1, -4, 6, -4, 1]),
  )
  def test_finite_difference_coefficients_1d(
      self,
      stencil,
      derivative_order,
      expected,
      accuracy_order=None
  ):
    result = layers_util.polynomial_accuracy_coefficients(
        [np.array(stencil)], FINITE_DIFF, [derivative_order], accuracy_order)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      dict(
          stencils=[[-0.5, 0.5], [-0.5, 0.5]],
          derivative_orders=[0, 0],
          expected=[[0.25, 0.25], [0.25, 0.25]]),
      dict(
          stencils=[[-0.5, 0.5], [-0.5, 0.5]],
          derivative_orders=[0, 1],
          expected=[[-0.5, 0.5], [-0.5, 0.5]]),
      dict(
          stencils=[[-0.5, 0.5], [-0.5, 0.5]],
          derivative_orders=[1, 1],
          expected=[[1, -1], [-1, 1]]),
      dict(
          stencils=[[-1, 0, 1], [-0.5, 0.5]],
          derivative_orders=[1, 0],
          expected=[[-0.25, -0.25], [0, 0], [0.25, 0.25]]),
  )
  def test_finite_difference_coefficients_2d(self, stencils, derivative_orders,
                                             expected):
    args = ([np.array(s) for s in stencils], FINITE_DIFF, derivative_orders)
    result = layers_util.polynomial_accuracy_coefficients(*args)
    np.testing.assert_allclose(result, expected)

    result = layers_util.polynomial_accuracy_coefficients(
        *args, accuracy_order=1)
    np.testing.assert_allclose(result, expected)

  def test_finite_difference_coefficients_3d(self):
    stencils = [[-0.5, 0.5] for _ in range(3)]
    derivative_orders = [0, 0, 0]
    expected = [
        [[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]
    ]
    args = ([np.array(s) for s in stencils], FINITE_DIFF, derivative_orders)
    result = layers_util.polynomial_accuracy_coefficients(*args)
    np.testing.assert_allclose(result, expected)

    result = layers_util.polynomial_accuracy_coefficients(
        *args, accuracy_order=1)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      dict(stencil=[-0.5, 0.5], derivative_order=0, expected=[1 / 2, 1 / 2]),
      dict(
          stencil=[-1.5, -0.5, 0.5, 1.5],
          derivative_order=0,
          accuracy_order=1,
          expected=[0, 1 / 2, 1 / 2, 0]),
      dict(stencil=[-1, 1], derivative_order=0, expected=[1 / 2, 1 / 2]),
      dict(stencil=[-1.5, -0.5], derivative_order=0, expected=[-1 / 2, 3 / 2]),
      dict(stencil=[-0.5, 0.5, 1.5],
           derivative_order=0,
           expected=[1 / 3, 5 / 6, -1 / 6]),
      dict(stencil=[-0.5, 0.5], derivative_order=1, expected=[-1, 1]),
      dict(stencil=[-1, 1], derivative_order=1, expected=[-1 / 2, 1 / 2]),
      dict(stencil=[-1, 0, 1], derivative_order=1, expected=[-1 / 2, 0, 1 / 2]),
      dict(stencil=[0.5, 1.5, 2.5], derivative_order=1, expected=[-2, 3, -1]),
      dict(
          stencil=[-1.5, -0.5, 0.5, 1.5],
          derivative_order=1,
          expected=[1 / 12, -5 / 4, 5 / 4, -1 / 12]),
      dict(
          stencil=[-.75, -0.25, 0.25, 0.75],
          derivative_order=1,
          expected=[1 / 6, -5 / 2, 5 / 2, -1 / 6]),
  )
  def test_finite_volume_coefficients_1d(self,
                                         stencil,
                                         derivative_order,
                                         expected,
                                         accuracy_order=None):
    step = stencil[1] - stencil[0]
    result = layers_util.polynomial_accuracy_coefficients(
        [np.array(stencil)], FINITE_VOL, [derivative_order], accuracy_order,
        grid_step=step)
    np.testing.assert_allclose(result, expected)


class ExtractPatchesTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_1D_even',
           x=np.arange(10, dtype=np.float32).reshape([10, 1]),
           patch_shape=(4,),
           indices_and_patches=(
               ((0,), np.array([9., 0., 1., 2.])),
               ((5,), np.array([4., 5., 6., 7.])))),
      dict(testcase_name='_1D_odd',
           x=np.arange(10, dtype=np.float32).reshape([10, 1]),
           patch_shape=(5,),
           indices_and_patches=(
               ((0,), np.array([8., 9., 0., 1., 2.])),
               ((5,), np.array([3., 4., 5., 6., 7.])))),
      dict(testcase_name='_2D_even',
           x=np.arange(16, dtype=np.float32).reshape([4, 4, 1]),
           patch_shape=(2, 2),
           indices_and_patches=(
               ((0, 0), np.array([0., 1., 4., 5.])),
               ((1, 1), np.array([5, 6, 9, 10])))),
      dict(testcase_name='_2D_odd',
           x=np.arange(16, dtype=np.float32).reshape([4, 4, 1]),
           patch_shape=(3, 3),
           indices_and_patches=(
               ((0, 0), np.array([15., 12., 13., 3., 0., 1., 7., 4., 5.])),
               ((1, 1), np.array([0., 1., 2., 4., 5., 6., 8., 9., 10.])))),
      dict(testcase_name='_3D_even',
           x=np.arange(125, dtype=np.float32).reshape([5, 5, 5, 1]),
           patch_shape=(2, 2, 2),
           indices_and_patches=(
               ((0, 0, 0),
                np.array([0., 1., 5., 6., 25., 26., 30., 31.])),
               ((3, 3, 3),
                np.array([93., 94., 98., 99., 118., 119., 123., 124.])))),
      dict(testcase_name='_3D_odd',
           x=np.arange(125, dtype=np.float32).reshape([5, 5, 5, 1]),
           patch_shape=(3, 3, 3),
           indices_and_patches=(
               ((0, 0, 0),
                np.array(
                    [124., 120., 121., 104., 100., 101., 109., 105., 106.,
                     24., 20., 21., 4., 0., 1., 9., 5., 6.,
                     49., 45., 46., 29., 25., 26., 34., 30., 31.])),
               ((3, 3, 3),
                np.array(
                    [62., 63., 64., 67., 68., 69., 72., 73., 74.,
                     87., 88., 89., 92., 93., 94., 97., 98., 99.,
                     112., 113., 114., 117., 118., 119., 122., 123., 124.])))),

  )
  def test_extract_patches(self, x, patch_shape, indices_and_patches):
    """Tests `layers_util.extract_patches`."""
    for method in ('roll', 'conv'):
      with self.subTest(f'method_{method}'):
        patches = layers_util.extract_patches(x, patch_shape, method=method)
        for idx, expected_patch in indices_and_patches:
          actual_patch = patches[idx]
          np.testing.assert_allclose(actual_patch, expected_patch)


if __name__ == '__main__':
  absltest.main()
