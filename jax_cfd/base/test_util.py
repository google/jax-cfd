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

"""Shared test utilities."""

from absl.testing import parameterized

from jax.config import config
from jax_cfd.base import grids
import numpy as np

config.parse_flags_with_absl()


class TestCase(parameterized.TestCase):
  """TestCase with assertions for arrays and grids.AlignedArray."""

  def _check_and_remove_alignment_and_grid(self, *arrays):
    """Check that array-like data values and other attributes match.

    If args type is GridArray, verify their offsets and grids match.
    If args type is GridVariable, verify their offsets, grids, and bc match.

    Args:
      *arrays: one or more Array, GridArray or GridVariable, but they all be the
        same type.

    Returns:
      The data-only arrays, with other attributes removed.
    """
    is_gridarray = [isinstance(array, grids.GridArray) for array in arrays]
    # GridArray
    if any(is_gridarray):
      self.assertTrue(
          all(is_gridarray), msg=f'arrays have mixed types: {arrays}')
      try:
        grids.consistent_offset(*arrays)
      except grids.InconsistentOffsetError as e:
        raise AssertionError(str(e)) from None
      try:
        grids.consistent_grid(*arrays)
      except grids.InconsistentGridError as e:
        raise AssertionError(str(e)) from None
      arrays = tuple(array.data for array in arrays)
    # GridVariable
    is_gridvariable = [
        isinstance(array, grids.GridVariable) for array in arrays
    ]
    if any(is_gridvariable):
      self.assertTrue(
          all(is_gridvariable), msg=f'arrays have mixed types: {arrays}')
      try:
        grids.consistent_offset(*arrays)
      except grids.InconsistentOffsetError as e:
        raise AssertionError(str(e)) from None
      try:
        grids.consistent_grid(*arrays)
      except grids.InconsistentGridError as e:
        raise AssertionError(str(e)) from None
      try:
        grids.unique_boundary_conditions(*arrays)
      except grids.InconsistentBoundaryConditionsError as e:
        raise AssertionError(str(e)) from None
      arrays = tuple(array.array.data for array in arrays)
    return arrays

  # pylint: disable=unbalanced-tuple-unpacking
  def assertArrayEqual(self, expected, actual, **kwargs):
    expected, actual = self._check_and_remove_alignment_and_grid(
        expected, actual)
    np.testing.assert_array_equal(expected, actual, **kwargs)

  def assertAllClose(self, expected, actual, **kwargs):
    expected, actual = self._check_and_remove_alignment_and_grid(
        expected, actual)
    np.testing.assert_allclose(expected, actual, **kwargs)

  # pylint: enable=unbalanced-tuple-unpacking

  