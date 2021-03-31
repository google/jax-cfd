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

  def _check_and_remove_alignment(self, *arrays):
    """If arrays are aligned, verify their offsets match."""
    is_aligned = [isinstance(array, grids.AlignedArray) for array in arrays]
    if any(is_aligned):
      self.assertTrue(all(is_aligned), msg=f'arrays have mixed types: {arrays}')
      try:
        grids.aligned_offset(*arrays)
      except grids.AlignmentError as e:
        raise AssertionError(str(e)) from None
      arrays = tuple(array.data for array in arrays)
    return arrays

  # pylint: disable=unbalanced-tuple-unpacking
  def assertArrayEqual(self, expected, actual, **kwargs):
    expected, actual = self._check_and_remove_alignment(expected, actual)
    np.testing.assert_array_equal(expected, actual, **kwargs)

  def assertAllClose(self, expected, actual, **kwargs):
    expected, actual = self._check_and_remove_alignment(expected, actual)
    np.testing.assert_allclose(expected, actual, **kwargs)
  # pylint: enable=unbalanced-tuple-unpacking
