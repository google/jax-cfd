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

from jax_cfd.base import resize
from jax_cfd.base import test_util
import numpy as np


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
    self.assertAllClose(expected, actual, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
