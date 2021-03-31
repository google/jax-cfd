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

"""Tests for jax_cfd.funcutils."""

from absl.testing import absltest
from absl.testing import parameterized
from jax_cfd.base import funcutils
from jax_cfd.base import test_util
import numpy as np


class TrajectoryTests(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='identity',
           trajectory_length=6,
           post_process=funcutils._identity),
      dict(testcase_name='squared_postprocessing',
           trajectory_length=12,
           post_process=lambda x: (x[0] ** 2,)),
  )
  def test_trajectory(self, trajectory_length, post_process):
    def step_fn(x):
      return (x[0] + 1,)

    trajectory_fn = funcutils.trajectory(
        step_fn, trajectory_length, post_process)

    initial_state = (2 * np.ones(1),)
    expected_frames = []
    frame = initial_state
    for _ in range(trajectory_length):
      frame = step_fn(frame)
      expected_frames.append(post_process(frame))
    expected_output = (np.stack([x[0] for x in expected_frames]),)
    _, actual_output = trajectory_fn(initial_state)
    for expected, actual in zip(expected_output, actual_output):
      self.assertAllClose(expected, actual, atol=1e-9)


if __name__ == '__main__':
  absltest.main()
