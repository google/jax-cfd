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
"""Tests for jax_cfd.data.visualization."""

import os.path

from absl.testing import absltest
from jax_cfd.base import test_util
from jax_cfd.data import visualization
import numpy as np


class VisualizationTest(test_util.TestCase):

  def test_trajectory_to_images_shape(self):
    """Tests that trajectory_to_images generates a list of images."""
    trajectory = np.random.uniform(size=(25, 32, 48))
    list_of_images = visualization.trajectory_to_images(trajectory)
    self.assertEqual(len(list_of_images), trajectory.shape[0])
    self.assertEqual(list_of_images[0].size, (48, 32))

    list_of_images = visualization.trajectory_to_images(
        trajectory, longest_side=96)
    self.assertEqual(len(list_of_images), trajectory.shape[0])
    self.assertEqual(list_of_images[0].size, (96, 64))

  def test_horizontal_facet_shape(self):
    """Tests that horizontal_facet generates images of expected size."""
    trajectory_a = np.random.uniform(size=(25, 32, 32))
    trajectory_b = np.random.uniform(size=(25, 32, 32))
    relative_separation_width = 0.25
    list_of_images_a = visualization.trajectory_to_images(trajectory_a)
    list_of_images_b = visualization.trajectory_to_images(trajectory_b)
    list_of_images_facet = visualization.horizontal_facet(
        [list_of_images_a, list_of_images_b], relative_separation_width)
    actual_width = list_of_images_facet[0].width
    expected_width = 32 * 2 + int(32 * relative_separation_width)
    self.assertEqual(actual_width, expected_width)

  def test_save_movie_local(self):
    """Tests that save_movie write gif to a file."""
    temp_dir = self.create_tempdir()
    temp_filename = os.path.join(temp_dir, 'tmp_file.gif')
    input_trajectory = np.random.uniform(size=(25, 32, 32))
    images = visualization.trajectory_to_images(input_trajectory)
    visualization.save_movie(images, temp_filename)
    self.assertTrue(os.path.exists(temp_filename))


if __name__ == '__main__':
  absltest.main()
