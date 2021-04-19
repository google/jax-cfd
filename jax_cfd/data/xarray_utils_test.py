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
"""Tests for jax_cfd.data.xarray_utils."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax_cfd.base import test_util
from jax_cfd.data import xarray_utils
import numpy as np
import xarray


class XarrayUtilsTest(test_util.TestCase):
  """Tests utility functions interacting with xarray."""

  @parameterized.parameters(
      dict(all_dims=('time', 'x', 'y', 'sample'), state_dims=('x', 'y'),),
      dict(all_dims=('x', 'y', 'z', 'sample'), state_dims=('x', 'z', 'y'),),
      dict(all_dims=('time', 'x'), state_dims=('x'),),
      dict(all_dims=('x', 'sample', 'y'), state_dims=('x', 'y'),),
      dict(all_dims=('x', 'z', 'y'), state_dims=('x', 'y', 'z'),),
  )
  def test_normalize(self, all_dims, state_dims):
    """Tests that `normalize` returns data with expected shapes and norms."""
    shape_key, value_key = jax.random.split(jax.random.PRNGKey(42), 2)
    input_shape = jax.random.randint(shape_key, (len(all_dims),), 1, 4)
    inputs = jax.random.normal(value_key, input_shape)

    non_state_dims = [dim for dim in all_dims if dim not in state_dims]
    get_dim_axis_fn = lambda dim: np.where(np.asarray(all_dims) == dim)[0][0]
    state_axes = [get_dim_axis_fn(dim) for dim in state_dims]

    # to compute expected values we move state dimensions to the first axes,
    # divide by the norm and then reshape back.
    inputs_ordered = np.moveaxis(inputs, state_axes, np.arange(len(state_axes)))
    vec_shape = (-1,) + inputs_ordered.shape[-len(non_state_dims):]
    inputs_vec = np.reshape(inputs_ordered, vec_shape)
    inputs_vec = inputs_vec / np.linalg.norm(inputs_vec, axis=0)
    normalized = np.reshape(inputs_vec, inputs_ordered.shape)
    expected = np.moveaxis(normalized, np.arange(len(state_axes)), state_axes)

    coords = {dim: np.arange(input_shape[i]) for i, dim in enumerate(all_dims)}
    array = xarray.DataArray(inputs, coords, all_dims)
    normalized_array = xarray_utils.normalize(array, state_dims)
    actual = normalized_array.transpose(*all_dims).values
    self.assertAllClose(expected, actual, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
