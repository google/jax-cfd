"""Tests for google3.research.simulation.whirl.models.towers."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized

import gin
import haiku as hk
import jax
from jax_cfd.base import test_util
from jax_cfd.ml import towers  # pylint: disable=unused-import


TOWERS = ['towers.forward_tower_factory', 'towers.residual_block_tower_factory']
SCALE_FNS = ['towers.fixed_scale', 'towers.scale_to_range']
NDIMS = [1, 2, 3]
INPUT_CHANNELS = [1, 6]


def test_parameters():
  product = itertools.product(TOWERS, SCALE_FNS, NDIMS, INPUT_CHANNELS)
  parameters = []
  for tower, scale_fn, ndim, input_channels in product:
    name = '_'.join([tower, scale_fn, f'{ndim}D', f'{input_channels}_channels'])
    parameters.append(dict(
        testcase_name=name,
        tower_module=tower,
        scale_fn_module=scale_fn,
        ndim=ndim,
        input_channels=input_channels))
  return parameters


@gin.configurable
def forward_pass_module(
    num_output_channels,
    ndim,
    tower_module=gin.REQUIRED
):
  """Constructs a function that initializes tower and applies it to inputs."""
  def forward_pass(inputs):
    return tower_module(num_output_channels, ndim)(inputs)

  return forward_pass


class TowersTest(test_util.TestCase):
  """Tests towers construction, configuration and composition."""

  def setUp(self):
    """Configures all scale_fns that have gin.REQUIRED values."""
    super().setUp()
    gin.enter_interactive_mode()
    config = '\n'.join([
        'towers.fixed_scale.rescaled_one = 0.3',
        'towers.scale_to_range.min_value = -1.23',
        'towers.scale_to_range.max_value = 1.21'
    ])
    gin.parse_config(config)

  @parameterized.named_parameters(*test_parameters())
  def test_output_shapes(
      self,
      tower_module,
      scale_fn_module,
      ndim,
      input_channels
  ):
    """Tests that towers produce outputs of expected shapes."""
    gin.enter_interactive_mode()
    config = '\n'.join([
        f'forward_pass_module.tower_module = @{tower_module}',
        f'{tower_module}.inputs_scale_fn = @{scale_fn_module}'
    ])
    gin.parse_config(config)

    num_output_channels = 5
    spatial_size = 17
    rng = jax.random.PRNGKey(42)
    inputs = jax.random.uniform(rng, (spatial_size,) * ndim + (input_channels,))

    forward_pass = hk.without_apply_rng(
        hk.transform(forward_pass_module(num_output_channels, ndim)))
    params = forward_pass.init(rng, inputs)
    output = forward_pass.apply(params, inputs)
    expected_output_shape = inputs.shape[:-1] + (num_output_channels,)
    actual_output_shape = output.shape
    self.assertEqual(actual_output_shape, expected_output_shape)


if __name__ == '__main__':
  absltest.main()
