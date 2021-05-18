"""Tests for google3.research.simulation.whirl.layers."""
import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.base import test_util
from jax_cfd.ml import layers
from jax_cfd.ml import layers_util
import numpy as np


KERNEL_SIZES = [3, 4]
ROLL_BY = [2, 4, 7]
RATES = [1, 2]


def conv_test_parameters(conv_modules, ndim, test_rate=True):
  if test_rate:
    product = itertools.product(conv_modules, KERNEL_SIZES, ROLL_BY, RATES)
  else:
    product = itertools.product(conv_modules, KERNEL_SIZES, ROLL_BY, [None])
  parameters = []
  for conv_module, kernel_size, roll_by, rate in product:
    name = '_'.join([
        conv_module.__name__, f'kernel_size_{kernel_size}',
        f'rollby_{roll_by}', f'rate_{rate}'])
    parameters.append(dict(
        testcase_name=name,
        conv_module=conv_module,
        kernel_shape=(kernel_size,) * ndim,
        roll_by=roll_by,
        rate=rate))
  return parameters


class ConvPeriodicTest(test_util.TestCase):
  """Tests all convolutions with periodic boundary conditions."""

  @parameterized.named_parameters(
      *(conv_test_parameters([layers.PeriodicConvTranspose1D], 1, False) +
        conv_test_parameters([layers.PeriodicConv1D], 1, True)))
  def test_equivariance_1d(self, conv_module, kernel_shape, roll_by, rate):
    input_shape = (32, 1)
    inputs = np.random.uniform(size=input_shape)

    def net_forward(x):
      if rate is not None:
        module = conv_module(1, kernel_shape, rate=rate)
      else:
        module = conv_module(1, kernel_shape)
      return module(x)

    net = hk.without_apply_rng(hk.transform(net_forward))
    rng = hk.PRNGSequence(42)
    net_params = net.init(next(rng), inputs)
    roll_conv = jnp.roll(net.apply(net_params, inputs), roll_by, 0)
    conv_roll = net.apply(net_params, jnp.roll(inputs, roll_by, 0))
    np.testing.assert_allclose(roll_conv, conv_roll)

  @parameterized.named_parameters(
      *(conv_test_parameters([layers.PeriodicConvTranspose2D], 2, False) +
        conv_test_parameters([layers.PeriodicConv2D], 2, True)))
  def test_equivariance_2d(self, conv_module, kernel_shape, roll_by, rate):
    input_shape = (32, 32, 1)
    inputs = np.random.uniform(size=input_shape)

    def net_forward(x):
      if rate is not None:
        module = conv_module(1, kernel_shape, rate=rate)
      else:
        module = conv_module(1, kernel_shape)
      return module(x)

    net = hk.without_apply_rng(hk.transform(net_forward))
    rng = hk.PRNGSequence(42)
    net_params = net.init(next(rng), inputs)
    roll_conv_x = jnp.roll(net.apply(net_params, inputs), roll_by, 0)
    conv_roll_x = net.apply(net_params, jnp.roll(inputs, roll_by, 0))
    roll_conv_y = jnp.roll(net.apply(net_params, inputs), roll_by, 1)
    conv_roll_y = net.apply(net_params, jnp.roll(inputs, roll_by, 1))
    np.testing.assert_allclose(roll_conv_x, conv_roll_x)
    np.testing.assert_allclose(roll_conv_y, conv_roll_y)

  @parameterized.named_parameters(
      *(conv_test_parameters([layers.PeriodicConvTranspose3D], 3, False) +
        conv_test_parameters([layers.PeriodicConv3D], 3, True)))
  def test_equivariance_3d(self, conv_module, kernel_shape, roll_by, rate):
    input_shape = (16, 16, 16, 1)
    inputs = np.random.uniform(size=input_shape)

    def net_forward(x):
      if rate is not None:
        module = conv_module(1, kernel_shape, rate=rate)
      else:
        module = conv_module(1, kernel_shape)
      return module(x)

    net = hk.without_apply_rng(hk.transform(net_forward))
    rng = hk.PRNGSequence(42)
    net_params = net.init(next(rng), inputs)
    roll_conv_x = jnp.roll(net.apply(net_params, inputs), roll_by, 0)
    conv_roll_x = net.apply(net_params, jnp.roll(inputs, roll_by, 0))
    roll_conv_y = jnp.roll(net.apply(net_params, inputs), roll_by, 1)
    conv_roll_y = net.apply(net_params, jnp.roll(inputs, roll_by, 1))
    roll_conv_z = jnp.roll(net.apply(net_params, inputs), roll_by, 2)
    conv_roll_z = net.apply(net_params, jnp.roll(inputs, roll_by, 2))
    np.testing.assert_allclose(roll_conv_x, conv_roll_x)
    np.testing.assert_allclose(roll_conv_y, conv_roll_y)
    np.testing.assert_allclose(roll_conv_z, conv_roll_z)

  @parameterized.named_parameters(
      {
          'testcase_name': '1d',
          'conv_module': layers.PeriodicConv1D,
          'input_shape': (32, 3),
          'kwargs': dict(output_channels=3, kernel_shape=(5,)),
          'tile_layout': (4,),
      },
      {
          'testcase_name': '2d_3x3',
          'conv_module': layers.PeriodicConv2D,
          'input_shape': (8, 8, 3),
          'kwargs': dict(output_channels=3, kernel_shape=(3, 3)),
          'tile_layout': (2, 4),
      },
      {
          'testcase_name': '2d_4x4',
          'conv_module': layers.PeriodicConv2D,
          'input_shape': (8, 8, 3),
          'kwargs': dict(output_channels=3, kernel_shape=(4, 4)),
          'tile_layout': (2, 1),
      },
      {
          'testcase_name': '3d',
          'conv_module': layers.PeriodicConv3D,
          'input_shape': (8, 8, 8, 3),
          'kwargs': dict(output_channels=3, kernel_shape=(3, 3, 3)),
          'tile_layout': (2, 4, 4),
      },
  )
  def test_tile_layout(self, conv_module, input_shape, kwargs, tile_layout):
    # pylint: disable=unnecessary-lambda
    inputs = np.random.uniform(size=input_shape)

    untiled_layout = (1,) * len(tile_layout)

    base_module = lambda x: conv_module(**kwargs, tile_layout=untiled_layout)(x)
    base_net = hk.without_apply_rng(hk.transform(base_module))

    tiled_module = lambda x: conv_module(**kwargs, tile_layout=tile_layout)(x)
    tiled_net = hk.without_apply_rng(hk.transform(tiled_module))

    params = base_net.init(jax.random.PRNGKey(42), inputs)

    base_out = base_net.apply(params, inputs)
    tiled_out = tiled_net.apply(params, inputs)

    np.testing.assert_allclose(base_out, tiled_out, atol=1e-6)

  @parameterized.named_parameters([
      ('size_60_stride_1', 60, 1),
      ('size_60_stride_2', 60, 2),
      ('size_60_stride_3', 60, 3),
      ('size_60_stride_5', 60, 5),
      ('size_45_stride_3', 45, 3),
  ])
  def test_roundtrip_spatial_alignment(self, input_size, stride):
    """Tests that ConvTansposed(Conv(x)) with identity params is identity op."""
    input_shape = (input_size, 1)
    inputs = np.random.uniform(size=input_shape)
    w_init = hk.initializers.Constant(
        np.reshape(np.asarray([0., 1., 0.]), (3, 1, 1)))
    b_init = hk.initializers.Constant(np.zeros((1,)))

    def net_forward(x):
      conv_args = {
          'output_channels': 1,
          'kernel_shape': (3,),
          'stride': stride,
          'w_init': w_init,
          'b_init': b_init,
      }
      conv = layers.PeriodicConv1D(**conv_args)
      conv_transpose = layers.PeriodicConvTranspose1D(**conv_args)
      return conv_transpose(conv(x))

    net = hk.without_apply_rng(hk.transform(net_forward))
    rng = hk.PRNGSequence(42)
    net_params = net.init(next(rng), inputs)
    output = net.apply(net_params, inputs)
    stride_mask = np.expand_dims(np.asarray([1] + [0] * (stride -1)), -1)
    mask = np.tile(stride_mask, (input_shape[0] // stride, 1))
    expected_output = inputs * mask
    np.testing.assert_allclose(output, expected_output)


class RescaleToRangeTest(test_util.TestCase):
  """Tests `rescale_to_range` layer."""

  @parameterized.named_parameters([
      ('rescale_1d', (32,)),
      ('rescale_2d', (16, 24)),
      ('rescale_3d', (8, 16, 16)),
  ])
  def test_min_max_shape(self, shape):
    """Tests that rescaled values have expected shapes and min/max values."""
    min_value = -0.4
    max_value = 0.73
    axes = tuple(np.arange(len(shape)))
    input_values = np.random.uniform(low=-5., high=5., size=shape)
    rescale = functools.partial(
        layers.rescale_to_range, min_value=min_value, max_value=max_value,
        axes=axes)
    output = rescale(input_values)
    actual_max = jnp.max(output)
    actual_min = jnp.min(output)
    self.assertEqual(shape, output.shape)  # shape shouldn't change
    self.assertAllClose(min_value, actual_min)
    self.assertAllClose(max_value, actual_max)

  @parameterized.named_parameters([
      ('rescale_1d', (32,)),
      ('rescale_2d', (16, 24)),
      ('rescale_3d', (8, 16, 16)),
  ])
  def test_correctness(self, shape):
    """Tests that rescaled values belong to expected range."""
    min_value = 0.
    max_value = 1.
    axes = tuple(np.arange(len(shape)))
    num_elements = np.prod(shape)
    input_values = np.random.uniform(low=-5., high=5., size=num_elements)
    input_values[0] = -10.
    input_values[-1] = 10.
    input_values = np.reshape(input_values, newshape=shape)
    rescale = functools.partial(
        layers.rescale_to_range, min_value=min_value, max_value=max_value,
        axes=axes)
    # we can also not even call net_init, since no parameters are needed.
    actual_output = rescale(input_values)
    expected_output = (input_values + 10.) / 20.
    self.assertAllClose(expected_output, actual_output)


def _name_test(ndim, stencil, derivs):
  return '{}d_stencil_{}_derivatives_{}'.format(ndim, stencil, derivs)


TESTS_1D = [
    (_name_test(1, stencil, derivs), (32,), stencil, derivs)
    for stencil, derivs in itertools.product([[3], [4]], [[0], [1]])
]


TESTS_2D = [
    (_name_test(2, stencil, derivs), (32, 32), stencil, derivs)
    for stencil, derivs in itertools.product([[3, 3], [4, 4]], [[0, 0], [1, 1]])
]


TESTS_3D = [
    (_name_test(3, stencil, derivs), (16, 16, 16), stencil, derivs)
    for stencil, derivs in itertools.product([[3, 3, 3], [4, 4, 4]],
                                             [[0, 0, 0], [1, 1, 0]])
]


TESTS_ALL = TESTS_1D + TESTS_2D + TESTS_3D


def _make_test_stencil(size, step):
  return np.array([i * step for i in range(-size // 2 + 1, size // 2 + 1)])


class PolynomialConstraintTest(test_util.TestCase):
  """Tests `PolynomialConstraint` module."""

  @parameterized.named_parameters(
      dict(testcase_name=name,  # pylint: disable=g-complex-comprehension
           grid_shape=shape,
           stencil_sizes=stencil,
           derivative_orders=derivs)
      for name, shape, stencil, derivs in TESTS_ALL
  )
  def test_shapes(self, grid_shape, stencil_sizes, derivative_orders):
    """Tests that PolynomialConstraint returns expected shapes."""
    ndims = len(grid_shape)
    grid_step = 0.1
    steps = (grid_step,) * ndims
    stencils = [_make_test_stencil(size, grid_step) for size in stencil_sizes]
    method = layers_util.Method.FINITE_VOLUME
    module = layers.PolynomialConstraint(
        stencils, derivative_orders, method, steps)
    inputs = np.random.uniform(size=grid_shape + (module.subspace_size,))
    outputs = module(inputs)
    actual_shape = outputs.shape
    expected_shape = grid_shape + (np.prod(stencil_sizes),)
    self.assertEqual(actual_shape, expected_shape)

  @parameterized.named_parameters(
      dict(testcase_name=name,  # pylint: disable=g-complex-comprehension
           grid_shape=shape,
           stencil_sizes=stencil,
           derivative_orders=derivs)
      for name, shape, stencil, derivs in TESTS_ALL
  )
  def test_values(self, grid_shape, stencil_sizes, derivative_orders):
    """Tests that result of PolynomialConstraint satisfies poly-constraints."""
    ndims = len(grid_shape)
    grid_step = 0.1
    steps = (grid_step,) * ndims
    stencils = [_make_test_stencil(size, grid_step) for size in stencil_sizes]
    method = layers_util.Method.FINITE_VOLUME
    module = layers.PolynomialConstraint(
        stencils, derivative_orders, method, steps)
    inputs = np.random.uniform(size=grid_shape + (module.subspace_size,))
    outputs = module(inputs)
    a, b = layers_util.polynomial_accuracy_constraints(
        stencils, method, derivative_orders, 1, grid_step)
    violation = jnp.transpose(jnp.tensordot(a, outputs, axes=[-1, -1])) - b
    np.testing.assert_allclose(jnp.max(violation), 0., atol=1e-2)


def _tower_factory(num_output_channels, ndims, conv_block):
  rescale_01 = functools.partial(layers.rescale_to_range, min_value=0.,
                                 max_value=1., axes=list(range(ndims)))
  components = [rescale_01]
  for output_channels, kernel_shape in zip([4], [[3]* ndims]):
    components.append(conv_block(output_channels, kernel_shape))
    components.append(jax.nn.relu)
  components.append(conv_block(num_output_channels, [3] * ndims))
  return hk.Sequential(components, name='tower')


class StencilCoefficientsTest(test_util.TestCase):
  """Tests StencilCoefficients module."""

  @parameterized.named_parameters([
      ('1d', (32,), (1,), (4,), layers.PeriodicConv1D, 1),
      ('2d', (16, 21), (0, 1), (5, 5), layers.PeriodicConv2D, 2),
      ('3d', (8, 8, 6), (0, 0, 0), (2, 2, 2), layers.PeriodicConv3D, 3),
  ])
  def test_output_shape(self, input_shape, derivative_orders, stencil_sizes,
                        conv_block, ndims):
    grid_step = 0.1
    steps = (grid_step,) * ndims
    stencils = [_make_test_stencil(size, grid_step) for size in stencil_sizes]
    tower_factory = functools.partial(_tower_factory, ndims=ndims,
                                      conv_block=conv_block)

    def compute_coefficients(inputs):
      net = layers.StencilCoefficients(
          stencils, derivative_orders, tower_factory, steps)
      return net(inputs)

    coefficients_model = hk.without_apply_rng(
        hk.transform(compute_coefficients))
    rng = hk.PRNGSequence(42)
    inputs = np.random.uniform(size=input_shape + (1,))
    params = coefficients_model.init(next(rng), inputs)
    outputs = coefficients_model.apply(params, inputs)
    actual_shape = outputs.shape
    expected_shape = input_shape + (np.prod(stencil_sizes),)
    self.assertEqual(actual_shape, expected_shape)


class SpatialDerivativeFromLogitsTest(test_util.TestCase):
  """Tests SpatialDerivativeFromLogits module."""

  @parameterized.named_parameters(
      dict(testcase_name='1D',
           derivative_orders=(0,),
           input_shape=(256,),
           input_offset=(.5,),
           target_offset=(1,),
           steps=(1,),
           stencil_shape=(5,),
           tile_layout=(4,)),
      dict(testcase_name='2D',
           derivative_orders=(0, 1),
           input_shape=(64, 64),
           input_offset=(0, 0),
           target_offset=(10, 0),
           steps=(.1, .1),
           stencil_shape=(4, 4,),
           tile_layout=None),
      dict(testcase_name='3D',
           derivative_orders=(0, 1, 0),
           input_shape=(32, 64, 16),
           input_offset=(.5, .5, .5),
           target_offset=(0, 0, 0),
           steps=(3, 3, 3),
           stencil_shape=(3, 3, 3),
           tile_layout=(8, 8, 8)),
  )
  def test_shape(self, derivative_orders, input_shape, input_offset,
                 target_offset, steps, stencil_shape, tile_layout):
    inputs = jnp.ones(input_shape)
    for extract_patches_method in ('conv', 'roll'):
      with self.subTest(f'method_{extract_patches_method}'):
        derivative_from_logits = layers.SpatialDerivativeFromLogits(
            stencil_shape, input_offset, target_offset, derivative_orders,
            steps, extract_patches_method, tile_layout)
        logits = jnp.ones(
            input_shape + (derivative_from_logits.subspace_size,))
        derivative = derivative_from_logits(jnp.expand_dims(inputs, -1), logits)
        expected_shape = input_shape + (1,)
        self.assertArrayEqual(expected_shape, derivative.shape)


class SpatialDerivativeTest(test_util.TestCase):
  """Tests SpatialDerivative module."""

  @parameterized.named_parameters([
      ('interpolation', (256,), (0,), np.sin, np.sin, 1e-1),
      ('first_derivative', (256,), (1,), np.sin, np.cos, 1e-1),
      ('second_derivative', (256,), (2,), np.sin, lambda x: -np.sin(x), 1e-1),
  ])
  def test_1d(self, grid_shape, derivative, initial_fn, expected_fn, atol):
    """Tests SpatialDerivative module in 1d."""
    ndims = len(grid_shape)
    grid = grids.Grid(grid_shape, domain=tuple([(0., 2. * np.pi) * ndims]))
    stencil_shape = (4,) * ndims
    tower_factory = functools.partial(
        _tower_factory, ndims=ndims, conv_block=layers.PeriodicConv1D)

    for extract_patches_method in ('conv', 'roll'):
      with self.subTest(f'method_{extract_patches_method}'):
        def module_forward(x):
          net = layers.SpatialDerivative(
              stencil_shape, grid.cell_center, grid.cell_faces[0], derivative,
              tower_factory, grid.step, extract_patches_method)  # pylint: disable=cell-var-from-loop
          return net(x)

        rng = jax.random.PRNGKey(41)
        spatial_derivative_model = hk.without_apply_rng(
            hk.transform(module_forward))

        x, = grid.mesh()
        x_target, = grid.mesh(offset=grid.cell_faces[0])
        inputs = jnp.expand_dims(initial_fn(x), -1)  # add channel dimension
        params = spatial_derivative_model.init(rng, inputs)
        outputs = spatial_derivative_model.apply(params, inputs)
        expected_outputs = np.expand_dims(expected_fn(x_target), -1)
        np.testing.assert_allclose(expected_outputs, outputs, atol=atol, rtol=0)

  @parameterized.named_parameters([
      ('interpolation', (128, 128), (0, 0),
       lambda x, y: np.sin(2 * x + y), lambda x, y: np.sin(2 * x + y), 0.2),
      ('first_derivative_x', (128, 128), (1, 0),
       lambda x, y: np.cos(2 * x + y), lambda x, y: -2 * np.sin(2 * x + y),
       0.1),
  ])
  def test_2d(self, grid_shape, derivative, initial_fn, expected_fn, atol):
    """Tests SpatialDerivative module in 2d."""
    ndims = len(grid_shape)
    grid = grids.Grid(grid_shape, domain=tuple([(0., 2. * np.pi)] * ndims))
    stencil_sizes = (3,) * ndims
    tower_factory = functools.partial(
        _tower_factory, ndims=ndims, conv_block=layers.PeriodicConv2D)

    for extract_patches_method in ('conv', 'roll'):
      with self.subTest(f'method_{extract_patches_method}'):
        def module_forward(inputs):
          net = layers.SpatialDerivative(
              stencil_sizes, grid.cell_center, grid.cell_center, derivative,
              tower_factory, grid.step, extract_patches_method)  # pylint: disable=cell-var-from-loop
          return net(inputs)

        rng = jax.random.PRNGKey(14)
        spatial_derivative_model = hk.without_apply_rng(
            hk.transform(module_forward))

        x, y = grid.mesh()
        inputs = np.expand_dims(initial_fn(x, y), -1)  # add channel dimension
        params = spatial_derivative_model.init(rng, inputs)
        outputs = spatial_derivative_model.apply(params, inputs)
        expected_outputs = np.expand_dims(expected_fn(x, y), -1)
        np.testing.assert_allclose(
            expected_outputs, outputs, atol=atol, rtol=0,
            err_msg=f'Failed for method "{extract_patches_method}"')

  def test_auxiliary_inputs(self):
    """Tests that auxiliary inputs don't change shape of the output."""
    grid = grids.Grid((64,), domain=tuple([(0., 2. * np.pi)]))
    stencil_sizes = (3,)
    tower_factory = functools.partial(
        _tower_factory, ndims=1, conv_block=layers.PeriodicConv1D)

    def module_forward(inputs, *auxiliary_inputs):
      net = layers.SpatialDerivative(
          stencil_sizes, grid.cell_center, grid.cell_center, (1,),
          tower_factory, grid.step)
      return net(inputs, *auxiliary_inputs)

    rng = jax.random.PRNGKey(14)
    spatial_derivative_model = hk.without_apply_rng(
        hk.transform(module_forward))

    inputs = np.expand_dims(grid.mesh()[0], -1)  # add channel dimension
    auxiliary_inputs = np.ones((64, 1))
    params = spatial_derivative_model.init(rng, inputs, auxiliary_inputs)
    outputs = spatial_derivative_model.apply(params, inputs, auxiliary_inputs)
    self.assertEqual(outputs.shape, (64, 1))

if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
