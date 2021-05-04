"""Tests for models_v2.equations."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.base import test_util
from jax_cfd.ml import equations
from jax_cfd.ml import physics_specifications

GRIDS = [
    grids.Grid((32, 32), domain=((0, 2 * jnp.pi),) * 2),
    grids.Grid((8, 8, 8), domain=((0, 2 * jnp.pi),) * 3),
]
C_INTERPOLATION_MODULES = [
    'interpolations.upwind',
    'interpolations.linear',
    'interpolations.lax_wendroff',
    'FusedLearnedInterpolation',
    'IndividualLearnedInterpolation',
]
PRESSURE_MODULES = [
    'pressures.fast_diagonalization',
    'pressures.conjugate_gradient',
]
FORCING_MODULES = [
    'forcings.filtered_linear_forcing',
    'forcings.kolmogorov_forcing',
    'forcings.taylor_green_forcing',
]
FORCING_SCALE = .1


def test_parameters():
  product = itertools.product(GRIDS,
                              C_INTERPOLATION_MODULES,
                              PRESSURE_MODULES,
                              FORCING_MODULES)
  parameters = []
  for grid, interpolation, pressure, forcing in product:
    name = '_'.join([module.split('.')[-1]
                     for module in (interpolation, pressure, forcing)])
    shape = 'x'.join(str(s) for s in grid.shape)
    name = f'{name}_{shape}'
    parameters.append(dict(
        testcase_name=name,
        c_interpolation_module=interpolation,
        pressure_module=pressure,
        grid=grid,
        forcing_module=forcing,
        convection_module='advections.self_advection',
        u_interpolation_module='interpolations.linear'))
  return parameterized.named_parameters(*parameters)


class NavierStokesModulesTest(test_util.TestCase):
  """Integration tests for equations and its submodules."""

  def _generate_inputs_and_outputs(self, config, grid):
    gin.enter_interactive_mode()
    gin.parse_config(config)
    dt = 0.1
    physics_specs = physics_specifications.get_physics_specs()
    def step_fwd(x):
      model = equations.modular_navier_stokes_model(
          grid, dt, physics_specs)
      return model(x)

    step_model = hk.without_apply_rng(hk.transform(step_fwd))
    inputs = []
    for seed, offset in enumerate(grid.cell_faces):
      rng_key = jax.random.PRNGKey(seed)
      data = jax.random.uniform(rng_key, grid.shape, jnp.float32)
      array = grids.AlignedArray(data, offset)
      inputs.append(array)
    inputs = tuple(inputs)
    rng = jax.random.PRNGKey(42)

    with funcutils.init_context():
      params = step_model.init(rng, inputs)
    self.assertIsNotNone(params)
    outputs = step_model.apply(params, inputs)
    return inputs, outputs

  @test_parameters()
  def test_all_modules(
      self,
      convection_module,
      c_interpolation_module,
      u_interpolation_module,
      pressure_module,
      forcing_module,
      grid
  ):
    """Intgeration tests checking that `step_fn` produces expected shape."""
    interp_module = 'advections.modular_advection'
    ns_module_name = 'equations.modular_navier_stokes_model'
    config = [
        f'{interp_module}.c_interpolation_module = @{c_interpolation_module}',
        f'{interp_module}.u_interpolation_module = @{u_interpolation_module}',
        f'{ns_module_name}.convection_module = @{convection_module}',
        f'{ns_module_name}.pressure_module = @{pressure_module}',
        f'{forcing_module}.scale = {FORCING_SCALE}',
        f'NavierStokesPhysicsSpecs.forcing_module = @{forcing_module}',
        'NavierStokesPhysicsSpecs.density = 1.',
        'NavierStokesPhysicsSpecs.viscosity = 0.1',
        'get_physics_specs.physics_specs_cls = @NavierStokesPhysicsSpecs',
    ]
    inputs, outputs = self._generate_inputs_and_outputs(config, grid)
    for u_output, u_input in zip(outputs, inputs):
      self.assertEqual(u_output.shape, u_input.shape)

  def test_smagorinsky(self):
    """Integration test checking that smag-LES produces expected shape."""
    diffusion_solver = 'implicit_diffusion_navier_stokes'
    evm_module_name = 'implicit_evm_solve_with_diffusion'
    config = [
        f'{diffusion_solver}.diffusion_module = @{evm_module_name}',
        f'{evm_module_name}.viscosity_module = @eddy_viscosity_model',
        'eddy_viscosity_model.viscosity_model = @smagorinsky_viscosity',
        'smagorinsky_viscosity.cs = 0.2',
        'NavierStokesPhysicsSpecs.forcing_module = @kolmogorov_forcing',
        'NavierStokesPhysicsSpecs.density = 1.',
        'NavierStokesPhysicsSpecs.viscosity = 0.1',
        'get_physics_specs.physics_specs_cls = @NavierStokesPhysicsSpecs',
    ]
    grid = GRIDS[0]
    inputs, outputs = self._generate_inputs_and_outputs(config, grid)
    for u_output, u_input in zip(outputs, inputs):
      self.assertEqual(u_output.shape, u_input.shape)

  def test_alternate_implementation_consistency(self):
    convection_module = 'advections.self_advection'
    advection_module = 'advections.modular_self_advection'
    interpolation_module = 'FusedLearnedInterpolation'
    pressure_module = 'pressures.fast_diagonalization'
    forcing_module = 'forcings.kolmogorov_forcing'
    ns_module_name = 'equations.modular_navier_stokes_model'
    grid = grids.Grid((32, 32), domain=((0, 2 * jnp.pi),) * 2)

    config = [
        f'{advection_module}.interpolation_module = @{interpolation_module}',
        f'{convection_module}.advection_module = @{advection_module}',
        f'{ns_module_name}.convection_module = @{convection_module}',
        f'{ns_module_name}.pressure_module = @{pressure_module}',
        f'{forcing_module}.scale = {FORCING_SCALE}',
        'FusedLearnedInterpolation.tags = ("u", "c")',
        f'NavierStokesPhysicsSpecs.forcing_module = @{forcing_module}',
        'NavierStokesPhysicsSpecs.density = 1.',
        'NavierStokesPhysicsSpecs.viscosity = 0.1',
        'get_physics_specs.physics_specs_cls = @NavierStokesPhysicsSpecs',
    ]
    _, outputs1 = self._generate_inputs_and_outputs(config, grid)

    config2 = config + [
        'FusedLearnedInterpolation.extract_patch_method = "conv"',
    ]
    _, outputs2 = self._generate_inputs_and_outputs(config2, grid)
    for out1, out2 in zip(outputs1, outputs2):
      self.assertAllClose(out1, out2)

    config2 = config + [
        'FusedLearnedInterpolation.fuse_constraints = True',
    ]
    _, outputs2 = self._generate_inputs_and_outputs(config2, grid)
    for out1, out2 in zip(outputs1, outputs2):
      self.assertAllClose(out1, out2)

    config2 = config + [
        'FusedLearnedInterpolation.fuse_constraints = True',
        'FusedLearnedInterpolation.fuse_patches = True',
    ]
    _, outputs2 = self._generate_inputs_and_outputs(config2, grid)
    for out1, out2 in zip(outputs1, outputs2):
      self.assertAllClose(out1, out2)

    config2 = config + [
        'FusedLearnedInterpolation.extract_patch_method = "conv"',
        'FusedLearnedInterpolation.fuse_constraints = True',
        'FusedLearnedInterpolation.tile_layout = (8, 1)',
    ]
    _, outputs2 = self._generate_inputs_and_outputs(config2, grid)
    for out1, out2 in zip(outputs1, outputs2):
      self.assertAllClose(out1, out2)


if __name__ == '__main__':
  absltest.main()
