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

"""Tests for jax_cfd.subgrid_models."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_cfd.base import advection
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.base import pressure
from jax_cfd.base import subgrid_models
from jax_cfd.base import test_util
import numpy as np


def periodic_grid_variable(data, offset, grid):
  return grids.GridVariable(
      array=grids.GridArray(data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


def zero_velocity_field(grid: grids.Grid) -> grids.GridVariableVector:
  """Returns an all-zero periodic velocity fields."""
  return tuple(periodic_grid_variable(jnp.zeros(grid.shape), o, grid)
               for o in grid.cell_faces)


def sinusoidal_velocity_field(grid: grids.Grid) -> grids.GridVariableVector:
  """Returns a divergence-free velocity flow on `grid`."""
  mesh_size = jnp.array(grid.shape) * jnp.array(grid.step)
  vs = tuple(jnp.sin(2. * np.pi * g / s)
             for g, s in zip(grid.mesh(), mesh_size))
  return tuple(periodic_grid_variable(v, o, grid)
               for v, o in zip(vs[1:] + vs[:1], grid.cell_faces))


def gaussian_force_field(grid: grids.Grid) -> grids.GridArrayVector:
  """Returns a 'Gaussian-shaped' force field in the 'x' direction."""
  mesh = grid.mesh()
  mesh_size = jnp.array(grid.shape) * jnp.array(grid.step)
  offsets = grid.cell_faces
  v = [grids.GridArray(
      jnp.exp(-sum([jnp.square(x / s - .5)
                    for x, s in zip(mesh, mesh_size)]) * 100.),
      offsets[0], grid)]
  for j in range(1, grid.ndim):
    v.append(grids.GridArray(jnp.zeros(grid.shape), offsets[j], grid))
  return tuple(v)


def gaussian_forcing(v: grids.GridVariableVector) -> grids.GridArrayVector:
  """Returns Gaussian field forcing."""
  grid = grids.consistent_grid(*v)
  return gaussian_force_field(grid)


def momentum(v: grids.GridVariableVector, density: float):
  """Returns the momentum due to velocity field `v`."""
  grid = grids.consistent_grid(*v)
  return jnp.array([u.data for u in v]).sum() * density * jnp.array(
      grid.step).prod()


def _convect_upwind(v: grids.GridVariableVector) -> grids.GridArrayVector:
  return tuple(advection.advect_upwind(u, v) for u in v)


class SubgridModelsTest(test_util.TestCase):

  def test_smagorinsky_viscosity(self):
    grid = grids.Grid((3, 3))
    v = (periodic_grid_variable(jnp.zeros(grid.shape), (1, 0.5), grid),
         periodic_grid_variable(jnp.zeros(grid.shape), (0.5, 1), grid))
    c00 = grids.GridArray(jnp.zeros(grid.shape), offset=(0, 0), grid=grid)
    c01 = grids.GridArray(jnp.zeros(grid.shape), offset=(0, 1), grid=grid)
    c10 = grids.GridArray(jnp.zeros(grid.shape), offset=(1, 0), grid=grid)
    c11 = grids.GridArray(jnp.zeros(grid.shape), offset=(1, 1), grid=grid)
    s_ij = grids.GridArrayTensor(np.array([[c00, c01], [c10, c11]]))
    viscosity = subgrid_models.smagorinsky_viscosity(
        s_ij=s_ij, v=v, dt=0.1, cs=0.2)
    self.assertIsInstance(viscosity, grids.GridArrayTensor)
    self.assertEqual(viscosity.shape, (2, 2))
    self.assertAllClose(viscosity[0, 0], c00)
    self.assertAllClose(viscosity[0, 1], c01)
    self.assertAllClose(viscosity[1, 0], c10)
    self.assertAllClose(viscosity[1, 1], c11)

  def test_evm_model(self):
    grid = grids.Grid((3, 3))
    v = (
        periodic_grid_variable(jnp.zeros(grid.shape), (1, 0.5), grid),
        periodic_grid_variable(jnp.zeros(grid.shape), (0.5, 1), grid))
    viscosity_fn = functools.partial(
        subgrid_models.smagorinsky_viscosity, dt=1.0, cs=0.2)
    acceleration = subgrid_models.evm_model(v, viscosity_fn)
    self.assertIsInstance(acceleration, tuple)
    self.assertLen(acceleration, 2)
    self.assertAllClose(acceleration[0], v[0].array)
    self.assertAllClose(acceleration[1], v[1].array)

  @parameterized.named_parameters(
      dict(
          testcase_name='sinusoidal_velocity_base',
          cs=0.0,
          velocity=sinusoidal_velocity_field,
          forcing=None,
          shape=(100, 100),
          step=(1., 1.),
          density=1.,
          viscosity=1e-4,
          convect=advection.convect_linear,
          pressure_solve=pressure.solve_cg,
          dt=1e-3,
          time_steps=1000,
          divergence_atol=1e-3,
          momentum_atol=2e-3),
      dict(
          testcase_name='gaussian_force_upwind_with_subgrid_model',
          cs=0.12,
          velocity=zero_velocity_field,
          forcing=gaussian_forcing,
          shape=(40, 40, 40),
          step=(1., 1., 1.),
          density=1.,
          viscosity=0,
          convect=_convect_upwind,
          pressure_solve=pressure.solve_cg,
          dt=1e-3,
          time_steps=100,
          divergence_atol=1e-4,
          momentum_atol=1e-4),
      dict(
          testcase_name='sinusoidal_velocity_with_subgrid_model',
          cs=0.12,
          velocity=sinusoidal_velocity_field,
          forcing=None,
          shape=(100, 100),
          step=(1., 1.),
          density=1.,
          viscosity=1e-4,
          convect=advection.convect_linear,
          pressure_solve=pressure.solve_fast_diag,
          dt=1e-3,
          time_steps=1000,
          divergence_atol=1e-3,
          momentum_atol=1e-3),
  )
  def test_divergence_and_momentum(
      self,
      cs,
      velocity,
      forcing,
      shape,
      step,
      density,
      viscosity,
      convect,
      pressure_solve,
      dt,
      time_steps,
      divergence_atol,
      momentum_atol,
  ):
    grid = grids.Grid(shape, step)
    kwargs = dict(
        density=density,
        viscosity=viscosity,
        cs=cs,
        dt=dt,
        grid=grid,
        convect=convect,
        pressure_solve=pressure_solve,
        forcing=forcing)
    # Explicit and implicit navier-stokes solvers:
    explicit_eq = subgrid_models.explicit_smagorinsky_navier_stokes(**kwargs)
    implicit_eq = subgrid_models.implicit_smagorinsky_navier_stokes(**kwargs)

    v_initial = velocity(grid)
    v_final = funcutils.repeated(explicit_eq, time_steps)(v_initial)
    # TODO(dkochkov) consider adding more thorough tests for these models.
    with self.subTest('divergence free'):
      divergence = fd.divergence(v_final)
      self.assertLess(jnp.max(divergence.data), divergence_atol)

    with self.subTest('conservation of momentum'):
      initial_momentum = momentum(v_initial, density)
      final_momentum = momentum(v_final, density)
      if forcing is not None:
        expected_change = (
            jnp.array([f.data for f in forcing(v_initial)]).sum() *
            jnp.array(grid.step).prod() * dt * time_steps)
      else:
        expected_change = 0
      expected_momentum = initial_momentum + expected_change
      self.assertAllClose(expected_momentum, final_momentum, atol=momentum_atol)

    with self.subTest('explicit-implicit consistency'):
      v_final_2 = funcutils.repeated(implicit_eq, time_steps)(v_initial)
      for axis in range(grid.ndim):
        self.assertAllClose(v_final[axis], v_final_2[axis], atol=1e-4,
                            err_msg=f'axis={axis}')


if __name__ == '__main__':
  absltest.main()
