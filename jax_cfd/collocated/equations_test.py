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

"""Tests for jax_cfd.equations."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.base import test_util
from jax_cfd.collocated import advection
from jax_cfd.collocated import equations
from jax_cfd.collocated import pressure
import numpy as np


def zero_velocity_field(grid: grids.Grid) -> grids.GridVariableVector:
  """Returns an all-zero periodic velocity fields."""
  offset = grid.cell_center
  data = jnp.zeros(grid.shape)
  return tuple(
      grids.GridVariable(grids.GridArray(data, offset, grid),
                         boundaries.periodic_boundary_conditions(grid.ndim))
      for _ in range(grid.ndim))


def sinusoidal_velocity_field(grid: grids.Grid) -> grids.GridVariableVector:
  """Returns a divergence-free velocity flow on `grid`."""
  offset = grid.cell_center
  mesh = grid.mesh(offset)
  mesh_size = jnp.array(grid.shape) * jnp.array(grid.step)
  vs = tuple(
      jnp.sin(2. * np.pi * g / s) for g, s in zip(mesh, mesh_size))
  return tuple(
      grids.GridVariable(grids.GridArray(v, offset, grid),
                         boundaries.periodic_boundary_conditions(grid.ndim))
      for v in vs[1:] + vs[:1])


def gaussian_force_field(grid) -> grids.GridArrayVector:
  """Returns a 'Gaussian-shaped' force field in the 'x' direction."""
  offset = grid.cell_center
  mesh = grid.mesh(offset)
  mesh_size = jnp.array(grid.shape) * jnp.array(grid.step)
  data = jnp.exp(-sum([jnp.square(x / s - .5)
                       for x, s in zip(mesh, mesh_size)]) * 100.)
  v = [grids.GridArray(data, offset, grid)]
  for _ in range(1, grid.ndim):
    v.append(grids.GridArray(jnp.zeros(grid.shape), offset, grid))
  return tuple(v)


def gaussian_forcing(v: grids.GridVariableVector) -> grids.GridArrayVector:
  """Returns Gaussian field forcing."""
  grid = grids.consistent_grid(*v)
  return gaussian_force_field(grid)


def momentum(v: grids.GridVariableVector, density: float) -> grids.Array:
  """Returns the momentum due to velocity field `v`."""
  grid = grids.consistent_grid(*v)
  return jnp.array([u.data for u in v]).sum() * density * jnp.array(
      grid.step).prod()


def _convect_upwind(v: grids.GridVariableVector) -> grids.GridArrayVector:
  return tuple(advection.advect_linear(u, v) for u in v)


class SemiImplicitNavierStokesTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='semi_implicit_sinusoidal_velocity_base',
           velocity=sinusoidal_velocity_field,
           forcing=None,
           shape=(100, 100),
           step=(1., 1.),
           density=1.,
           viscosity=1e-4,
           convect=None,
           pressure_solve=pressure.solve_cg,
           dt=1e-3,
           time_steps=1000,
           divergence_atol=1e-3,
           momentum_atol=2e-3),
      dict(testcase_name='semi_implicit_gaussian_force_upwind',
           velocity=zero_velocity_field,
           forcing=gaussian_forcing,
           shape=(40, 40, 40),
           step=(1., 1., 1.),
           density=1.,
           viscosity=None,
           convect=_convect_upwind,
           pressure_solve=pressure.solve_cg,
           dt=1e-3,
           time_steps=100,
           divergence_atol=1e-4,
           momentum_atol=2e-4),
  )
  def test_divergence_and_momentum(
      self, velocity, forcing, shape, step, density, viscosity, convect,
      pressure_solve, dt, time_steps, divergence_atol, momentum_atol,
  ):
    grid = grids.Grid(shape, step)

    navier_stokes = equations.semi_implicit_navier_stokes(
        density,
        viscosity,
        dt,
        grid,
        convect=convect,
        pressure_solve=pressure_solve,
        forcing=forcing)

    v_initial = velocity(grid)
    v_final = funcutils.repeated(navier_stokes, time_steps)(v_initial)

    divergence = fd.centered_divergence(v_final)
    self.assertLess(jnp.max(divergence.data), divergence_atol)

    initial_momentum = momentum(v_initial, density)
    final_momentum = momentum(v_final, density)
    if forcing is not None:
      expected_change = (
          jnp.array([f.data for f in forcing(v_initial)]).sum() *
          jnp.array(grid.step).prod() * dt * time_steps)
    else:
      expected_change = 0
    self.assertAllClose(
        initial_momentum + expected_change, final_momentum, atol=momentum_atol)


if __name__ == '__main__':
  absltest.main()
