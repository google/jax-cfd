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

"""Examples of defining equations."""
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from jax_cfd.base import advection
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.base import pressure

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

AlignedArray = grids.AlignedArray
AlignedField = Tuple[AlignedArray, ...]


def sum_fields(*args):
  return jax.tree_multimap(lambda *a: sum(a), *args)


def stable_time_step(
    max_velocity: float,
    max_courant_number: float,
    viscosity: float,
    grid: grids.Grid,
    implicit_diffusion: bool = False,
) -> float:
  """Calculate a stable time step for Navier-Stokes."""
  dt = advection.stable_time_step(max_velocity, max_courant_number, grid)
  if not implicit_diffusion:
    diffusion_dt = diffusion.stable_time_step(viscosity, grid)
    if diffusion_dt < dt:
      raise ValueError(f'stable time step for diffusion is smaller than '
                       f'the chosen timestep: {diffusion_dt} vs {dt}')
  return dt


def dynamic_time_step(v: AlignedField,
                      max_courant_number: float,
                      viscosity: float,
                      grid: grids.Grid,
                      implicit_diffusion: bool = False) -> float:
  """Pick a dynamic time-step for Navier-Stokes based on stable advection."""
  v_max = jnp.sqrt(jnp.max(sum(u.data ** 2 for u in v)))
  return stable_time_step(
      v_max, max_courant_number, viscosity, grid, implicit_diffusion)


def semi_implicit_navier_stokes(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[Callable] = None,
    diffuse: Callable = diffusion.diffuse,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[Callable] = None,
) -> Callable:
  """Returns a function that performs a time step of Navier Stokes."""

  if convect is None:
    def convect(v, grid):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, grid, dt) for u in v)

  convect = jax.named_call(convect, name='convection')
  diffuse = jax.named_call(diffuse, name='diffusion')
  pressure_projection = jax.named_call(pressure.projection, name='pressure')

  # TODO(jamieas): Consider a scheme where pressure calculations and
  # advection/diffusion are staggered in time.
  @jax.named_call
  def navier_stokes_step(v: AlignedField) -> AlignedField:
    """Computes state at time `t + dt` using first order time integration."""
    convection = convect(v, grid)
    accelerations = [convection]
    if viscosity:
      diffusion_ = tuple(diffuse(u, viscosity / density, grid) for u in v)
      accelerations.append(diffusion_)
    if forcing is not None:
      # TODO(shoyer): include time in state?
      f = forcing(v, grid)
      accelerations.append(tuple(f_i / density for f_i in f))
    v_t = sum_fields(*accelerations)
    v = tuple(u + u_t * dt for u, u_t in zip(v, v_t))
    v = pressure_projection(v, grid, pressure_solve)
    return v
  return navier_stokes_step


def implicit_diffusion_navier_stokes(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[Callable] = None,
    diffusion_solve: Callable = diffusion.solve_fast_diag,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[Callable] = None,
) -> Callable:
  """Returns a function that performs a time step of Navier Stokes."""

  if convect is None:
    def convect(v, grid):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, grid, dt) for u in v)

  convect = jax.named_call(convect, name='convection')
  pressure_projection = jax.named_call(pressure.projection, name='pressure')
  diffusion_solve = jax.named_call(diffusion_solve, name='diffusion')

  @jax.named_call
  def navier_stokes_step(v: AlignedField) -> AlignedField:
    """Computes state at time `t + dt` using first order time integration."""
    convection = convect(v, grid)
    accelerations = [convection]
    if forcing is not None:
      # TODO(shoyer): include time in state?
      f = forcing(v, grid)
      accelerations.append(tuple(f_i / density for f_i in f))
    v_t = sum_fields(*accelerations)
    v = tuple(u + u_t * dt for u, u_t in zip(v, v_t))
    v = pressure_projection(v, grid, pressure_solve)
    v = diffusion_solve(v, viscosity, dt, grid)
    return v
  return navier_stokes_step
