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
from typing import Callable, Optional

import jax

from jax_cfd.base import advection as base_advection
from jax_cfd.base import grids
from jax_cfd.collocated import diffusion
from jax_cfd.collocated import pressure

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


def sum_fields(*args):
  return jax.tree_map(lambda *a: sum(a), *args)


def semi_implicit_navier_stokes(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressure.solve_cg,
    forcing: Optional[ForcingFn] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused

  if convect is None:
    def convect(v):  # pylint: disable=function-redefined
      return tuple(
          base_advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

  convect = jax.named_call(convect, name='convection')
  diffuse = jax.named_call(diffuse, name='diffusion')
  pressure_projection = jax.named_call(
      pressure.projection, name='pressure')

  @jax.named_call
  def navier_stokes_step(v: GridVariableVector) -> GridVariableVector:
    """Computes state at time `t + dt` using first order time integration."""
    # Collect the acceleration terms
    convection = convect(v)
    accelerations = [convection]
    if viscosity is not None:
      diffusion_ = tuple(diffuse(u, viscosity / density) for u in v)
      accelerations.append(diffusion_)
    if forcing is not None:
      # TODO(shoyer): include time in state?
      force = forcing(v)
      accelerations.append(tuple(f / density for f in force))
    dvdt = sum_fields(*accelerations)
    # Update v by taking a time step
    v = tuple(
        grids.GridVariable(u.array + dudt * dt, u.bc)
        for u, dudt in zip(v, dvdt))
    # Pressure projection to incompressible velocity field
    v = pressure_projection(v, pressure_solve)
    return v
  return navier_stokes_step
