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


"""Prepare initial conditions for simulations."""
import functools
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import filter_utils
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.base import pressure
import numpy as np

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

Array = Union[np.ndarray, jnp.DeviceArray]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


def wrap_velocities(
    v: Sequence[Array],
    grid: grids.Grid,
    bcs: Sequence[BoundaryConditions],
) -> GridVariableVector:
  """Wrap velocity arrays for input into simulations."""
  return tuple(grids.GridVariable(grids.GridArray(u, offset, grid), bc)
               for u, offset, bc in zip(v, grid.cell_faces, bcs))


def _log_normal_pdf(x, mode, variance=.25):
  """Unscaled PDF for a log normal given `mode` and log variance 1."""
  mean = jnp.log(mode) + variance
  logx = jnp.log(x)
  return jnp.exp(-(mean - logx)**2 / 2 / variance - logx)


def _max_speed(v):
  return jnp.linalg.norm([u.data for u in v], axis=0).max()


def filtered_velocity_field(
    rng_key: grids.Array,
    grid: grids.Grid,
    maximum_velocity: float = 1,
    peak_wavenumber: float = 3,
    iterations: int = 3,
) -> GridVariableVector:
  """Create divergence-free velocity fields with appropriate spectral filtering.

  Args:
    rng_key: key for seeding the random initial velocity field.
    grid: the grid on which the velocity field is defined.
    maximum_velocity: the maximum speed in the velocity field.
    peak_wavenumber: the velocity field will be filtered so that the largest
      magnitudes are associated with this wavenumber.
    iterations: the number of repeated pressure projection and normalization
      iterations to apply.
  Returns:
    A divergence free velocity field with the given maximum velocity. Associates
    periodic boundary conditions with the velocity field components.
  """
  # Log normal distribution peaked at `peak_wavenumber`. Note that we have to
  # divide by `k ** (ndim - 1)` to account for the volume of the
  # `ndim - 1`-sphere of values with wavenumber `k`.
  def spectral_density(k):
    return _log_normal_pdf(k, peak_wavenumber) / k ** (grid.ndim - 1)
  # TODO(b/156601712): Switch back to the vmapped implementation of filtering:
  # noise = jax.random.normal(rng_key, (grid.ndim,) + grid.shape)
  # filtered = wrap_velocities(jax.vmap(spectral.filter, (None, 0, None))(
  #     spectral_density, noise, grid), grid)
  keys = jax.random.split(rng_key, grid.ndim)
  velocity_components = []
  boundary_conditions = []
  for k in keys:
    noise = jax.random.normal(k, grid.shape)
    velocity_components.append(
        filter_utils.filter(spectral_density, noise, grid))
    boundary_conditions.append(
        boundaries.periodic_boundary_conditions(grid.ndim))
  velocity = wrap_velocities(velocity_components, grid, boundary_conditions)

  def project_and_normalize(v: GridVariableVector):
    v = pressure.projection(v)
    vmax = _max_speed(v)
    v = tuple(
        grids.GridVariable(maximum_velocity * u.array / vmax, u.bc) for u in v)
    return v
  # Due to numerical precision issues, we repeatedly normalize and project the
  # velocity field. This ensures that it is divergence-free and achieves the
  # specified maximum velocity.
  return funcutils.repeated(project_and_normalize, iterations)(velocity)


def initial_velocity_field(
    velocity_fns: Tuple[Callable[..., Array], ...],
    grid: grids.Grid,
    velocity_bc: Optional[Sequence[BoundaryConditions]] = None,
    pressure_solve: Callable = pressure.solve_fast_diag,
    iterations: Optional[int] = None,
) -> GridVariableVector:
  """Given velocity functions on arrays, returns the velocity field on the grid.

  Typical usage example:
    grid = cfd.grids.Grid((128, 128))
    x_velocity_fn = lambda x, y: jnp.sin(x) * jnp.cos(y)
    y_velocity_fn = lambda x, y: jnp.zeros_like(x)
    v0 = initial_velocity_field((x_velocity_fn, y_velocity_fn), grid, 5)

  Args:
    velocity_fns: functions for computing each velocity component. These should
      takes the args (x, y, ...) and return an array of the same shape.
    grid: the grid on which the velocity field is defined.
    velocity_bc: the boundary conditions to associate with each velocity
      component. If unspecified, uses periodic boundary conditions.
    pressure_solve: method used to solve pressure projection.
    iterations: if specified, the number of iterations of applied projection
      onto an incompressible velocity field.

  Returns:
    Velocity components defined with expected offsets on the grid.
  """
  if velocity_bc is None:
    velocity_bc = (
        boundaries.periodic_boundary_conditions(grid.ndim),) * grid.ndim
  v = tuple(
      grids.GridVariable(grid.eval_on_mesh(v_fn, offset), bc) for v_fn, offset,
      bc in zip(velocity_fns, grid.cell_faces, velocity_bc))
  if iterations is not None:
    projection = functools.partial(pressure.projection, solve=pressure_solve)
    v = funcutils.repeated(projection, iterations)(v)
  return v
