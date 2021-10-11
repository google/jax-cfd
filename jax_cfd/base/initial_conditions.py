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
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from jax_cfd.base import filter_utils
from jax_cfd.base import funcutils
from jax_cfd.base import grids
from jax_cfd.base import pressure
import numpy as np

Array = Union[np.ndarray, jnp.DeviceArray]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def wrap_velocities(v, grid: grids.Grid) -> GridArrayVector:
  """Wrap velocity arrays for input into simulations."""
  return tuple(grids.GridArray(u, offset, grid)
               for u, offset in zip(v, grid.cell_faces))


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
) -> GridArrayVector:
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
    A divergence free velocity field with the given maximum velocity.
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
  for k in keys:
    noise = jax.random.normal(k, grid.shape)
    velocity_components.append(
        filter_utils.filter(spectral_density, noise, grid))
  filtered = wrap_velocities(velocity_components, grid)

  def project_and_normalize(v: GridArrayVector):
    # TODO(pnorgaard) remove temporary GridVariable hack
    v = tuple(grids.make_gridvariable_from_gridarray(u) for u in v)
    v = pressure.projection(v)
    v = tuple(u.array for u in v)  # strip boundary conditiosn
    vmax = _max_speed(v)
    v = tuple(maximum_velocity * u / vmax for u in v)
    return v
  # Due to numerical precision issues, we repeatedly normalize and project the
  # velocity field. This ensures that it is divergence-free and achieves the
  # specified maximum velocity.
  return funcutils.repeated(project_and_normalize, iterations)(filtered)


def initial_velocity_field(
    velocity_fns: Tuple[Callable[..., Array], ...],
    grid: grids.Grid,
    iterations: Optional[int] = None) -> GridVariableVector:
  """Given velocity functions on arrays, returns the velocity field on the grid.

  Typical usage example:
    grid = cfd.grids.Grid((128, 128))
    x_velocity_fn = lambda x, y: jnp.sin(x) * jnp.cos(y)
    y_velocity_fn = lambda x, y: jnp.zeros_like(x)
    v0 = initial_velocity_field((x_velocity_fn, y_velocity_fn), grid, 5)

  Args:
    velocity_fns: Functions for computing each velocity component. These should
      takes the args (x, y, ...) and return an array of the same shape.
    grid: the grid on which the velocity field is defined.
    iterations: if specified, the number of iterations of applied projection
      onto an incompressible velocity field.

  Returns:
    Velocity components defined with expected offsets on the grid.
  """
  # TODO(pnorgaard) Migrate boundary conditions to an input arg
  u_bc = grids.BoundaryConditions((grids.PERIODIC,) * grid.ndim)
  v = tuple(
      grids.GridVariable(grid.eval_on_mesh(v_fn, offset), u_bc)
      for v_fn, offset in zip(velocity_fns, grid.cell_faces))
  if iterations is not None:
    v = funcutils.repeated(pressure.projection, iterations)(v)
  return v
