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

"""Code for subgrid models."""
import functools
from typing import Any, Callable, Mapping, Optional, Tuple

import jax
from jax_cfd.base import equations
from jax_cfd.base import finite_differences
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.base import interpolation
import numpy as np


AlignedArray = grids.AlignedArray
AlignedField = Tuple[AlignedArray, ...]
InterpolationFn = interpolation.InterpolationFn
ViscosityFn = Callable[[grids.Tensor, AlignedField, grids.Grid], grids.Tensor]


def smagorinsky_viscosity(
    s_ij: grids.Tensor,
    v: AlignedField,
    grid: grids.Grid,
    dt: float = None,
    cs: float = 0.2,
    interpolate_fn: InterpolationFn = interpolation.linear
) -> AlignedField:
  """Computes eddy viscosity based on Smagorinsky model.

  This viscosity model computes scalar eddy viscosity at `grid.cell_center` and
  then interpolates it to offsets of the strain rate tesnor `s_ij`. Based on:
  https://en.wikipedia.org/wiki/Large_eddy_simulation#Smagorinsky-Lilly_model

  Args:
    s_ij: strain rate tensor that is equal to the forward finite difference
      derivatives of the velocity field `(d(u_i)/d(x_j) + d(u_j)/d(x_i)) / 2`.
    v: velocity field, passed to `interpolate_fn`.
    grid: grid object.
    dt: integration time step passed to `interpolate_fn`. Can be `None` if
      `interpolate_fn` is independent of `dt`. Default: `None`.
    cs: the Smagorinsky constant.
    interpolate_fn: interpolation method to use for viscosity interpolations.

  Returns:
    tensor of AlignedArray's containing values of the eddy viscosity at the
      same grid offsets as the strain tensor `s_ij`.
  """
  s_ij_offsets = [array.offset for array in s_ij.ravel()]
  unique_offsets = list(set(s_ij_offsets))
  cell_center = grid.cell_center
  interpolate_to_center = lambda x: interpolate_fn(x, cell_center, grid, v, dt)
  centered_s_ij = np.vectorize(interpolate_to_center)(s_ij)
  # geometric average
  cutoff = np.prod(np.array(grid.step))**(1 / grid.ndim)
  viscosity = (cs * cutoff)**2 * np.sqrt(
      2 * np.trace(centered_s_ij.dot(centered_s_ij)))
  viscosities_dict = {
      offset: interpolate_fn(viscosity, offset, grid, v, dt).data
      for offset in unique_offsets}
  viscosities = [viscosities_dict[offset] for offset in s_ij_offsets]
  return jax.tree_unflatten(jax.tree_util.tree_structure(s_ij), viscosities)


def evm_model(
    v: AlignedField,
    grid: grids.Grid,
    viscosity_fn: ViscosityFn,
) -> AlignedField:
  """Computes acceleration due to eddy viscosity turbulence model.

  Eddy viscosity models compute a turbulence closure term as a divergence of
  the subgrid-scale stress tensor, which is expressed as velocity dependent
  viscosity times the rate of strain tensor. This module delegates computation
  of the eddy-viscosity to `viscosity_fn` function.

  Args:
    v: velocity field.
    grid: a `grids.Grid` object representing spatial discretization.
    viscosity_fn: function that computes viscosity values at the same offsets
      as strain rate tensor provided as input.

  Returns:
    acceleration of the velocity field `v`.
  """
  s_ij = grids.Tensor([
      [0.5 * (finite_differences.forward_difference(v[i], grid, j) +  # pylint: disable=g-complex-comprehension
              finite_differences.forward_difference(v[j], grid, i))
       for j in range(grid.ndim)]
      for i in range(grid.ndim)])
  viscosity = viscosity_fn(s_ij, v, grid)
  tau = jax.tree_multimap(lambda x, y: -2. * x * y, viscosity, s_ij)
  return tuple(-finite_differences.divergence(tau[i, :], grid)
               for i in range(grid.ndim))


# TODO(dkochkov) remove when b/160947162 is resolved.
def implicit_evm_solve_with_diffusion(
    v: AlignedField,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    configured_evm_model: Callable,  # pylint: disable=g-bare-generic
    cg_kwargs: Optional[Mapping[str, Any]] = None
) -> AlignedField:
  """Implicit solve for eddy viscosity model combined with diffusion.

  This method is intended to be used with `implicit_diffusion_navier_stokes` to
  avoid potential numerical instabilities associated with fast diffusion modes.

  Args:
    v: current velocity field.
    viscosity: constant viscosity coefficient.
    dt: time step of implicit integration.
    grid: a `grids.Grid` object specifying spatial discretization.
    configured_evm_model: eddy viscosity model with specified `viscosity_fn`.
    cg_kwargs: keyword arguments passed to jax.scipy.sparse.linalg.cg.

  Returns:
    velocity field advanced in time by `dt`.
  """
  if cg_kwargs is None:
    cg_kwargs = {}
  cg_kwargs = dict(cg_kwargs)
  cg_kwargs.setdefault('tol', 1e-6)
  cg_kwargs.setdefault('atol', 1e-6)

  vector_laplacian = np.vectorize(
      functools.partial(finite_differences.laplacian, grid=grid))

  def linear_op(v):
    acceleration = configured_evm_model(v, grid)
    return tuple(v - dt * (acceleration + viscosity * vector_laplacian(v)))

  # We normally prefer fast diagonalization, but that requires an outer
  # product structure for the linear operation, which doesn't hold here.
  # TODO(shoyer): consider adding a preconditioner
  v_prime, _ = jax.scipy.sparse.linalg.cg(linear_op, v, **cg_kwargs)
  return v_prime


def explicit_smagorinsky_navier_stokes(dt, cs, forcing, **kwargs):
  """Constructs explicit navier-stokes model with Smagorinsky viscosity term.

  Navier-Stokes model that uses explicit time stepping for the eddy viscosity
  model based on Smagorinsky closure term.

  Args:
    dt: time step to be performed.
    cs: smagorinsky constant.
    forcing: forcing term.
    **kwargs: other keyword arguments to be passed to
      `equations.semi_implicit_navier_stokes`.

  Returns:
    A function that performs a single step of time evolution of navier-stokes
    equations with Smagorinsky turbulence model.
  """
  viscosity_fn = functools.partial(
      smagorinsky_viscosity, dt=dt, cs=cs)
  smagorinsky_acceleration = functools.partial(
      evm_model, viscosity_fn=viscosity_fn)
  if forcing is None:
    forcing = smagorinsky_acceleration
  else:
    forcing = forcings.sum_forcings(forcing, smagorinsky_acceleration)
  return equations.semi_implicit_navier_stokes(dt=dt, forcing=forcing, **kwargs)


def implicit_smagorinsky_navier_stokes(dt, cs, **kwargs):
  """Constructs implicit navier-stokes model with Smagorinsky viscosity term.

  Navier stokes model that uses implicit time stepping for the eddy viscosity
  model based on Smagorinsky closure term. The implicit step is performed using
  conjugate gradients and is combined with diffusion solve.

  Args:
    dt: time step to be performed.
    cs: smagorinsky constant.
    **kwargs: other keyword arguments to be passed to
      `equations.implicit_diffusion_navier_stokes`.

  Returns:
    A function that performs a single step of time evolution of navier-stokes
    equations with Smagorinsky turbulence model.
  """
  viscosity_fn = functools.partial(
      smagorinsky_viscosity, dt=dt, cs=cs)
  smagorinsky_acceleration = functools.partial(
      evm_model, viscosity_fn=viscosity_fn)
  diffusion_solve_with_evm = functools.partial(
      implicit_evm_solve_with_diffusion,
      configured_evm_model=smagorinsky_acceleration)
  return equations.implicit_diffusion_navier_stokes(
      diffusion_solve=diffusion_solve_with_evm, dt=dt, **kwargs)
