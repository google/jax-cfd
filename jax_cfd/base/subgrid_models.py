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
from typing import Any, Callable, Mapping, Optional

import jax
from jax_cfd.base import boundaries
from jax_cfd.base import equations
from jax_cfd.base import finite_differences
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.base import interpolation
import numpy as np


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = interpolation.InterpolationFn
ViscosityFn = Callable[[grids.GridArrayTensor, GridVariableVector],
                       grids.GridArrayTensor]

# TODO(pnorgaard) Refactor subgrid_models to interpolate, then differentiate


def smagorinsky_viscosity(
    s_ij: grids.GridArrayTensor,
    v: GridVariableVector,
    dt: Optional[float] = None,
    cs: float = 0.2,
    interpolate_fn: InterpolationFn = interpolation.linear
) -> grids.GridArrayTensor:
  """Computes eddy viscosity based on Smagorinsky model.

  This viscosity model computes scalar eddy viscosity at `grid.cell_center` and
  then interpolates it to offsets of the strain rate tesnor `s_ij`. Based on:
  https://en.wikipedia.org/wiki/Large_eddy_simulation#Smagorinsky-Lilly_model

  Args:
    s_ij: strain rate tensor that is equal to the forward finite difference
      derivatives of the velocity field `(d(u_i)/d(x_j) + d(u_j)/d(x_i)) / 2`.
    v: velocity field, passed to `interpolate_fn`.
    dt: integration time step passed to `interpolate_fn`. Can be `None` if
      `interpolate_fn` is independent of `dt`. Default: `None`.
    cs: the Smagorinsky constant.
    interpolate_fn: interpolation method to use for viscosity interpolations.

  Returns:
    tensor of GridArray's containing values of the eddy viscosity at the
      same grid offsets as the strain tensor `s_ij`.
  """
  # Present implementation:
  #   - s_ij is a GridArrayTensor
  #   - v is converted to a GridVariableVector
  #   - interpolation method is wrapped so that interpolated quanity is a
  #     GridArray (rather than GridVariable), using periodic BC.
  #
  # This should be revised so that s_ij is computed by first interpolating
  # velocity and then computing s_ij via finite differences, producing
  # a `GridVariableTensor`. Then no wrapper or GridArray/GridVariable
  # conversion hacks are needed.
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('smagorinsky_viscosity only valid for periodic BC.')
  bc = grids.unique_boundary_conditions(*v)

  def wrapped_interp_fn(c, offset, v, dt):
    return interpolate_fn(grids.GridVariable(c, bc), offset, v, dt).array

  grid = grids.consistent_grid(*s_ij.ravel(), *v)
  bc = boundaries.periodic_boundary_conditions(grid.ndim)
  s_ij_offsets = [array.offset for array in s_ij.ravel()]
  unique_offsets = list(set(s_ij_offsets))
  cell_center = grid.cell_center
  interpolate_to_center = lambda x: wrapped_interp_fn(x, cell_center, v, dt)
  centered_s_ij = np.vectorize(interpolate_to_center)(s_ij)
  # geometric average
  cutoff = np.prod(np.array(grid.step))**(1 / grid.ndim)
  viscosity = (cs * cutoff)**2 * np.sqrt(
      2 * np.trace(centered_s_ij.dot(centered_s_ij)))
  viscosities_dict = {
      offset: wrapped_interp_fn(viscosity, offset, v, dt).data
      for offset in unique_offsets}
  viscosities = [viscosities_dict[offset] for offset in s_ij_offsets]
  return jax.tree_unflatten(jax.tree_util.tree_structure(s_ij), viscosities)


def evm_model(
    v: GridVariableVector,
    viscosity_fn: ViscosityFn,
) -> GridArrayVector:
  """Computes acceleration due to eddy viscosity turbulence model.

  Eddy viscosity models compute a turbulence closure term as a divergence of
  the subgrid-scale stress tensor, which is expressed as velocity dependent
  viscosity times the rate of strain tensor. This module delegates computation
  of the eddy-viscosity to `viscosity_fn` function.

  Args:
    v: velocity field.
    viscosity_fn: function that computes viscosity values at the same offsets
      as strain rate tensor provided as input.

  Returns:
    acceleration of the velocity field `v`.
  """
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('evm_model only valid for periodic BC.')
  grid = grids.consistent_grid(*v)
  bc = boundaries.periodic_boundary_conditions(grid.ndim)
  s_ij = grids.GridArrayTensor([
      [0.5 * (finite_differences.forward_difference(v[i], j) +  # pylint: disable=g-complex-comprehension
              finite_differences.forward_difference(v[j], i))
       for j in range(grid.ndim)]
      for i in range(grid.ndim)])
  viscosity = viscosity_fn(s_ij, v)
  tau = jax.tree_map(lambda x, y: -2. * x * y, viscosity, s_ij)
  return tuple(-finite_differences.divergence(  # pylint: disable=g-complex-comprehension
      tuple(grids.GridVariable(t, bc)  # use velocity bc to compute diverence
            for t in tau[i, :]))
               for i in range(grid.ndim))


# TODO(dkochkov) remove when b/160947162 is resolved.
def implicit_evm_solve_with_diffusion(
    v: GridVariableVector,
    viscosity: float,
    dt: float,
    configured_evm_model: Callable,  # pylint: disable=g-bare-generic
    cg_kwargs: Optional[Mapping[str, Any]] = None
) -> GridVariableVector:
  """Implicit solve for eddy viscosity model combined with diffusion.

  This method is intended to be used with `implicit_diffusion_navier_stokes` to
  avoid potential numerical instabilities associated with fast diffusion modes.

  Args:
    v: current velocity field.
    viscosity: constant viscosity coefficient.
    dt: time step of implicit integration.
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

  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError(
        'implicit_evm_solve_with_diffusion only valid for periodic BC.')
  bc = grids.unique_boundary_conditions(*v)
  vector_laplacian = np.vectorize(finite_differences.laplacian)

  # the arg v from the outer function.
  def linear_op(velocity):
    v_var = tuple(grids.GridVariable(u, bc) for u in velocity)
    acceleration = configured_evm_model(v_var)
    return tuple(
        velocity - dt * (acceleration + viscosity * vector_laplacian(v_var)))

  # We normally prefer fast diagonalization, but that requires an outer
  # product structure for the linear operation, which doesn't hold here.
  # TODO(shoyer): consider adding a preconditioner
  v_prime, _ = jax.scipy.sparse.linalg.cg(linear_op, tuple(u.array for u in v),
                                          **cg_kwargs)
  return tuple(
      grids.GridVariable(u_prime, u.bc) for u_prime, u in zip(v_prime, v))


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
