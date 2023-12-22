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
from jax_cfd.base import diffusion
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


def centered_strain_rate_tensor(
    v: grids.GridVariableVector,
) -> grids.GridArrayTensor:
  """Computes centered strain rate tensor.

  Args:
    v: velocity field.

  Returns:
    centered strain rate tensor.
  """

  grid = grids.consistent_grid(*v)
  v_centered = tuple(interpolation.linear(u, grid.cell_center) for u in v)

  def make_strain_rate_bc(v):
    #  viscocity vanishes at the boundary.
    types = []
    for ax in range(grid.ndim):
      if v[0].bc.types[ax][0] == boundaries.BCType.PERIODIC:
        types.append((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC))
      elif v[0].bc.types[ax][0] == boundaries.BCType.DIRICHLET and v[
          0].bc.types[ax][1] == boundaries.BCType.DIRICHLET:
        types.append((boundaries.BCType.DIRICHLET, boundaries.BCType.DIRICHLET))
      else:
        raise ValueError(
            f'boundary condition {v[0].bc.types[ax]} is not implemented')
    return boundaries.HomogeneousBoundaryConditions(types)

  strain_rate_bc = make_strain_rate_bc(v)
  s_ij = grids.GridArrayTensor([[  # pylint: disable=g-complex-comprehension
      strain_rate_bc.impose_bc(
          0.5 * (finite_differences.central_difference(v_centered[i], j) +
                 finite_differences.central_difference(v_centered[j], i)))
      for j in range(grid.ndim)] for i in range(grid.ndim)])
  return s_ij


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
  del dt, interpolate_fn
  grid = grids.consistent_grid(*s_ij.ravel(), *v)
  s_ij_offsets = [array.offset for array in s_ij.ravel()]
  unique_offsets = list(set(s_ij_offsets))
  unique_offsets = list(
      set([tuple(abs(o) for o in offset) for offset in unique_offsets]))
  if len(unique_offsets) > 1 or unique_offsets[0] != grid.cell_center:
    raise ValueError('This function requires cell-centered strain rate tensor.')
  # geometric average
  cutoff = np.prod(np.array(grid.step))**(1 / grid.ndim)
  s_ij_array = grids.GridArrayTensor(
      [[s_ij[i, j].array for j in range(grid.ndim)] for i in range(grid.ndim)])
  s_abs = np.sqrt(
      2 * np.trace(s_ij_array.dot(s_ij_array)))
  viscosity = (cs * cutoff)**2
  return viscosity * s_abs


def evm_model(
    v: GridVariableVector,
    viscosity_fn: ViscosityFn,
) -> GridVariableVector:
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
  s_ij = centered_strain_rate_tensor(v)
  viscosity = viscosity_fn(s_ij, v)
  tau = grids.GridArrayTensor([
      [s.bc.impose_bc(2 * viscosity * s.array) for s in s_ij[i]]
      for i in range(grid.ndim)])
  strain_rate_div = tuple(-finite_differences.centered_divergence(tau[i])
                          for i in range(grid.ndim))
  homogeneous_bc = lambda u: boundaries.HomogeneousBoundaryConditions(u.bc.types   # pylint: disable=g-long-lambda
                                                                     )
  strain_rate_div = tuple(
      homogeneous_bc(u).impose_bc(strain_div)
      for u, strain_div in zip(v, strain_rate_div))
  return tuple(
      interpolation.linear(strain_div, u.offset)
      for strain_div, u in zip(strain_rate_div, v))


# TODO(dkochkov) remove when b/160947162 is resolved.
def implicit_evm_solve_with_diffusion(
    v: GridVariableVector,
    viscosity: float,
    dt: float,
    configured_evm_model: Callable,  # pylint: disable=g-bare-generic
    cg_kwargs: Optional[Mapping[str, Any]] = None,
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
  bc_vals = tuple(u.bc for u in v)
  offset_vals = tuple(u.offset for u in v)
  vector_laplacian = finite_differences.laplacian
  homogeneous_bc = tuple(
      boundaries.HomogeneousBoundaryConditions(bc_val.types)
      for bc_val in bc_vals)
  v_with_homogeneous_bc_array = tuple(
      diffusion.rhs_transform(u.array, bc_val, True)
      for u, bc_val in zip(v, bc_vals))
  v_with_homogeneous_bc = tuple(
      bc.impose_bc(u)
      for u, bc in zip(v_with_homogeneous_bc_array, homogeneous_bc))
  # the arg v from the outer function.
  def linear_op(velocity):
    v_var = tuple(
        bc.pad_and_impose_bc(u, offset)
        for u, bc, offset in zip(velocity, homogeneous_bc, offset_vals))
    acceleration = configured_evm_model(v_var)
    return tuple(v.trim_boundary() - dt *
                 (a.trim_boundary() + viscosity * vector_laplacian(v))
                 for v, a in zip(v_var, acceleration))

  # We normally prefer fast diagonalization, but that requires an outer
  # product structure for the linear operation, which doesn't hold here.
  # TODO(shoyer): consider adding a preconditioner
  v_prime, _ = jax.scipy.sparse.linalg.cg(
      linear_op, tuple(u.trim_boundary() for u in v_with_homogeneous_bc),
      **cg_kwargs)
  v_prime = tuple(
      interpolation.linear(bc.pad_and_impose_bc(u_prime), o)
      for u_prime, bc, o in zip(v_prime, homogeneous_bc, offset_vals))
  v_prime_with_constant_bc_array = tuple(
      diffusion.rhs_transform(u.array, bc_val, False)
      for u, bc_val in zip(v_prime, bc_vals))
  return tuple(
      bc.impose_bc(u) for u, bc in zip(v_prime_with_constant_bc_array, bc_vals))


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
  def smagorinsky_acceleration_array(v):
    v = smagorinsky_acceleration(v)
    return tuple(u.array for u in v)
  if forcing is None:
    forcing = smagorinsky_acceleration_array
  else:
    forcing = forcings.sum_forcings(forcing, smagorinsky_acceleration_array)
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
