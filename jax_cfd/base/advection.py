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

"""Module for functionality related to advection."""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import interpolation

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = interpolation.InterpolationFn
# TODO(dkochkov) Consider testing if we need operator splitting methods.


def _advect_aligned(cs: GridVariableVector, v: GridVariableVector) -> GridArray:
  """Computes fluxes and the associated advection for aligned `cs` and `v`.

  The values `cs` should consist of a single quantity `c` that has been
  interpolated to the offset of the components of `v`. The components of `v` and
  `cs` should be located at the faces of a single (possibly offset) grid cell.
  We compute the advection as the divergence of the flux on this control volume.

  The boundary condition on the flux is inherited from the scalar quantity `c`.

  A typical example in three dimensions would have

  ```
  cs[0].offset == v[0].offset == (1., .5, .5)
  cs[1].offset == v[1].offset == (.5, 1., .5)
  cs[2].offset == v[2].offset == (.5, .5, 1.)
  ```

  In this case, the returned advection term would have offset `(.5, .5, .5)`.

  Args:
    cs: a sequence of `GridArray`s; a single value `c` that has been
      interpolated so that it is aligned with each component of `v`.
    v: a sequence of `GridArrays` describing a velocity field. Should be defined
      on the same Grid as cs.

  Returns:
    An `GridArray` containing the time derivative of `c` due to advection by
    `v`.

  Raises:
    ValueError: `cs` and `v` have different numbers of components.
    AlignmentError: if the components of `cs` are not aligned with those of `v`.
  """
  # TODO(jamieas): add more sophisticated alignment checks, ensuring that the
  # values are located on the faces of a control volume.
  if len(cs) != len(v):
    raise ValueError('`cs` and `v` must have the same length;'
                     f'got {len(cs)} vs. {len(v)}.')
  flux = tuple(c.array * u.array for c, u in zip(cs, v))
  bcs = tuple(
      boundaries.get_flux_bc_from_velocity_and_scalar(v[i], cs[i], i)
      for i in range(len(v)))
  flux = tuple(bc.impose_bc(f) for f, bc in zip(flux, bcs))
  return -fd.divergence(flux)


def advect_general(
    c: GridVariable,
    v: GridVariableVector,
    u_interpolation_fn: InterpolationFn,
    c_interpolation_fn: InterpolationFn,
    dt: Optional[float] = None) -> GridArray:
  """Computes advection of a scalar quantity `c` by the velocity field `v`.

  This function follows the following procedure:

    1. Interpolate each component of `v` to the corresponding face of the
       control volume centered on `c`.
    2. Interpolate `c` to the same control volume faces.
    3. Compute the flux `cu` using the aligned values.
    4. Set the boundary condition on flux, which is inhereited from `c`.
    5. Return the negative divergence of the flux.

  Args:
    c: the quantity to be transported.
    v: a velocity field. Should be defined on the same Grid as c.
    u_interpolation_fn: method for interpolating velocity field `v`.
    c_interpolation_fn: method for interpolating scalar field `c`.
    dt: unused time-step.

  Returns:
    The time derivative of `c` due to advection by `v`.
  """
  target_offsets = grids.control_volume_offsets(c)
  aligned_v = tuple(u_interpolation_fn(u, target_offset, v, dt)
                    for u, target_offset in zip(v, target_offsets))
  aligned_c = tuple(c_interpolation_fn(c, target_offset, aligned_v, dt)
                    for target_offset in target_offsets)
  return _advect_aligned(aligned_c, aligned_v)


def advect_linear(c: GridVariable,
                  v: GridVariableVector,
                  dt: Optional[float] = None) -> GridArray:
  """Computes advection using linear interpolations."""
  return advect_general(c, v, interpolation.linear, interpolation.linear, dt)


def advect_upwind(c: GridVariable,
                  v: GridVariableVector,
                  dt: Optional[float] = None) -> GridArray:
  """Computes advection using first-order upwind interpolation on `c`."""
  return advect_general(c, v, interpolation.linear, interpolation.upwind, dt)


def _align_velocities(v: GridVariableVector) -> Tuple[GridVariableVector]:
  """Returns interpolated components of `v` needed for convection.

  Args:
    v: a velocity field.

  Returns:
    A d-tuple of d-tuples of `GridVariable`s `aligned_v`, where `d = len(v)`.
    The entry `aligned_v[i][j]` is the component `v[i]` after interpolation to
    the appropriate face of the control volume centered around `v[j]`.
  """
  grid = grids.consistent_grid(*v)
  offsets = tuple(grids.control_volume_offsets(u) for u in v)
  aligned_v = tuple(
      tuple(interpolation.linear(v[i], offsets[i][j])
            for j in range(grid.ndim))
      for i in range(grid.ndim))
  return aligned_v


def _velocities_to_flux(
    aligned_v: Tuple[GridVariableVector]) -> Tuple[GridVariableVector]:
  """Computes the fluxes across the control volume faces for a velocity field.

  This is the flux associated with the nonlinear term `vv` for velocity `v`.
  The boundary condition on the flux is inherited from `v`.

  Args:
    aligned_v: a d-tuple of d-tuples of `GridVariable`s such that the entry
    `aligned_v[i][j]` is the component `v[i]` after interpolation to
    the appropriate face of the control volume centered around `v[j]`. This is
    the output of `_align_velocities`.

  Returns:
    A tuple of tuples `flux` of `GridVariable`s with the same structure as
    `aligned_v`. The entry `flux[i][j]` is `aligned_v[i][j] * aligned_v[j][i]`.
  """
  ndim = len(aligned_v)
  flux = [tuple() for _ in range(ndim)]
  for i in range(ndim):
    for j in range(ndim):
      if i <= j:
        bc = boundaries.get_flux_bc_from_velocity_and_scalar(
            aligned_v[j][i], aligned_v[i][j], j)
        flux[i] += (bc.impose_bc(aligned_v[i][j].array *
                                 aligned_v[j][i].array),)
      else:
        flux[i] += (flux[j][i],)
  return tuple(flux)


def convect_linear(v: GridVariableVector) -> GridArrayVector:
  """Computes convection/self-advection of the velocity field `v`.

  This function is conceptually equivalent to

  ```
  def convect_linear(v, grid):
    return tuple(advect_linear(u, v, grid) for u in v)
  ```

  However, there are several optimizations to avoid duplicate operations.

  Args:
    v: a velocity field.

  Returns:
    A tuple containing the time derivative of each component of `v` due to
    convection.
  """
  # TODO(jamieas): consider a more efficient vectorization of this function.
  # TODO(jamieas): incorporate variable density.
  aligned_v = _align_velocities(v)
  fluxes = _velocities_to_flux(aligned_v)
  return tuple(-fd.divergence(flux) for flux in fluxes)


def advect_van_leer(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridArray:
  """Computes advection of a scalar quantity `c` by the velocity field `v`.

  Implements Van-Leer flux limiting scheme that uses second order accurate
  approximation of fluxes for smooth regions of the solution. This scheme is
  total variation diminishing (TVD). For regions with high gradients flux
  limitor transformes the scheme into a first order method. For [1] for
  reference. This function follows the following procedure:

    1. Interpolate each component of `v` to the corresponding face of the
       control volume centered on `c`. In most cases satisfied by design.
    2. Computes upwind flux for each direction.
    3. Computes higher order flux correction based on neighboring values of `c`.
    4. Combines fluxes and assigns flux boundary condition.
    5. Returns the negative divergence of fluxes.

  Args:
    c: the quantity to be transported.
    v: a velocity field. Should be defined on the same Grid as c.
    dt: time step for which this scheme is TVD and second order accurate
      in time.

  Returns:
    The time derivative of `c` due to advection by `v`.

  #### References

  [1]:  MIT 18.336 spring 2009 Finite Volume Methods Lecture 19.
        go/mit-18.336-finite_volume_methods-19

  """
  # TODO(dkochkov) reimplement this using apply_limiter method.
  offsets = grids.control_volume_offsets(c)
  aligned_v = tuple(interpolation.linear(u, offset)
                    for u, offset in zip(v, offsets))
  flux = []
  for axis, (u, h) in enumerate(zip(aligned_v, c.grid.step)):
    c_center = c.data
    c_left = c.shift(-1, axis=axis).data
    c_right = c.shift(+1, axis=axis).data
    upwind_flux = grids.applied(jnp.where)(
        u.array > 0, u.array * c_center, u.array * c_right)

    # Van-Leer Flux correction is computed in steps to avoid `nan`s.
    # Formula for the flux correction df for advection with positive velocity is
    # df_{i} = 0.5 * (1-gamma) * dc_{i}
    # dc_{i} = 2(c_{i+1} - c_{i})(c_{i} - c_{i-1})/(c_{i+1}-c_{i})
    # gamma is the courant number = u * dt / h
    diffs_prod = 2 * (c_right - c_center) * (c_center - c_left)
    neighbor_diff = c_right - c_left
    safe = diffs_prod > 0
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    forward_correction = jnp.where(
        safe, diffs_prod / jnp.where(safe, neighbor_diff, 1), 0
    )
    # for negative velocity we simply need to shift the correction along v axis.
    # Cast to GridVariable so that we can apply a shift() operation.
    forward_correction_array = grids.GridVariable(
        grids.GridArray(forward_correction, u.offset, u.grid), u.bc)
    backward_correction_array = forward_correction_array.shift(+1, axis)
    backward_correction = backward_correction_array.data
    abs_velocity = abs(u.array)
    courant_numbers = (dt / h) * abs_velocity
    pre_factor = 0.5 * (1 - courant_numbers) * abs_velocity
    flux_correction = pre_factor * grids.applied(jnp.where)(
        u.array > 0, forward_correction, backward_correction)
    flux.append(upwind_flux + flux_correction)
  # Assign flux boundary condition
  flux_bc = [
      boundaries.get_flux_bc_from_velocity_and_scalar(u, c, direction)
      for direction, u in enumerate(v)
  ]
  flux = tuple(
      flux_bc[axis].impose_bc(flux[axis]) for axis in range(c.grid.ndim))
  advection = -fd.divergence(flux)
  return advection


def advect_step_semilagrangian(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridVariable:
  """Semi-Lagrangian advection of a scalar quantity.

  Note that unlike the other advection functions, this function returns values
  at the next time-step, not the time derivative.

  Args:
    c: the quantity to be transported.
    v: a velocity field. Should be defined on the same Grid as c.
    dt: desired time-step.

  Returns:
    Advected quantity at the next time-step -- *not* the time derivative.
  """
  # Reference: "Learning to control PDEs with Differentiable Physics"
  # https://openreview.net/pdf?id=HyeSin4FPB (see Appendix A)
  grid = grids.consistent_grid(c, *v)

  # TODO(shoyer) Enable lower domains != 0 for this function.
  # Hint: indices = [
  #     -o + (x - l) * n / (u - l)
  #     for (l, u), o, x, n in zip(grid.domain, c.offset, coords, grid.shape)
  # ]
  if not all(d[0] == 0 for d in grid.domain):
    raise ValueError(
        f'Grid domains currently must start at zero. Found {grid.domain}')
  coords = [x - dt * interpolation.linear(u, c.offset).data
            for x, u in zip(grid.mesh(c.offset), v)]
  indices = [x / s - o for s, o, x in zip(grid.step, c.offset, coords)]
  if not boundaries.has_all_periodic_boundary_conditions(c):
    raise NotImplementedError('non-periodic BCs not yet supported')
  c_advected = grids.applied(jax.scipy.ndimage.map_coordinates)(
      c.array, indices, order=1, mode='wrap')
  return GridVariable(c_advected, c.bc)


# TODO(dkochkov) Implement advect_with_flux_limiter method.
# TODO(dkochkov) Consider moving `advect_van_leer` to test based on performance.
def advect_van_leer_using_limiters(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridArray:
  """Implements Van-Leer advection by applying TVD limiter to Lax-Wendroff."""
  c_interpolation_fn = interpolation.apply_tvd_limiter(
      interpolation.lax_wendroff, limiter=interpolation.van_leer_limiter)
  return advect_general(c, v, interpolation.linear, c_interpolation_fn, dt)


def stable_time_step(max_velocity: float,
                     max_courant_number: float,
                     grid: grids.Grid) -> float:
  """Calculate a stable time step size for explicit advection.

  The calculation is based on the CFL condition for advection.

  Args:
    max_velocity: maximum velocity.
    max_courant_number: the Courant number used to choose the time step. Smaller
      numbers will lead to more stable simulations. Typically this should be in
      the range [0.5, 1).
    grid: a `Grid` object.

  Returns:
    The prescribed time interval.
  """
  dx = min(grid.step)
  return max_courant_number * dx / max_velocity
