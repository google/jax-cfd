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

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids
from jax_cfd.base import interpolation

AlignedArray = grids.AlignedArray
AlignedField = Tuple[AlignedArray, ...]
# TODO(dkochkov) Consider testing if we need operator splitting methods.


def _advect_aligned(cs: AlignedField,
                    v: AlignedField,
                    grid: grids.Grid) -> AlignedArray:
  """Computes fluxes and the associated advection for aligned `cs` and `v`.

  The values `cs` should consist of a single quantity `c` that has been
  interpolated to the offset of the components of `v`. The components of `v` and
  `cs` should be located at the faces of a single (possibly offset) grid cell.
  We compute the advection as the divergence of the flux on this control volume.

  A typical example in three dimensions would have

  ```
  cs[0].offset == v[0].offset == (1., .5, .5)
  cs[1].offset == v[1].offset == (.5, 1., .5)
  cs[2].offset == v[2].offset == (.5, .5, 1.)
  ```

  In this case, the returned advection term would have offset `(.5, .5, .5)`.

  Args:
    cs: a sequence of `AlignedArray`s; a single value `c` that has been
      interpolated so that it is aligned with each component of `v`.
    v: a sequence of `AlignedArrays` describing a velocity field.
    grid: the `Grid` on which `cs` and `v` are located.

  Returns:
    An `AlignedArray` containing the time derivative of `c` due to advection by
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
  uc = tuple(c * u for c, u in zip(cs, v))
  return -fd.divergence(uc, grid)


def advect_general(
    c: AlignedArray,
    v: AlignedField,
    grid: grids.Grid,
    u_interpolation_fn: Callable[..., AlignedArray],
    c_interpolation_fn: Callable[..., AlignedArray],
    dt: float = None
):
  """Computes advection of a scalar quantity `c` by the velocity field `v`.

  This function follows the following procedure:

    1. Interpolate each component of `v` to the corresponding face of the
       control volume centered on `c`.
    2. Interpolate `c` to the same control volume faces.
    3. Compute the flux `cu` using the aligned values.
    4. Return the negative divergence of the flux.

  Args:
    c: the quantity to be transported.
    v: a velocity field.
    grid: the `Grid` on which `c` and `v` are located.
    u_interpolation_fn: method for interpolating velocity field `v`.
    c_interpolation_fn: method for interpolating scalar field `c`.
    dt: unused time-step.

  Returns:
    The time derivative of `c` due to advection by `v`.
  """
  target_offsets = grids.control_volume_offsets(c)
  aligned_v = tuple(u_interpolation_fn(u, target_offset, grid, v, dt)
                    for u, target_offset in zip(v, target_offsets))
  aligned_c = tuple(c_interpolation_fn(c, target_offset, grid, aligned_v, dt)
                    for target_offset in target_offsets)
  res = _advect_aligned(aligned_c, aligned_v, grid)
  return res


def advect_linear(c: AlignedArray,
                  v: AlignedField,
                  grid: grids.Grid,
                  dt: Optional[float] = None) -> AlignedArray:
  """Computes advection using linear interpolations."""
  return advect_general(
      c, v, grid, interpolation.linear, interpolation.linear, dt)


def advect_upwind(c: AlignedArray,
                  v: AlignedField,
                  grid: grids.Grid,
                  dt: Optional[float] = None) -> AlignedArray:
  """Computes advection using first-order upwind interpolation on `c`."""
  return advect_general(
      c, v, grid, interpolation.linear, interpolation.upwind, dt)


def _align_velocities(v: AlignedField,
                      grid: grids.Grid) -> Tuple[AlignedField]:
  """Returns interpolated components of `v` needed for convection.

  Args:
    v: a velocity field.
    grid: the `Grid` on which `v` is located.

  Returns:
    A d-tuple of d-tuples of `AlignedArray`s `aligned_v`, where `d = len(v)`.
    The entry `aligned_v[i][j]` is the component `v[i]` after interpolation to
    the appropriate face of the control volume centered around `v[j]`.
  """
  offsets = tuple(grids.control_volume_offsets(u) for u in v)
  aligned_v = tuple(
      tuple(interpolation.linear(v[i], offsets[i][j], grid)
            for j in range(grid.ndim))
      for i in range(grid.ndim))
  return aligned_v


def _velocities_to_flux(
    aligned_v: Tuple[AlignedField]) -> Tuple[AlignedField]:
  """Computes the fluxes across the control volume faces for a velocity field.

  Args:
    aligned_v: a d-tuple of d-tuples of `AlignedArray`s such that the entry
    `aligned_v[i][j]` is the component `v[i]` after interpolation to
    the appropriate face of the control volume centered around `v[j]`. This is
    the output of `_align_velocities`.

  Returns:
    A tuple of tuples `flux` of `AlignedArray`s with the same structure as
    `aligned_v`. The entry `flux[i][j]` is `aligned_v[i][j] * aligned_v[j][i]`.
  """
  ndim = len(aligned_v)
  flux = [tuple() for _ in range(ndim)]
  for i in range(ndim):
    for j in range(ndim):
      if i <= j:
        flux[i] += (aligned_v[i][j] * aligned_v[j][i],)
      else:
        flux[i] += (flux[j][i],)
  return tuple(flux)


def convect_linear(v: AlignedField,
                   grid: grids.Grid) -> AlignedField:
  """Computes convection/self-advection of the velocity field `v`.

  This function is conceptually equivalent to

  ```
  def convect_linear(v, grid):
    return tuple(advect_linear(u, v, grid) for u in v)
  ```

  However, there are several optimizations to avoid duplicate operations.

  Args:
    v: a velocity field.
    grid: the `Grid` on which `c` and `v` are located.

  Returns:
    A tuple containing the time derivative of each component of `v` due to
    convection.
  """
  # TODO(jamieas): consider a more efficient vectorization of this function.
  # TODO(jamieas): incorporate variable density.
  aligned_v = _align_velocities(v, grid)
  flux = _velocities_to_flux(aligned_v)
  return tuple(-fd.divergence(f, grid) for f in flux)


def advect_van_leer(
    c: AlignedArray,
    v: AlignedField,
    grid: grids.Grid,
    dt: float
) -> AlignedArray:
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
    4. Combines fluxes and returns the negative divergence.

  Args:
    c: the quantity to be transported.
    v: a velocity field.
    grid: the `Grid` on which `c` and `v` are located.
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
  aligned_v = tuple(interpolation.linear(u, offset, grid)
                    for u, offset in zip(v, offsets))

  fluxes = []
  for axis, (u, h) in enumerate(zip(aligned_v, grid.step)):
    c_center = c.data
    c_left = grid.shift(c, -1, axis=axis).data
    c_right = grid.shift(c, +1, axis=axis).data
    upwind_flux = grids.applied(jnp.where)(u > 0, u * c_center, u * c_right)

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
    forward_correction_array = grids.AlignedArray(forward_correction, u.offset)
    backward_correction_array = grid.shift(forward_correction_array, +1, axis)
    backward_correction = backward_correction_array.data
    abs_velocity = abs(u)
    courant_numbers = (dt / h) * abs_velocity
    pre_factor = 0.5 * (1 - courant_numbers) * abs_velocity
    flux_correction = pre_factor * grids.applied(jnp.where)(
        u > 0, forward_correction, backward_correction)
    flux = upwind_flux + flux_correction
    fluxes.append(flux)
  advection = -fd.divergence(fluxes, grid)
  return advection


def advect_step_semilagrangian(
    c: AlignedArray,
    v: AlignedField,
    grid: grids.Grid,
    dt: float
) -> AlignedArray:
  """Semi-Lagrangian advection of a scalar quantity.

  Note that unlike the other advection functions, this function returns values
  at the next time-step, not the time derivative.

  Args:
    c: the quantity to be transported.
    v: a velocity field.
    grid: the `Grid` on which `c` and `v` are located.
    dt: desired time-step.

  Returns:
    Advected quantity at the next time-step -- *not* the time derivative.
  """
  # Reference: "Learning to control PDEs with Differentiable Physics"
  # https://openreview.net/pdf?id=HyeSin4FPB (see Appendix A)
  coords = [x - dt * interpolation.linear(u, c.offset, grid).data
            for x, u in zip(grid.mesh(c.offset), v)]
  indices = [x / s - o for s, o, x in zip(grid.step, c.offset, coords)]
  if set(grid.boundaries) != {grids.PERIODIC}:
    raise NotImplementedError('non-periodic BCs not yet supported')
  c_advected = grids.applied(jax.scipy.ndimage.map_coordinates)(
      c, indices, order=1, mode='wrap')
  return c_advected


# TODO(dkochkov) Implement advect_with_flux_limiter method.
# TODO(dkochkov) Consider moving `advect_van_leer` to test based on performance.
def advect_van_leer_using_limiters(
    c: AlignedArray,
    v: AlignedField,
    grid: grids.Grid,
    dt: float
) -> AlignedArray:
  """Implements Van-Leer advection by applying TVD limiter to Lax-Wendroff."""
  c_interpolation_fn = interpolation.apply_tvd_limiter(
      interpolation.lax_wendroff, limiter=interpolation.van_leer_limiter)
  return advect_general(
      c, v, grid, interpolation.linear, c_interpolation_fn, dt)


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
