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
"""Utility functions for xarray datasets, naming and metadata.

Note on metadata conventions:

When we store data onto xarray.Dataset objects, we are (currently) a little
sloppy about coordinate metadata: we store only a single array for each set of
coordinate values, even though components of our velocity fields are typically
staggered. This is convenient for quick-and-dirty analytics, but means that
variables at the "same" coordinates location may actually be dislocated by any
offset within the unit cell.
"""
import functools
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax_cfd.base import grids
import numpy as np
import pandas
import xarray

# pytype complains about valid operations with xarray (e.g., see b/153704639),
# so it isn't worth the trouble of running it.
# pytype: skip-file


#
# xarray `Dataset` names for coordinates and attributes.
#

XR_VELOCITY_NAMES = ('u', 'v', 'w')
XR_SPATIAL_DIMS = ('x', 'y', 'z')
XR_WAVENUMBER_DIMS = ('kx', 'ky', 'kz')
XR_SAMPLE_NAME = 'sample'
XR_TIME_NAME = 'time'
XR_OFFSET_NAME = 'offset'

XR_SAVE_GRID_SIZE_ATTR_NAME = 'save_grid_size'
XR_DOMAIN_SIZE_NAME = 'domain_size'
XR_NDIM_ATTR_NAME = 'ndim'
XR_STABLE_TIME_STEP_ATTR_NAME = 'stable_time_step'


def velocity_trajectory_to_xarray(
    trajectory: Tuple[Union[grids.Array, grids.AlignedArray], ...],
    grid: grids.Grid = None,
    time: np.ndarray = None,
    attrs: Dict[str, Any] = None,
    samples: bool = False,
    prefix_name: str = '',
) -> xarray.Dataset:
  """Convert a trajectory of velocities to an xarray `Dataset`."""
  dimension = len(trajectory)
  dims = (((XR_SAMPLE_NAME,) if samples else ())
          + (XR_TIME_NAME,)
          + XR_SPATIAL_DIMS[:dimension])

  data_vars = {}
  for component in range(dimension):
    name = XR_VELOCITY_NAMES[component]
    data = trajectory[component]
    if isinstance(data, grids.AlignedArray):
      data = data.data
    var_attrs = {}
    if grid is not None:
      var_attrs[XR_OFFSET_NAME] = grid.cell_faces[component]
    data_vars[prefix_name + name] = xarray.Variable(dims, data, var_attrs)

  if samples:
    num_samples = next(iter(data_vars.values())).shape[0]
    sample_ids = np.arange(num_samples)
  else:
    sample_ids = None
  coords = construct_coords(grid, time, sample_ids)

  return xarray.Dataset(data_vars, coords, attrs)


def construct_coords(
    grid: Optional[grids.Grid] = None,
    times: Optional[np.ndarray] = None,
    sample_ids: Optional[np.ndarray] = None,
) -> Mapping[str, np.ndarray]:
  """Create coordinate arrays."""
  coords = {}
  if grid is not None:
    axes = grid.axes(grid.cell_center)
    coords.update({dim: ax for dim, ax in zip(XR_SPATIAL_DIMS, axes)})
  if times is not None:
    coords[XR_TIME_NAME] = times
  if sample_ids is not None:
    coords[XR_SAMPLE_NAME] = sample_ids
  return coords


def grid_from_attrs(dataset_attrs) -> grids.Grid:
  """Constructs a `Grid` object from dataset attributes."""
  grid_size = dataset_attrs[XR_SAVE_GRID_SIZE_ATTR_NAME]
  ndim = dataset_attrs[XR_NDIM_ATTR_NAME]
  grid_shape = (grid_size,) * ndim
  if XR_DOMAIN_SIZE_NAME in dataset_attrs:
    domain_size = dataset_attrs[XR_DOMAIN_SIZE_NAME]
  elif 'domain_size_multiple' in dataset_attrs:
    # TODO(shoyer): remove this legacy case, once we no longer use datasets
    # generated prior to 2020-09-18
    domain_size = 2 * np.pi * dataset_attrs['domain_size_multiple']
  else:
    raise ValueError(
        f'could not figure out domain size from attrs:\n{dataset_attrs}')
  grid_domain = [(0, domain_size)] * ndim
  grid = grids.Grid(grid_shape, domain=grid_domain)
  return grid


def vorticity_2d(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate vorticity on a 2D dataset."""
  # Vorticity is calculated from staggered velocities at offset=(1, 1).
  dy = ds.y[1] - ds.y[0]
  dx = ds.x[1] - ds.x[0]
  dv_dx = (ds.v.roll(x=-1, roll_coords=False) - ds.v) / dx
  du_dy = (ds.u.roll(y=-1, roll_coords=False) - ds.u) / dy
  return (dv_dx - du_dy).rename('vorticity')


def enstrophy_2d(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate entrosphy over a 2D dataset."""
  return (vorticity_2d(ds) ** 2 / 2).rename('enstrophy')


def magnitude(
    u: xarray.DataArray,
    v: Optional[xarray.DataArray] = None,
    w: Optional[xarray.DataArray] = None,
) -> xarray.DataArray:
  """Calculate the magnitude of a velocity field."""
  total = sum((c * c.conj()).real for c in [u, v, w] if c is not None)
  return total ** 0.5


def speed(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate speed at each point in a velocity field."""
  args = [ds[k] for k in XR_VELOCITY_NAMES if k in ds]
  return magnitude(*args).rename('speed')


def kinetic_energy(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate kinetic energy at each point in a velocity field."""
  return (speed(ds) ** 2 / 2).rename('kinetic_energy')


def fourier_transform(array: xarray.DataArray) -> xarray.DataArray:
  """Calculate the fourier transform of an array, with labeled coordinates."""
  # TODO(shoyer): consider switching to use xrft? https://github.com/xgcm/xrft
  dims = [dim for dim in XR_SPATIAL_DIMS if dim in array.dims]
  axes = [-1, -2, -3][:len(dims)]
  result = xarray.apply_ufunc(
      functools.partial(np.fft.fftn, axes=axes), array,
      input_core_dims=[dims],
      output_core_dims=[['k' + d for d in dims]],
      output_sizes={'k' + d: array.sizes[d] for d in dims},
      output_dtypes=[np.complex128],
      dask='parallelized')
  for d in dims:
    step = float(array.coords[d][1] - array.coords[d][0])
    freqs = 2 * np.pi * np.fft.fftfreq(array.sizes[d], step)
    result.coords['k' + d] = freqs
  # Ensure frequencies are in ascending order (equivalent to fftshift)
  rolls = {'k' + d: array.sizes[d] // 2 for d in dims}
  return result.roll(rolls, roll_coords=True)


def periodic_correlate(u, v):
  """Periodic correlation of arrays `u`, `v`, implemented using the FFT."""
  return np.fft.ifft(np.fft.fft(u).conj() * np.fft.fft(v)).real


def spatial_autocorrelation(array, spatial_axis='x'):
  """Computes spatial autocorrelation of `array` along `spatial_axis`."""
  spatial_axis_size = array.sizes[spatial_axis]
  out_axis_name = 'd' + spatial_axis
  full_result = xarray.apply_ufunc(
      lambda x: periodic_correlate(x, x) / spatial_axis_size, array,
      input_core_dims=[[spatial_axis]],
      output_core_dims=[[out_axis_name]])
  # we only report the unique half of the autocorrelation.
  num_unique_displacements = spatial_axis_size // 2
  result = full_result.isel({out_axis_name: slice(0, num_unique_displacements)})
  displacement_coords = array.coords[spatial_axis][:num_unique_displacements]
  result.coords[out_axis_name] = (out_axis_name, displacement_coords)
  return result


@functools.partial(jax.jit, static_argnums=(0,), backend='cpu')
def _jax_numpy_add_at_zeros(shape, indices, values):
  result = jnp.zeros(shape, dtype=values.dtype)
  # equivalent to np.add.at(result, (..., indices), array), but much faster
  return result.at[..., indices].add(values)


def _binned_sum_numpy(
    array: np.ndarray,
    indices: np.ndarray,
    num_bins: int,
) -> np.ndarray:
  """NumPy helper function for summing over bins."""
  mask = np.logical_not(np.isnan(indices))
  int_indices = indices[mask].astype(int)
  shape = array.shape[:-indices.ndim] + (num_bins,)
  result = _jax_numpy_add_at_zeros(shape, int_indices, array[..., mask])
  return np.asarray(result)


def groupby_bins_sum(
    array: xarray.DataArray,
    group: xarray.DataArray,
    bins: np.ndarray,
    labels: np.ndarray,
) -> xarray.DataArray:
  """Faster equivalent of Xarray's groupby_bins(...).sum()."""
  # TODO(shoyer): remove this in favor of groupby_bin() once xarray's
  # implementation is improved: https://github.com/pydata/xarray/issues/4473
  bin_name = group.name + '_bins'
  indices = group.copy(
      data=pandas.cut(np.ravel(group), bins, labels=False).reshape(group.shape)
  )
  result = xarray.apply_ufunc(
      _binned_sum_numpy, array, indices,
      input_core_dims=[indices.dims, indices.dims],
      output_core_dims=[[bin_name]],
      output_dtypes=[array.dtype],
      output_sizes={bin_name: labels.size},
      kwargs={'num_bins': bins.size - 1},
      dask='parallelized',
  )
  result[bin_name] = labels
  return result


def _isotropize_binsum(ndim, energy):
  """Calculate energy spectrum summing over bins in wavenumber space."""
  wavenumbers = [energy[name] for name in XR_WAVENUMBER_DIMS[:ndim]]
  k_spacing = max(float(k[1] - k[0]) for k in wavenumbers)
  k_max = min(float(w.max()) for w in wavenumbers) - 0.5 * k_spacing
  k = magnitude(*wavenumbers).rename('k')

  bounds = k_spacing * (np.arange(1, round(k_max / k_spacing) + 2) - 0.5)
  labels = k_spacing * np.arange(1, round(k_max / k_spacing) + 1)
  binned = groupby_bins_sum(energy, k, bounds, labels)
  spectrum = binned.rename(k_bins='k')
  return spectrum


def _isotropize_interpolation_2d(
    energy, interpolation_method, num_quadrature_points,
):
  """Caclulate energy spectrum of a 2D signal with interpolation."""
  # Calculate even spaced discrete levels for wavenumber magnitude
  wavenumbers = [energy[name] for name in XR_WAVENUMBER_DIMS[:2]]
  k_spacing = max(float(k[1] - k[0]) for k in wavenumbers)
  k_max = min(float(w.max()) for w in wavenumbers) - 0.5 * k_spacing

  k_values = k_spacing * np.arange(1, round(k_max / k_spacing) + 1)
  k = xarray.DataArray(k_values, dims='k', coords={'k': k_values})

  angle_values = np.linspace(
      0, 2 * np.pi, num=num_quadrature_points, endpoint=False)
  angle = xarray.DataArray(angle_values, dims='angle')

  # Sample the spectrum at each point on the boundary of the circle with
  # with radius k
  kx = k * np.cos(angle)
  ky = k * np.sin(angle)

  # Interpolation on log(energy), which is much smoother in wavenumber space
  # than the energy itself (which decays quite rapidly)
  density = np.exp(
      np.log(energy).interp(kx=kx, ky=ky, method=interpolation_method)
  )

  # Integrate over the edge of each circle
  spectrum = 2 * np.pi * k * density.mean('angle')
  return spectrum


def isotropize(
    array: xarray.DataArray,
    method: Optional[str] = None,
    interpolation_method: str = 'linear',
    num_quadrature_points: int = 100,
) -> xarray.DataArray:
  """Isotropize an ND spectrum by averaging over all angles.

  Args:
    array: array to isotropically average, with one or more dimensions
      correspondings to wavenumbers.
    method: either "interpolation" or "binsum".
    interpolation_method: either "linear" or "nearest". Only used if using
      method="interpolation".
    num_quadrature_points: number of points to use when integrating over
      wavenumbers with method="interpolation".

  Returns:
    Energy spectra as a function of wavenumber magnitude.
  """
  ndim = sum(dim in array.dims for dim in XR_WAVENUMBER_DIMS)
  if ndim == 0:
    raise ValueError(f'no frequency dimensions found: {array.dims}')

  if method is None:
    method = 'interpolation' if ndim == 2 else 'binsum'

  if method == 'interpolation':
    if ndim != 2:
      raise ValueError('interpolation not yet supported for non-2D inputs')
    # TODO(shoyer): switch to more accurate algorithms for both 1D and 3D, too:
    # - 1D can simply add up the energy at positive and negative frequencies
    # - 3D can efficiently integrate over all angles using Lebedev quadrature:
    #   https://en.wikipedia.org/wiki/Lebedev_quadrature
    return _isotropize_interpolation_2d(
        array, interpolation_method, num_quadrature_points)
  elif method == 'binsum':
    # NOTE(shoyer): I believe this function is equivalent to
    # xrft.isotropize(), but is faster & more efficient because we
    # use groupby_bins_sum(). See https://github.com/xgcm/xrft/issues/9
    return _isotropize_binsum(ndim, array)
  else:
    raise ValueError(f'invalid method: {method}')


def energy_spectrum(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate the kinetic energy spectra at each wavenumber.

  Args:
    ds: dataset with `u`, `v` and/or `w` velocity components and corresponding
      spatial dimensions.

  Returns:
    Energy spectra as a function of wavenumber instead of space.
  """
  ndim = sum(dim in ds.dims for dim in 'xyz')
  velocity_components = list(XR_VELOCITY_NAMES[:ndim])
  fourier_ds = ds[velocity_components].map(fourier_transform)
  return kinetic_energy(fourier_ds)


def isotropic_energy_spectrum(
    ds: xarray.Dataset,
    average_dims: Tuple[str, ...] = (),
) -> xarray.DataArray:
  """Calculate the energy spectra at each scalar wavenumber.

  Args:
    ds: dataset with `u`, `v` and/or `w` velocity components and corresponding
      spatial dimensions.
    average_dims: dimensions to average over before isotropic averaging.

  Returns:
    Energy spectra as a function of wavenumber magnitude, without spatial
    dimensions.
  """
  return isotropize(energy_spectrum(ds).mean(average_dims))


def velocity_spatial_correlation(
    ds: xarray.Dataset,
    axis: str
) ->xarray.Dataset:
  """Computes velocity correlation along `axis` for all velocity components."""
  ndim = sum(dim in ds.dims for dim in 'xyz')
  velocity_components = list(XR_VELOCITY_NAMES[:ndim])
  correlation_fn = lambda x: spatial_autocorrelation(x, axis)
  correlations = ds[velocity_components].map(correlation_fn)
  name_mapping = {u: '_'.join([u, axis, 'correlation'])
                  for u in velocity_components}
  return correlations.rename(name_mapping)


def normalize(array: xarray.DataArray, state_dims: Tuple[str, ...]):
  """Returns `array` with slices along `state_dims` normalized to unity."""
  norm = np.sqrt((array ** 2).sum(state_dims))
  return array / norm
