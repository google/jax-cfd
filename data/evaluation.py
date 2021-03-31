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
"""Utility methods for evaluation of trained models."""
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax_cfd.data import xarray_utils as xr_utils
import numpy as np
import xarray

# pytype complains about valid operations with xarray (e.g., see b/153704639),
# so it isn't worth the trouble of running it.
# pytype: skip-file


def absolute_error(
    array: xarray.DataArray,
    eval_model_name: str = 'learned',
    target_model_name: str = 'ground_truth',
) -> xarray.DataArray:
  """Computes absolute error between to be evaluated and target models.

  Args:
    array: xarray.DataArray that contains model dimension with `eval_model_name`
      and `target_model_name` coordinates.
    eval_model_name: name of the model that is being evaluated.
    target_model_name: name of the model representing the ground truth values.

  Returns:
    xarray.DataArray containing absolute value of errors between
    `eval_model_name` and `target_model_name` models.
  """
  predicted = array.sel(model=eval_model_name)
  target = array.sel(model=target_model_name)
  return abs(predicted - target).rename('_'.join([predicted.name, 'error']))


def state_correlation(
    array: xarray.DataArray,
    eval_model_name: str = 'learned',
    target_model_name: str = 'ground_truth',
    non_state_dims: Tuple[str, ...] = (xr_utils.XR_SAMPLE_NAME,
                                       xr_utils.XR_TIME_NAME),
    non_state_dims_to_average: Tuple[str, ...] = (xr_utils.XR_SAMPLE_NAME,),
) -> xarray.DataArray:
  """Computes normalized correlation of `array` between target and eval models.

  The dimensions of the `array` are expected to consists of state dimensions
  that are interpreted as a vector parametrizing the configuration of the system
  and `non_state_dims`, that optionally are averaged over if included in
  `non_state_dims_to_average`.

  Args:
    array: xarray.DataArray that contains model dimension with `eval_model_name`
      and `target_model_name` coordinates.
    eval_model_name: name of the model that is being evaluated.
    target_model_name: name of the model representing the ground truth values.
    non_state_dims: tuple of dimension names that are not a part of the state.
    non_state_dims_to_average: tuple of `non_state_dims` to average over.

  Returns:
    xarray.DataArray containing normalized correlation between `eval_model_name`
    and `target_model_name` models.
  """
  predicted = array.sel(model=eval_model_name)
  target = array.sel(model=target_model_name)
  state_dims = list(set(predicted.dims) - set(non_state_dims))
  predicted = xr_utils.normalize(predicted, state_dims)
  target = xr_utils.normalize(target, state_dims)
  result = (predicted * target).sum(state_dims).mean(non_state_dims_to_average)
  return result.rename('_'.join([array.name, 'correlation']))


def approximate_quantiles(ds, quantile_thresholds):
  """Approximate quantiles of all arrays in the given xarray.Dataset."""
  # quantiles can't be done in a blocked fashion in the current version of dask,
  # so for now select only the first time step and create a single chunk for
  # each array.
  return ds.isel(time=0).chunk(-1).quantile(q=quantile_thresholds)


def below_error_threshold(
    array: xarray.DataArray,
    threshold: xarray.DataArray,
    eval_model_name: str = 'learned',
    target_model_name: str = 'ground_truth',
) -> xarray.DataArray:
  """Compute if eval model error is below a threshold based on the target."""
  predicted = array.sel(model=eval_model_name)
  target = array.sel(model=target_model_name)
  return abs(predicted - target) <= threshold


def average(
    array: xarray.DataArray,
    ndim: int,
    non_spatial_dims: Tuple[str, ...] = (xr_utils.XR_SAMPLE_NAME,)
) -> xarray.DataArray:
  """Computes spatial and `non_spatial_dims` mean over `array`.

  Since complex values are not supported in netcdf format we currently check if
  imaginary part can be discarded, otherwise an error is raised.

  Args:
    array: xarray.DataArray to take a mean of. Expected to have `ndim` spatial
      dimensions with names as in `xr_utils.XR_SPATIAL_DIMS`.
    ndim: number of spatial dimensions.
    non_spatial_dims: tuple of dimension names to average besides space.

  Returns:
    xarray.DataArray with `ndim` spatial dimensions and `non_spatial_dims`
    reduced to mean values.

  Raises:
    ValueError: if `array` contains non-real imaginary values.

  """
  dims = list(non_spatial_dims) + list(xr_utils.XR_SPATIAL_DIMS[:ndim])
  dims = [dim for dim in dims if dim in array.dims]
  mean_values = array.mean(dims)
  if np.iscomplexobj(mean_values):
    raise ValueError('complex values are not supported.')
  return mean_values


def energy_spectrum_metric(threshold=0.01):
  """Computes an energy spectrum metric that checks if a simulation failed."""
  @jax.jit
  def _energy_spectrum_metric(arr, ground_truth):
    diff = jnp.abs(jnp.log(arr) - jnp.log(ground_truth))
    metric = jnp.sum(jnp.where(ground_truth > threshold, diff, 0), axis=-1)
    cutoff = jnp.sum(
        jnp.where((arr > threshold) & (ground_truth < threshold),
                  jnp.abs(jnp.log(arr)), 0),
        axis=-1)
    return metric + cutoff

  energy_spectrum_ds = lambda a, b: xarray.apply_ufunc(  # pylint: disable=g-long-lambda
      _energy_spectrum_metric, a, b, input_core_dims=[['kx'], ['kx']]).mean(
          dim='sample')
  return energy_spectrum_ds


def u_x_correlation_metric(threshold=0.5):
  """Computes a spacial spectrum metric that checks if a simulation failed."""
  @jax.jit
  def _u_x_correlation_metric(arr, ground_truth):
    diff = (jnp.abs(arr - ground_truth))
    metric = jnp.sum(
        jnp.where(jnp.abs(ground_truth) > threshold, diff, 0), axis=-1)
    cutoff = jnp.sum(
        jnp.where(
            (jnp.abs(arr) > threshold) & (jnp.abs(ground_truth) < threshold),
            jnp.abs(arr), 0),
        axis=-1)
    return metric + cutoff

  u_x_correlation_ds = lambda a, b: xarray.apply_ufunc(   # pylint: disable=g-long-lambda
      _u_x_correlation_metric, a, b, input_core_dims=[['dx'], ['dx']]).mean(
          dim='sample')
  return u_x_correlation_ds


def temporal_autocorrelation(array):
  """Computes temporal autocorrelation of array."""
  dt = array['time'][1] - array['time'][0]
  length = array.sizes['time']
  subsample = max(1, int(1. / dt))

  def _autocorrelation(array):

    def _corr(x, d):
      del x
      arr1 = jnp.roll(array, d, 0)
      ans = arr1 * array
      ans = jnp.sum(
          jnp.where(
              jnp.arange(length).reshape(-1, 1, 1, 1) >= d, ans / length, 0),
          axis=0)
      return d, ans

    _, full_result = jax.lax.scan(_corr, 0, jnp.arange(0, length, subsample))
    return full_result

  full_result = jax.jit(_autocorrelation)(
      jnp.array(array.transpose('time', 'sample', 'x', 'model').u))
  full_result = xarray.Dataset(
      data_vars=dict(t_corr=(['time', 'sample', 'x', 'model'], full_result)),
      coords={
          'dt': np.array(array.time[slice(None, None, subsample)]),
          'sample': array.sample,
          'x': array.x,
          'model': array.model
      })
  return full_result


def u_t_correlation_metric(threshold=0.5):
  """Computes a temporal spectrum metric that checks if a simulation failed."""
  @jax.jit
  def _u_t_correlation_metric(arr, ground_truth):
    diff = (jnp.abs(arr - ground_truth))
    metric = jnp.sum(
        jnp.where(jnp.abs(ground_truth) > threshold, diff, 0), axis=-1)
    cutoff = jnp.sum(
        jnp.where(
            (jnp.abs(arr) > threshold) & (jnp.abs(ground_truth) < threshold),
            jnp.abs(arr), 0),
        axis=-1)
    return jnp.mean(metric + cutoff)

  return _u_t_correlation_metric


def compute_summary_dataset(
    model_ds: xarray.Dataset,
    ground_truth_ds: xarray.Dataset,
    quantile_thresholds: Sequence[float] = (0.1, 1.0),
    non_spatial_dims: Tuple[str, ...] = (xr_utils.XR_SAMPLE_NAME,)
) -> xarray.Dataset:
  """Computes sample and space averaged summaries of predictions and errors.

  Args:
    model_ds: dataset containing trajectories unrolled using the model.
    ground_truth_ds: dataset containing ground truth trajectories.
    quantile_thresholds: quantile thresholds to use for "within error" metrics.
    non_spatial_dims: tuple of dimension names to average besides space.

  Returns:
    xarray.Dataset containing observables and absolute value errors
    averaged over sample and spatial dimensions.
  """
  ndim = ground_truth_ds.attrs['ndim']
  eval_model_name = 'eval_model'
  target_model_name = 'ground_truth'
  combined_dataset = xarray.concat([model_ds, ground_truth_ds], dim='model')
  combined_dataset.coords['model'] = [eval_model_name, target_model_name]
  combined_dataset = combined_dataset.sel(time=slice(None, 500))
  summaries = [combined_dataset[u] for u in xr_utils.XR_VELOCITY_NAMES[:ndim]]
  spectrum = xr_utils.energy_spectrum(combined_dataset).rename(
      'energy_spectrum')
  summaries += [
      xr_utils.kinetic_energy(combined_dataset),
      xr_utils.speed(combined_dataset),
      spectrum,
  ]
  # TODO(dkochkov) Check correlations in NS and enable it for 2d and 3d.
  if ndim == 1:
    correlations = xr_utils.velocity_spatial_correlation(combined_dataset, 'x')
    time_correlations = temporal_autocorrelation(combined_dataset)
    summaries += [correlations[variable] for variable in correlations]
    u_x_corr_sum = [
        xarray.DataArray((u_x_correlation_metric(threshold)(    # pylint: disable=g-complex-comprehension
            correlations.sel(model=eval_model_name),
            correlations.sel(model=target_model_name))).u_x_correlation)
        for threshold in [0.5]
    ]
    if not time_correlations.t_corr.isnull().any():
      # autocorrelation is a constant, so it is expanded to be part of summaries
      u_t_corr_sum = [
          xarray.ones_like(u_x_corr_sum[0]).rename('autocorrelation') *   # pylint: disable=g-complex-comprehension
          u_t_correlation_metric(threshold)(
              jnp.array(time_correlations.t_corr.sel(model=eval_model_name)),
              jnp.array(time_correlations.t_corr.sel(model=target_model_name)))
          for threshold in [0.5]
      ]
    else:
      # if the trajectory goes to nan, it just reports a large number
      u_t_corr_sum = [
          xarray.ones_like(u_x_corr_sum[0]).rename('autocorrelation') * np.infty
          for threshold in [0.5]
      ]
    energy_sum = [
        energy_spectrum_metric(threshold)(   # pylint: disable=g-complex-comprehension
            spectrum.sel(model=eval_model_name, kx=slice(0, spectrum.kx.max())),
            spectrum.sel(
                model=target_model_name,
                kx=slice(0, spectrum.kx.max()))).rename('energy_spectrum_%f' %
                                                        threshold)
        for threshold in [0.001, 0.01, 0.1, 1.0, 10]
    ]  # pylint: disable=g-complex-comprehension
    custom_summaries = u_x_corr_sum + energy_sum + u_t_corr_sum
  if ndim == 2:
    summaries += [
        xr_utils.enstrophy_2d(combined_dataset),
        xr_utils.vorticity_2d(combined_dataset),
        xr_utils.isotropic_energy_spectrum(
            combined_dataset,
            average_dims=non_spatial_dims).rename('energy_spectrum')
    ]
  if ndim >= 2:
    custom_summaries = []

  mean_summaries = [
      average(s.sel(model=eval_model_name), ndim).rename(s.name + '_mean')
      for s in summaries
  ]
  error_summaries = [
      average(absolute_error(s, eval_model_name, target_model_name), ndim)
      for s in summaries
  ]
  correlation_summaries = [
      state_correlation(s, eval_model_name, target_model_name)
      for s in summaries
      if s.name in xr_utils.XR_VELOCITY_NAMES + ('vorticity',)
  ]

  summaries_ds = xarray.Dataset({array.name: array for array in summaries})
  thresholds = approximate_quantiles(
      summaries_ds, quantile_thresholds=quantile_thresholds).compute()

  threshold_summaries = []
  for threshold_quantile in quantile_thresholds:
    for summary in summaries:
      name = summary.name
      error_threshold = thresholds[name].sel(
          quantile=threshold_quantile, drop=True)
      below_error = below_error_threshold(summary, error_threshold,
                                          eval_model_name, target_model_name)
      below_error.name = f'{name}_within_q={threshold_quantile}'
      threshold_summaries.append(average(below_error, ndim))

  all_summaries = (
      mean_summaries + error_summaries + threshold_summaries +
      correlation_summaries + custom_summaries)
  return xarray.Dataset({array.name: array for array in all_summaries})
