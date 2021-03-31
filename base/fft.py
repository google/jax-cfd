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

"""FFT functions for JAX-CFD.

TPUs don't (yet) implement multi-dimensional FFTs.
"""
import functools

import jax
import jax.numpy as jnp


def _tpu_override(original):
  """Override the implementation of a function on TPUs only."""
  def decorator(override):
    @jax.jit
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
      if all(device.platform == 'tpu' for device in jax.local_devices()):
        return override(*args, **kwargs)
      else:
        return original(*args, **kwargs)
    return wrapper
  return decorator


@_tpu_override(jnp.fft.fftn)
def fftn(array):
  out = array
  for axis in reversed(range(array.ndim)):
    out = jnp.fft.fft(out, axis=axis)
  return out


@_tpu_override(jnp.fft.ifftn)
def ifftn(array):
  out = array
  for axis in range(array.ndim):
    out = jnp.fft.ifft(out, axis=axis)
  return out


@_tpu_override(jnp.fft.rfftn)
def rfftn(array):
  out = jnp.fft.rfft(array, axis=-1)
  for axis in reversed(range(array.ndim - 1)):
    out = jnp.fft.fft(out, axis=axis)
  return out


@_tpu_override(jnp.fft.irfftn)
def irfftn(array):
  out = array
  for axis in range(array.ndim - 1):
    out = jnp.fft.ifft(out, axis=axis)
  out = jnp.fft.irfft(out, axis=-1)
  return out
