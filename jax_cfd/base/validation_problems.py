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

"""Descriptions and analytical solutions for validation problems."""

import abc

from typing import Optional, Sequence, Tuple

from jax_cfd.base import grids
import jax.numpy as jnp
import numpy as np


AlignedArray = grids.AlignedArray
Offsets = Sequence[Sequence[float]]
AlignedField = Tuple[AlignedArray, ...]


class Problem(metaclass=abc.ABCMeta):
  """An abstract class for Navier-Stokes problems."""

  @property
  def grid(self):
    return self._grid

  @property
  def density(self):
    return self._density

  @property
  def viscosity(self):
    return self._viscosity

  def force(self,
            offsets: Optional[Offsets] = None) -> Optional[AlignedField]:
    del offsets  # Unused
    return None

  @abc.abstractmethod
  def velocity(self,
               t: float,
               offsets: Optional[Offsets] = None) -> AlignedField:
    pass


class TaylorGreen(Problem):
  """2D Taylor Green vortices with analytic solution for velocity.

  See https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex.
  """
  # TODO(jamieas): consider parameterizing problems in terms of Reynolds
  # number.

  def __init__(self,
               shape: Tuple[int, int],
               density: float = 1,
               viscosity: float = 0,
               kx: float = 1,
               ky: float = 1):
    self._grid = grids.Grid(shape=shape,
                            domain=[(0., 2. * np.pi),
                                    (0., 2. * np.pi)])
    self._density = density
    self._viscosity = viscosity
    self._kx = kx
    self._ky = ky

  def velocity(
      self,
      t: float = 0,
      offsets: Optional[Offsets] = None) -> AlignedField:
    """Returns an analytic solution for velocity at time `t`."""
    if offsets is None:
      offsets = self.grid.cell_faces

    scale = jnp.exp(-2 * self.viscosity * t)

    ux, uy = self.grid.mesh(offsets[0])
    u = grids.AlignedArray(
        scale * jnp.cos(self._kx * ux) * jnp.sin(self._ky * uy), offsets[0])

    vx, vy = self.grid.mesh(offsets[1])
    v = grids.AlignedArray(
        -scale * jnp.sin(self._kx * vx) * jnp.cos(self._ky * vy), offsets[1])

    return (u, v)

