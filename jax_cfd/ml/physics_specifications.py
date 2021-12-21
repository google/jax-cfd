"""Modules with PhysicsSpecifications for various equations.

To ensure that all components of the pipeline obtain the expected PhysicsSpecs
all modules (except specializing on a particular equation) must instantiate
PhysicsSpecs objects using `get_physics_specs`, which should be
configured appropriately.
"""

import dataclasses
from typing import Optional

import gin
from jax_cfd.ml import forcings


ForcingModule = forcings.ForcingModule


@gin.configurable
def get_physics_specs(physics_specs_cls=gin.REQUIRED):
  """Returns an instance of `physics_specs_cls`, configured by gin."""
  return physics_specs_cls()


@gin.register
@dataclasses.dataclass
class BasePhysicsSpecs:
  """Base class for keeping physical parameters and forcing module."""
  forcing_module: Optional[ForcingModule]


@gin.register
@dataclasses.dataclass
class KsPhysicsSpecs(BasePhysicsSpecs):
  """Configurable physical parameters for Kuramoto-Sivashinsky models."""


@gin.register
@dataclasses.dataclass
class NavierStokesPhysicsSpecs(BasePhysicsSpecs):
  """Configurable physical parameters and modules for Navier-Stokes models."""
  density: float
  viscosity: float


@gin.configurable
@dataclasses.dataclass
class SpectralNavierStokesPhysicsSpecs(BasePhysicsSpecs):
  viscosity: float
  drag: float
  smooth: bool
