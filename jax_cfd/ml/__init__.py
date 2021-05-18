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

"""An ML modeling library built on Haiku and Gin-Config for JAX-CFD."""

import jax_cfd.ml.advections
import jax_cfd.ml.decoders
import jax_cfd.ml.diffusions
import jax_cfd.ml.encoders
import jax_cfd.ml.equations
import jax_cfd.ml.forcings
import jax_cfd.ml.interpolations
import jax_cfd.ml.layers
import jax_cfd.ml.layers_util
import jax_cfd.ml.model_builder
import jax_cfd.ml.model_utils
import jax_cfd.ml.networks
import jax_cfd.ml.nonlinearities
import jax_cfd.ml.optimizer_modules
import jax_cfd.ml.physics_specifications
import jax_cfd.ml.pressures
import jax_cfd.ml.tiling
import jax_cfd.ml.time_integrators
import jax_cfd.ml.towers
import jax_cfd.ml.viscosities
