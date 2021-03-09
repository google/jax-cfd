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

"""Non-learned "base" physics routines for JAX-CFD."""

import jax_cfd.base.advection
import jax_cfd.base.array_utils
import jax_cfd.base.diffusion
import jax_cfd.base.equations
import jax_cfd.base.fast_diagonalization
import jax_cfd.base.finite_differences
import jax_cfd.base.forcings
import jax_cfd.base.funcutils
import jax_cfd.base.grids
import jax_cfd.base.initial_conditions
import jax_cfd.base.interpolation
import jax_cfd.base.pressure
import jax_cfd.base.resize
import jax_cfd.base.spectral
import jax_cfd.base.subgrid_models
import jax_cfd.base.validation_problems
