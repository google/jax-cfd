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

"""Collocated grid versions of "base" physics routines for JAX-CFD."""

import jax_cfd.collocated.advection
import jax_cfd.collocated.diffusion
import jax_cfd.collocated.equations
import jax_cfd.collocated.initial_conditions
import jax_cfd.collocated.pressure
