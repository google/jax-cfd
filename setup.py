# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup JAX-CFD."""
import setuptools

base_requires = ['jax', 'numpy', 'scipy']
data_requires = ['matplotlib', 'seaborn', 'Pillow', 'xarray']
ml_requires = ['dm-haiku', 'einops', 'gin-config']
tests_requires = ['absl-py', 'pytest', 'pytest-xdist', 'scikit-image']

setuptools.setup(
    name='jax-cfd',
    version='0.1.0',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=base_requires,
    extras_require={
        'data': data_requires,
        'ml': ml_requires,
        'tests': tests_requires,
        'complete': data_requires + ml_requires + tests_requires,
    },
    url='https://github.com/google/jax-cfd',
    packages=setuptools.find_packages(),
    python_requires='>=3',
)
