# JAX-CFD: Computational Fluid Dynamics in JAX

Authors: Dmitrii Kochkov, Jamie A. Smith, Stephan Hoyer

JAX-CFD is an experimental research project for exploring the potential of
machine learning, automatic differentiation and hardware accelerators (GPU/TPU)
for computational fluid dynamics. It is implemented in
[JAX](https://github.com/google/jax).

To learn more about our general approach, read our paper [Machine learning accelerated computational fluid dynamics](https://arxiv.org/abs/2102.01010).

## Getting started

Take a look at "demo" notebook in the "notebooks" directory for an example of
how to use JAX-CFD for simulate a 2D turbulent flow.

We are currently preparing more example notebooks, inculding:

- Reusing our training data and/or evaluation setup (without running JAX-CFD)
- Simulations using our pre-trained turbulence models.
- Training a simple hybrid ML + CFD model from scratch

## Organization

JAX-CFD is organized around sub-modules:

- `jax_cfd.base`: core numerical methods for CFD, written in JAX.
- `jax_cfd.ml` (not yet released): machine learning augmented models for CFD,
  written in JAX and [Haiku](https://dm-haiku.readthedocs.io/en/latest/).
- `jax_cfd.data`: data processing utilities for preparing, evaluating and
  post-processing data created with JAX-CFD, written in
  [Xarray](http://xarray.pydata.org/) and
  [Pillow](https://pillow.readthedocs.io/).

## Numerics

JAX-CFD is currently focused on unsteady turbulent flows:

- *Spatial discretization*: Finite volume/difference methods on a staggered
  grid (the "Arakawa C" or "MAC" grid) with pressure at the center of each cell
  and velocity components defined on corresponding faces.
- *Temporal discretization*: Currently only first-order temporal
  discretization, using explicit time-stepping for advection and either implicit
  or explicit time-stepping for diffusion.
- *Pressure solves*: Either CG or fast diagonalization with real-valued FFTs
  (suitable for periodic boundary conditions).
- *Boundary conditions*: Currently only periodic boundary conditions are
  supported.
- *Advection*: We implement 2nd order accurate "Van Leer" schemes.
- *Closures*: We currently implement Smagorinsky eddy-viscosity models.

TODO: add a notebook explaining our numerical models in more depth.

In the long term, we're interested in expanding JAX-CFD to implement methods
relevant for related research, e.g.,

- Spectral methods
- Colocated grids
- Alternative boundary conditions (e.g., non-periodic boundaries and immersed
  boundary methods)
- Higher order time-stepping
- Geometric multigrid
- Steady state simulation (e.g., RANS)
- Distributed simulations across multiple TPUs/GPUs

We would welcome collaboration on any of these! Please reach out (either on
GitHub or by email) to coordinate before starting significant work.

## Other awesome projects

Other differentiable CFD codes compatible with deep learning:

- [PhiFlow](https://github.com/tum-pbs/PhiFlow/) supports TensorFlow, PyTorch and JAX
- [Fluid simulation in Autograd](https://github.com/HIPS/autograd#end-to-end-examples)

JAX for science:

- [JAX-MD](https://github.com/google/jax-md)
- [JAX-DFT](https://github.com/google-research/google-research/tree/master/jax_dft)
- [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo)

Did we miss something? Please let us know!

## Citation

```
@misc{kochkov2021machine,
      title={Machine learning accelerated computational fluid dynamics},
      author={Dmitrii Kochkov and Jamie A. Smith and Ayya Alieva and Qing Wang and Michael P. Brenner and Stephan Hoyer},
      year={2021},
      eprint={2102.01010},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn}
}
```
