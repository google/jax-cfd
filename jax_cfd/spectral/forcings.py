"""Forcing functions for spectral equations."""

import jax
import jax.numpy as jnp
from jax_cfd.base import grids


def random_forcing_module(grid: grids.Grid,
                          seed: int = 0,
                          n: int = 20,
                          offset=(0,)):
  """Implements the forcing described in Bar-Sinai et al. [*].

  Args:
    grid: grid to use for the x-axis
    seed: random seed for computing the random waves
    n: number of random waves to use
    offset: offset for the x-axis. Defaults to (0,) for the Fourier basis.
  Returns:
    Time dependent forcing function.

  [*] Bar-Sinai, Yohai, Stephan Hoyer, Jason Hickey, and Michael P. Brenner.
  "Learning data-driven discretizations for partial differential equations."
  Proceedings of the National Academy of Sciences 116, no. 31 (2019):
  15344-15349.
  """

  key = jax.random.PRNGKey(seed)

  ks = jnp.array([3, 4, 5, 6])

  key, subkey = jax.random.split(key)
  kx = jax.random.choice(subkey, ks, shape=(n,))

  key, subkey = jax.random.split(key)
  amplitude = jax.random.uniform(subkey, minval=-0.5, maxval=0.5, shape=(n,))

  key, subkey = jax.random.split(key)
  omega = jax.random.uniform(subkey, minval=-0.4, maxval=0.4, shape=(n,))

  key, subkey = jax.random.split(key)
  phi = jax.random.uniform(subkey, minval=0, maxval=2 * jnp.pi, shape=(n,))

  xs, = grid.axes(offset=offset)

  def forcing_fn(t):

    @jnp.vectorize
    def eval_force(x):
      f = amplitude * jnp.sin(omega * t - x * kx + phi)
      return f.sum()

    return eval_force(xs)

  return forcing_fn
