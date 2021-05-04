"""Registry of nonlinearities that can be used in neural networks."""

import gin
import jax
import jax.numpy as jnp


relu = gin.external_configurable(jax.nn.relu)
tanh = gin.external_configurable(jnp.tanh)
softplus = gin.external_configurable(jax.nn.softplus)
swish = gin.external_configurable(jax.nn.swish)
elu = gin.external_configurable(jax.nn.elu)
