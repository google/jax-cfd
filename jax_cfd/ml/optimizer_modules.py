"""Configurable optimizers from JAX."""
import gin
from jax.experimental import optimizers


@gin.configurable
def optimizer(value):
  return value


gin.external_configurable(optimizers.adam)
gin.external_configurable(optimizers.momentum)
gin.external_configurable(optimizers.nesterov)

gin.external_configurable(optimizers.exponential_decay)
gin.external_configurable(optimizers.inverse_time_decay)
gin.external_configurable(optimizers.polynomial_decay)
gin.external_configurable(optimizers.piecewise_constant)
