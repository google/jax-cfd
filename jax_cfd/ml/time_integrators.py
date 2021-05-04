"""Methods for time integration of first order differential equations."""
import gin
import haiku as hk
import jax


# TODO(dkochkov) Include other integrators such as DP; RK methods;
# TODO(dkochkov) Add option to include input as in funcutils.trajectory.


@gin.register
def euler_integrator(
    derivative_module,
    initial_state,
    dt,
    num_steps,
):
  """Integrates ode defined by `derivative_module` using euler method.

  Args:
    derivative_module: hk.Module that computes time derivative.
    initial_state: initial state for time integration.
    dt: time step.
    num_steps: number time steps `dt` to integrate for.

  Returns:
   final state at time `t + num_steps * dt` and `dt` spaced trajectory.
  """
  def _single_step(state, _):
    deriv = derivative_module(state)
    next_state = jax.tree_multimap(lambda x, dxdt: x + dt * dxdt, state, deriv)
    return next_state, next_state

  return hk.scan(_single_step, initial_state, None, num_steps)
