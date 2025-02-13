

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jax_ib.base import advection
from jax_ib.base import diffusion
from jax_ib.base import grids
from jax_ib.base import pressure
from jax_cfd.base import pressure as pressureCFD
from jax_ib.base import time_stepping
from jax_ib.base import boundaries
from jax_ib.base import finite_differences
import tree_math
from jax_ib.base import particle_class
from jax_cfd.base import equations as equationsCFD

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]
ForcingFn = Callable[[GridVariableVector], GridArrayVector]
BCFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]
BCFn_new =  Callable[[GridVariableVector, float], GridVariableVector]
IBMFn =  Callable[[particle_class.All_Variables, float], GridVariableVector]
GradPFn = Callable[[GridVariable], GridArrayVector]

PosFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]

DragFn =  Callable[[particle_class.All_Variables], particle_class.All_Variables]


def _wrap_term_as_vector(fun, *, name):
  return tree_math.unwrap(jax.named_call(fun, name=name), vector_argnums=0)


def navier_stokes_explicit_terms(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    forcing: Optional[ForcingFn] = None,
    
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused

  if convect is None:
    def convect(v):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

  def diffuse_velocity(v, *args):
    return tuple(diffuse(u, *args) for u in v)

  convection = _wrap_term_as_vector(convect, name='convection')
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    dv_dt = convection(v)
    if viscosity is not None:
      dv_dt += diffusion_(v, viscosity / density)
    if forcing is not None:
      dv_dt += forcing(v) / density
    
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs





def explicit_Reserve_BC(
    ReserveBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Reserve_boundary(v, *args):
    return ReserveBC(v, *args)
   Reserve_bc_ = _wrap_term_as_vector(Reserve_boundary, name='Reserve_BC')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Reserve_bc(v):
       
       return Reserve_bc_(v,step_time)

   return _Reserve_bc

def explicit_update_BC(
    updateBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Update_boundary(v, *args):
    return updateBC(v, *args)
   Update_bc_ = _wrap_term_as_vector(Update_boundary, name='Update_BC')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Update_bc(v):
       
       return Update_bc_(v,step_time)

   return _Update_bc


def explicit_IBM_Force(
    cal_IBM_force: IBMFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def IBM_FORCE(v, *args):
    return cal_IBM_force(v, *args)
   IBM_FORCE_ = _wrap_term_as_vector(IBM_FORCE, name='IBM_FORCE')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _IBM_FORCE(v):
       
       return IBM_FORCE_(v,step_time)

   return _IBM_FORCE



def explicit_Update_position(
    cal_Update_Position: PosFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Update_Position(v, *args):
    return cal_Update_Position(v, *args)
   Update_Position_ = _wrap_term_as_vector(Update_Position, name='Update_Position')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Update_Position(v):
       
       return Update_Position_(v,step_time)

   return _Update_Position


def explicit_Calc_Drag(
    cal_Drag: DragFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Calculate_Drag(v, *args):
    return cal_Drag(v, *args)
   Calculate_Drag_ = _wrap_term_as_vector(Calculate_Drag, name='Calculate_Drag')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Calculate_Drag(v):
       
       return Calculate_Drag_(v,step_time)

   return _Calculate_Drag

def explicit_Pressure_Gradient(
    cal_Pressure_Grad: GradPFn,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Pressure_Grad(v):
    return cal_Pressure_Grad(v)
   Pressure_Grad_ = _wrap_term_as_vector(Pressure_Grad, name='Pressure_Grad')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Pressure_Grad(v):
       
       return Pressure_Grad_(v)

   return _Pressure_Grad

def semi_implicit_navier_stokes_timeBC(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressureCFD.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler_updated,
    IBM_forcing: IBMFn=None,
    Updating_Position:PosFn=None ,
    Pressure_Grad:GradPFn=finite_differences.forward_difference,
    Drag_fn:DragFn=None,
    
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing)

  pressure_projection = jax.named_call(pressure.projection_and_update_pressure, name='pressure')
  Reserve_BC = explicit_Reserve_BC(ReserveBC = boundaries.Reserve_BC,step_time = dt)
  update_BC = explicit_update_BC(updateBC = boundaries.update_BC,step_time = dt)
  IBM_force = explicit_IBM_Force(cal_IBM_force = IBM_forcing,step_time = dt)
  Update_Position =  explicit_Update_position(cal_Update_Position = Updating_Position,step_time = dt)
  Pressure_Grad =  explicit_Pressure_Gradient(cal_Pressure_Grad = Pressure_Grad)
  Calculate_Drag =  explicit_Calc_Drag(cal_Drag = Drag_fn,step_time = dt)
  #jax.named_call(boundaries.update_BC, name='Update_BC')
  # TODO(jamieas): Consider a scheme where pressure calculations and
  # advection/diffusion are staggered in time.
  ode = time_stepping.ExplicitNavierStokesODE_BCtime(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve),
      update_BC,
      Reserve_BC,
      IBM_force,
      Update_Position,
      Pressure_Grad,
      Calculate_Drag,
  )
  step_fn = time_stepper(ode, dt)
  return step_fn


def semi_implicit_navier_stokes_penalty(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressureCFD.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler_penalty,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing)

  pressure_projection = jax.named_call(pressure.projection_and_update_pressure, name='pressure')
  Reserve_BC = explicit_Reserve_BC(ReserveBC = boundaries.Reserve_BC,step_time = dt)
  update_BC = explicit_update_BC(updateBC = boundaries.update_BC,step_time = dt)
  #jax.named_call(boundaries.update_BC, name='Update_BC')
  # TODO(jamieas): Consider a scheme where pressure calculations and
  # advection/diffusion are staggered in time.
  ode = time_stepping.ExplicitNavierStokesODE_Penalty(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve),
      update_BC,
      Reserve_BC,
  )
  step_fn = time_stepper(ode, dt)
  return step_fn
