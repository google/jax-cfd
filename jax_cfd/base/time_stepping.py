import dataclasses
from typing import Callable, Sequence, TypeVar
import jax
import tree_math
from jax_ib.base import boundaries
from jax_ib.base import grids
from jax_cfd.base import time_stepping
from jax_ib.base import particle_class


PyTreeState = TypeVar("PyTreeState")
TimeStepFn = Callable[[PyTreeState], PyTreeState]


class ExplicitNavierStokesODE_Penalty:
  """Spatially discretized version of Navier-Stokes.

  The equation is given by:

    ∂u/∂t = explicit_terms(u)
    0 = incompressibility_constraint(u)
  """

  def __init__(self, explicit_terms, pressure_projection,update_BC,Reserve_BC):
    self.explicit_terms = explicit_terms
    self.pressure_projection = pressure_projection
    self.update_BC = update_BC
    self.Reserve_BC = Reserve_BC


  def explicit_terms(self, state):
    """Explicitly evaluate the ODE."""
    raise NotImplementedError

  def pressure_projection(self, state):
    """Enforce the incompressibility constraint."""
    raise NotImplementedError

  def update_BC(self, state):
    """Update Wall BC """
    raise NotImplementedError

  def Reserve_BC(self, state):
    """Revert spurious updates of Wall BC """
    raise NotImplementedError

class ExplicitNavierStokesODE_BCtime:
  """Spatially discretized version of Navier-Stokes.

  The equation is given by:

    ∂u/∂t = explicit_terms(u)
    0 = incompressibility_constraint(u)
  """

  def __init__(self, explicit_terms, pressure_projection,update_BC,Reserve_BC,IBM_force,Update_Position,Pressure_Grad,Calculate_Drag):
    self.explicit_terms = explicit_terms
    self.pressure_projection = pressure_projection
    self.update_BC = update_BC
    self.Reserve_BC = Reserve_BC
    self.IBM_force = IBM_force
    self.Update_Position = Update_Position
    self.Pressure_Grad = Pressure_Grad
    self.Calculate_Drag = Calculate_Drag

  def explicit_terms(self, state):
    """Explicitly evaluate the ODE."""
    raise NotImplementedError

  def pressure_projection(self, state):
    """Enforce the incompressibility constraint."""
    raise NotImplementedError

  def update_BC(self, state):
    """Update Wall BC """
    raise NotImplementedError

  def Reserve_BC(self, state):
    """Revert spurious updates of Wall BC """
    raise NotImplementedError
  def IBM_force(self, state):
    """Revert spurious updates of Wall BC """
    raise NotImplementedError

  def Update_Position(self, state):
    """Revert spurious updates of Wall BC """
    raise NotImplementedError

  def Pressure_Grad(self, state):
    """Revert spurious updates of Wall BC """
    raise NotImplementedError
    
  def Calculate_Drag(self, state):
    """Revert spurious updates of Wall BC """
    raise NotImplementedError

@dataclasses.dataclass
class ButcherTableau_updated:
  a: Sequence[Sequence[float]]
  b: Sequence[float]
  c: Sequence[float]
  # TODO(shoyer): add c, when we support time-dependent equations.

  def __post_init__(self):
    if len(self.a) + 1 != len(self.b):
      raise ValueError("inconsistent Butcher tableau")
      
      
def navier_stokes_rk_updated(
    tableau: ButcherTableau_updated,
    equation: ExplicitNavierStokesODE_BCtime,
    time_step: float,
) -> TimeStepFn:
  """Create a forward Runge-Kutta time-stepper for incompressible Navier-Stokes.

  This function implements the reference method (equations 16-21), rather than
  the fast projection method, from:
    "Fast-Projection Methods for the Incompressible Navier–Stokes Equations"
    Fluids 2020, 5, 222; doi:10.3390/fluids5040222

  Args:
    tableau: Butcher tableau.
    equation: equation to use.
    time_step: overall time-step size.

  Returns:
    Function that advances one time-step forward.
  """
  # pylint: disable=invalid-name
  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  P = tree_math.unwrap(equation.pressure_projection)
  M = tree_math.unwrap(equation.update_BC)
  R = tree_math.unwrap(equation.Reserve_BC)
  IBM = tree_math.unwrap(equation.IBM_force)
  Update_Pos = tree_math.unwrap(equation.Update_Position) 
  Grad_Pressure = tree_math.unwrap(equation.Pressure_Grad) 
  Drag_Calculation = tree_math.unwrap(equation.Calculate_Drag)  

  a = tableau.a
  b = tableau.b
  num_steps = len(b)
  
  @tree_math.wrap
  def step_fn(u0):
    #print('vector',u0)
    #new_time = 0#u0[0].bc.time_stamp + dt
    u = [None] * num_steps
    k = [None] * num_steps

    def convert_to_velocity_vecot(u0):
        u = u0.tree
        return tree_math.Vector(tuple(u[i].array for i in range(len(u))))
        
    def convert_to_velocity_tree(m,bcs):
        return tree_math.Vector(tuple(grids.GridVariable(v,bc) for v,bc in zip(m.tree,bcs)))
    
    def convert_all_variabl_to_velocity_vecot(u0):
        u = u0.tree.velocity
        #return tree_math.Vector(tuple(grids.GridVariable(v.array,v.bc) for v in u))
        return  tree_math.Vector(u)
    def covert_veloicty_to_All_variable_vecot(particles,m,pressure,Drag,Step_count,MD_var):
        u = m.tree
        #return tree_math.Vector(particle_class.All_Variables(particles, tuple(grids.GridVariable(v.array,v.bc) for v in u),pressure))
        return tree_math.Vector(particle_class.All_Variables(particles,u,pressure,Drag,Step_count,MD_var))
    
    def velocity_bc(u0):
        u = u0.tree.velocity
        return tuple(u[i].bc for i in range(len(u)))
    
    def the_particles(u0):
        return u0.tree.particles
    def the_pressure(u0):
        return u0.tree.pressure
    def the_Drag(u0):
        return u0.tree.Drag
    
    
    particles = the_particles(u0)
    ubc = velocity_bc(u0)  
    pressure = the_pressure(u0)
    Drag = the_Drag(u0)
    Step_count = u0.tree.Step_count
    MD_var = u0.tree.MD_var
    
    
    u0 = convert_all_variabl_to_velocity_vecot(u0)

    
    u[0] = convert_to_velocity_vecot(u0)
    k[0] = convert_to_velocity_vecot(F(u0))
    dP = Grad_Pressure(tree_math.Vector(pressure))

   

    u0 = convert_to_velocity_vecot(u0)
    
    for i in range(1, num_steps):
        #u_star = u0[ww].array + sum(a[i-1][j]*k[j][ww].array for j in range(i) if a[i-1][j])
        
      u_star = u0 + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
    
      #u[i] = P(R(u_star))
      u[i] = convert_to_velocity_vecot(P(convert_to_velocity_tree(u_star,ubc)))  
      k[i] = convert_to_velocity_vecot(F(convert_to_velocity_tree(u[i],ubc)))

    #for ww in range(0,len(u0)):
    u_star = u0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])-dP
    
    Force = IBM(covert_veloicty_to_All_variable_vecot(particles,convert_to_velocity_tree(u_star,ubc),pressure,Drag,Step_count,MD_var))
    
    
    Drag_variable = Drag_Calculation(covert_veloicty_to_All_variable_vecot(particles,Force,pressure,Drag,Step_count,MD_var))
    Drag = the_Drag(Drag_variable)
    
    
    Force = convert_to_velocity_vecot(Force)
    
    #Tree_force =  convert_to_velocity_tree(Force,ubc)
    
    
    
    u_star_star = u_star + dt * Force
    
   # for i in range(0,2):
   #     Force = IBM(covert_veloicty_to_All_variable_vecot(particles,convert_to_velocity_tree(u_star_star,ubc),pressure,Drag))
   #     if i==1:
   #         Drag_variable = Drag_Calculation(covert_veloicty_to_All_variable_vecot(particles,Force,pressure,Drag))
   #         Drag = the_Drag(Drag_variable)
   #     Force = convert_to_velocity_vecot(Force)
   #     u_star_star = u_star+ dt * Force
    
    #u_final = P(R(u_star))
    #u_final = P(Force)
    
    
    
    
    u_final = convert_to_velocity_tree(u_star_star,ubc)
    
   
    u_final = covert_veloicty_to_All_variable_vecot(particles,u_final,pressure,Drag,Step_count,MD_var)
    
    
    
    u_final = P(u_final)
    #u_final = P(u_star_star)
    u_final = M(u_final)
    
    
    
        
    u_final = Update_Pos(u_final) # the time step counter is also updated
    
    return u_final

  return step_fn

def navier_stokes_rk_penalty(
    tableau: ButcherTableau_updated,
    equation: ExplicitNavierStokesODE_BCtime,
    time_step: float,
) -> TimeStepFn:
  """Create a forward Runge-Kutta time-stepper for incompressible Navier-Stokes.

  This function implements the reference method (equations 16-21), rather than
  the fast projection method, from:
    "Fast-Projection Methods for the Incompressible Navier–Stokes Equations"
    Fluids 2020, 5, 222; doi:10.3390/fluids5040222

  Args:
    tableau: Butcher tableau.
    equation: equation to use.
    time_step: overall time-step size.

  Returns:
    Function that advances one time-step forward.
  """
  # pylint: disable=invalid-name
  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  P = tree_math.unwrap(equation.pressure_projection)
  M = tree_math.unwrap(equation.update_BC)
  R = tree_math.unwrap(equation.Reserve_BC)

  a = tableau.a
  b = tableau.b
  num_steps = len(b)
  
  @tree_math.wrap
  def step_fn(u0):
    #print('vector',u0)
    #new_time = 0#u0[0].bc.time_stamp + dt
    u = [None] * num_steps
    k = [None] * num_steps

    def convert_to_velocity_vecot(u0):
        u = u0.tree
        return tree_math.Vector(tuple(u[i].array for i in range(len(u))))
        
    def convert_to_velocity_tree(m,bcs):
        return tree_math.Vector(tuple(grids.GridVariable(v,bc) for v,bc in zip(m.tree,bcs)))
    
    def convert_all_variabl_to_velocity_vecot(u0):
        u = u0.tree.velocity
        #return tree_math.Vector(tuple(grids.GridVariable(v.array,v.bc) for v in u))
        return  tree_math.Vector(u)
    def covert_veloicty_to_All_variable_vecot(particles,m,pressure,Drag,Step_count,MD_var):
        u = m.tree
        #return tree_math.Vector(particle_class.All_Variables(particles, tuple(grids.GridVariable(v.array,v.bc) for v in u),pressure))
        return tree_math.Vector(particle_class.All_Variables(particles,u,pressure,Drag,Step_count,MD_var))
    
    def velocity_bc(u0):
        u = u0.tree.velocity
        return tuple(u[i].bc for i in range(len(u)))
    
    def the_particles(u0):
        return u0.tree.particles
    def the_pressure(u0):
        return u0.tree.pressure
    def the_Drag(u0):
        return u0.tree.Drag

    particles = the_particles(u0)
    ubc = velocity_bc(u0)  
    pressure = the_pressure(u0)
    Drag = the_Drag(u0)
    Step_count = u0.tree.Step_count
    MD_var = u0.tree.MD_var

    
    u0 = convert_all_variabl_to_velocity_vecot(u0)
    
    u[0] = convert_to_velocity_vecot(u0)
    k[0] = convert_to_velocity_vecot(F(u0))
   
   
    
    u0 = convert_to_velocity_vecot(u0)
    
    for i in range(1, num_steps):
        #u_star = u0[ww].array + sum(a[i-1][j]*k[j][ww].array for j in range(i) if a[i-1][j])
        
      u_star = u0 + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
    
      #u[i] = P(R(u_star))
      u[i] = convert_to_velocity_vecot(P(convert_to_velocity_tree(u_star,ubc)))  
      k[i] = convert_to_velocity_vecot(F(convert_to_velocity_tree(u[i],ubc)))

    #for ww in range(0,len(u0)):
    u_star = u0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])

    u_final = convert_to_velocity_tree(u_star,ubc)
    
    u_final = covert_veloicty_to_All_variable_vecot(particles,u_final,pressure,Drag,Step_count,MD_var)
    u_final = P(u_final)
    #
    u_final = M(u_final)
    
        
    
    
    return u_final

  return step_fn

def forward_euler_penalty(
    equation: ExplicitNavierStokesODE_Penalty, time_step: float, 
) -> TimeStepFn:
  return jax.named_call(
      navier_stokes_rk_penalty(
          ButcherTableau_updated(a=[], b=[1], c=[0]),
          equation,
          time_step),
      name="forward_euler",
  )

def forward_euler_updated(
    equation: ExplicitNavierStokesODE_BCtime, time_step: float, 
) -> TimeStepFn:
  return jax.named_call(
      navier_stokes_rk_updated(
          ButcherTableau_updated(a=[], b=[1], c=[0]),
          equation,
          time_step),
      name="forward_euler",
  )


