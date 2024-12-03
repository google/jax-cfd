from jax_ib.base import particle_class as pc
import jax
import jax.numpy as jnp


def Update_particle_position_Multiple_and_MD_Step(step_fn,all_variables,dt):
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t =velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param    
    New_eq = lambda t:Displacement_EQ(displacement_param,t)
    dx_dt = jax.jacrev(New_eq)

    
    #MD_var = step_fn(MD_var)
    
    U0 =dx_dt(current_t)
    #print(U0)
    Newparticle_center = jnp.array([particle_centers[:,0]+dt*U0[0],particle_centers[:,1]+dt*U0[1]]).T
    #print(Newparticle_center)
    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    rotation_param = particles.rotation_param
    
    MD_var = step_fn(all_variables)

    New_particles = pc.particle(Newparticle_center,param_geometry,displacement_param,rotation_param,mygrids,shape_fn,Displacement_EQ,particles.Rotation_EQ)
    
    return pc.All_Variables(New_particles,velocity,pressure,Drag,Step_count,MD_var)
    
    
def Update_particle_position_Multiple(all_variables,dt):
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t =velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param
    New_eq = lambda t:Displacement_EQ(displacement_param,t)
    dx_dt = jax.jacrev(New_eq)

    
    
    U0 =dx_dt(current_t)
    #print(U0)
    Newparticle_center = jnp.array([particle_centers[:,0]+dt*U0[0],particle_centers[:,1]+dt*U0[1]]).T
    #print(Newparticle_center)
    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    rotation_param = particles.rotation_param
    
    MD_var = all_variables.MD_var

    New_particles = pc.particle(Newparticle_center,param_geometry,displacement_param,rotation_param,mygrids,shape_fn,Displacement_EQ,particles.Rotation_EQ)
    
    return pc.All_Variables(New_particles,velocity,pressure,Drag,Step_count,MD_var)
