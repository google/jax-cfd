import jax.numpy as jnp
import jax
from jax_ib.base import grids

def integrate_trapz(integrand,dx,dy):
    return jnp.trapz(jnp.trapz(integrand,dx=dx),dx=dy)


def Integrate_Field_Fluid_Domain(field):
    
    
    grid = field.grid
   # offset = field.offset
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
   # X,Y =grid.mesh(offset)
    
    return integrate_trapz(field.data,dxEUL,dyEUL)

def IBM_force_GENERAL(field,Xi,particle_center,geom_param,Grid_p,shape_fn,discrete_fn,surface_fn,dx_dt,domega_dt,rotation,dt):
    
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    current_t = field.bc.time_stamp
    #current_t = 0.0
    xp0,yp0 = shape_fn(geom_param,Grid_p)
    #print('yp',yp0,'xp',xp0)
    #print('angle',current_t,rotation(current_t),particle_center)
    #print(yp0)
    xp = (xp0)*jnp.cos(rotation(current_t))-(yp0)*jnp.sin(rotation(current_t))+particle_center[0]
    yp = (xp0)*jnp.sin(rotation(current_t))+(yp0 )*jnp.cos(rotation(current_t))+particle_center[1]
    surface_coord =[(xp)/dxEUL-offset[0],(yp)/dyEUL-offset[1]]
    #print(rotation(current_t))
    velocity_at_surface = surface_fn(field,xp,yp)
    
    if Xi==0:
        position_r = -(yp-particle_center[1])
    elif Xi==1:
        position_r = (xp-particle_center[0])
    
    U0 = dx_dt(current_t)
    #print('U0',U0)
    Omega=domega_dt(current_t)    
    UP= U0[Xi] + Omega*position_r 
    #print(xp)
    #print('XI',Xi,UP,len(UP))
    force = (UP -velocity_at_surface)/dt
    
   # if Xi==0:
        #plt.plot(xp,force)
        #maxforce =  delta_approx_logistjax(xp[0],X,0.003,1)
   #     maxforce = discrete_fn(xp[3],X)
   #     plt.imshow(maxforce)
   #     print('Maxforce',jnp.max(maxforce))
    #    print(xp)
    x_i = jnp.roll(xp,-1)
    y_i = jnp.roll(yp,-1)
    dxL = x_i-xp
    dyL = y_i-yp
    dS = jnp.sqrt(dxL**2 + dyL**2)
    
    
    def calc_force(F,xp,yp,dxi,dyi,dss):
        return F*discrete_fn(jnp.sqrt((xp-X)**2 + (yp-Y)**2),0,dxEUL)*dss
        #return F*discrete_fn(xp-X,0,dxEUL)*discrete_fn(yp-Y,0,dyEUL)*dss
        #return F*discrete_fn(xp,X,dxEUL)*discrete_fn(yp,Y,dyEUL)*dss**2
    def foo(tree_arg):
        F,xp,yp,dxi,dyi,dss = tree_arg
        return calc_force(F,xp,yp,dxi,dyi,dss)
    
    def foo_pmap(tree_arg):
        #print(tree_arg)
        return jnp.sum(jax.vmap(foo,in_axes=1)(tree_arg),axis=0)
    divider=jax.device_count()
    n = len(xp)//divider
    mapped = []
    for i in range(divider):
       # print(i)
        mapped.append([force[i*n:(i+1)*n],xp[i*n:(i+1)*n],yp[i*n:(i+1)*n],dxL[i*n:(i+1)*n],dyL[i*n:(i+1)*n],dS[i*n:(i+1)*n]])
    #mapped = jnp.array([force,xp,yp])
    #remapped = mapped.reshape(())#jnp.array([[force[:n],xp[:n],yp[:n]],[force[n:],xp[n:],yp[n:]]])
    
    #return cfd.grids.GridArray(jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)),axis=0),offset,grid)
    return jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)),axis=0)

def IBM_Multiple_NEW(field, Xi, particles,discrete_fn,surface_fn,dt):
    Grid_p = particles.generate_grid()
    shape_fn = particles.shape
    Displacement_EQ = particles.Displacement_EQ
    Rotation_EQ = particles.Rotation_EQ
    Nparticles = len(particles.particle_center)
    particle_center = particles.particle_center
    geom_param = particles.geometry_param
    displacement_param = particles.displacement_param
    rotation_param = particles.rotation_param
    force = jnp.zeros_like(field.data)
    for i in range(Nparticles):
        Xc = lambda t:Displacement_EQ([displacement_param[i]],t)
        rotation = lambda t:Rotation_EQ([rotation_param[i]],t)
        dx_dt = jax.jacrev(Xc)
        domega_dt = jax.jacrev(rotation)
        force+= IBM_force_GENERAL(field,Xi,particle_center[i],geom_param[i],Grid_p,shape_fn,discrete_fn,surface_fn,dx_dt,domega_dt,rotation,dt)
    return grids.GridArray(force,field.offset,field.grid)


def calc_IBM_force_NEW_MULTIPLE(all_variables,discrete_fn,surface_fn,dt):
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0,1]
    ibm_forcing = lambda field,Xi:IBM_Multiple_NEW(field, Xi, particles,discrete_fn,surface_fn,dt)
    
    return tuple(grids.GridVariable(ibm_forcing(field,Xi),field.bc) for field,Xi in zip(velocity,axis))
