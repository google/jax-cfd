import jax
import jax.numpy as jnp


def displacement(parameters,t):
    A0,f = list(*parameters)
    return jnp.array([A0/2*jnp.cos(2*jnp.pi*f*t),0.])
    
    
def rotation(parameters,t):
    alpha0,beta,f,phi = list(*parameters)
    return alpha0 + beta*jnp.sin(2*jnp.pi*f*t+phi)

def Displacement_Foil_Fourier_Dotted_Mutliple(parameters,t):

    alpha0=jnp.array(list(list(zip(*parameters))[0]))
    f =jnp.array(list(list(zip(*parameters))[1]))
    phi = jnp.array(list(list(zip(*parameters))[2]))
    alpha = jnp.array(list(list(zip(*parameters))[3]))
    beta = jnp.array(list(list(zip(*parameters))[4]))
    p = jnp.array(list(list(zip(*parameters))[5]))
    
    size_parameters = alpha.shape[1]
    N_particles =len(alpha)

    ## Create an array of the size (nparticles, nparameters)
    frequencies = jnp.array([jnp.arange(1,size_parameters+1)]*N_particles)

    ## multiply and add arrays

    inside_function =jnp.add(2*jnp.pi*t*frequencies*f.reshape(N_particles,1),phi.reshape(N_particles,1))
  
    alpha_1 = (alpha*jnp.sin(inside_function)).sum(axis=1)
    
    alpha_1 += p*(beta*jnp.cos(inside_function)).sum(axis=1)
    
    
    return jnp.array([-alpha0*t,alpha_1])

    
def rotation_Foil_Fourier_Dotted_Mutliple(parameters,t):
    #alpha0,f,phi,alpha,beta,p = parameters
    alpha0=jnp.array(list(list(zip(*parameters))[0]))
    f =jnp.array(list(list(zip(*parameters))[1]))
    phi = jnp.array(list(list(zip(*parameters))[2]))
    alpha = jnp.array(list(list(zip(*parameters))[3]))
    beta = jnp.array(list(list(zip(*parameters))[4]))
    p = jnp.array(list(list(zip(*parameters))[5]))
    
    size_parameters = alpha.shape[1]
    N_particles =len(alpha)

    ## Create an array of the size (nparticles, nparameters)
    frequencies = jnp.array([jnp.arange(1,size_parameters+1)]*N_particles)

    ## multiply and add arrays

    inside_function =jnp.add(2*jnp.pi*t*frequencies*f.reshape(N_particles,1),phi.reshape(N_particles,1))
  
    alpha_1 = (alpha*jnp.sin(inside_function)).sum(axis=1)
    
    alpha_1 += p*(beta*jnp.cos(inside_function)).sum(axis=1)
    
    #if N_particles>1:
    return alpha0*t + alpha_1

def rotation_Foil_Fourier_Dotted_Mutliple_NORMALIZED(parameters,t):
    #alpha0,f,phi,alpha,beta,p = parameters
    alpha0=jnp.array(list(list(zip(*parameters))[0]))
    f =jnp.array(list(list(zip(*parameters))[1]))
    phi = jnp.array(list(list(zip(*parameters))[2]))
    alpha = jnp.array(list(list(zip(*parameters))[3]))
    beta = jnp.array(list(list(zip(*parameters))[4]))
    theta_av = jnp.array(list(list(zip(*parameters))[5]))
    p = jnp.array(list(list(zip(*parameters))[6]))
    
    size_parameters = alpha.shape[1]
    N_particles =len(alpha)

    ## Create an array of the size (nparticles, nparameters)
    frequencies = jnp.array([jnp.arange(1,size_parameters+1)]*N_particles)

    ## multiply and add arrays

    inside_function =jnp.add(2*jnp.pi*t*frequencies*f.reshape(N_particles,1),phi.reshape(N_particles,1))
  
    alpha_1 = (alpha*jnp.sin(inside_function)).sum(axis=1)
    
    alpha_1 += p*(beta*jnp.cos(inside_function)).sum(axis=1)
    
    inside_function2 =jnp.add(2*jnp.pi*frequencies,phi.reshape(N_particles,1))
  
    alpha_2 = (alpha*jnp.sin(inside_function2)).sum(axis=1)
    
    alpha_2 += p*(beta*jnp.cos(inside_function2)).sum(axis=1)
    
    inside_function3 =jnp.add(2*jnp.pi*frequencies*0.0,phi.reshape(N_particles,1))
  
    alpha_3 = (alpha*jnp.sin(inside_function3)).sum(axis=1)
    
    alpha_3 += p*(beta*jnp.cos(inside_function3)).sum(axis=1)
    
  
    #if N_particles>1:
    return theta_av*(alpha0*t + alpha_1)/(alpha0 + alpha_2-alpha_3)
    #return (alpha0 + alpha_2)
    #else:
    #    return (alpha0*t + alpha_1)[0]
