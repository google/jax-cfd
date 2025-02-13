import jax
import jax.numpy as jnp


def delta_approx_logistjax(x,x0,w):
    
    return 1/(w*jnp.sqrt(2*jnp.pi))*jnp.exp(-0.5*((x-x0)/w)**2)



def new_surf_fn(field,xp,yp,discrete_fn):
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dx = grid.step[0]
    dy = grid.step[1]
    
    def calc_force(xp,yp):
        return jnp.sum(field.data*discrete_fn(xp,X,dx)*discrete_fn(yp,Y,dy)*dx*dy)
    def foo(tree_arg):
        xp,yp = tree_arg
        return calc_force(xp,yp)
    
    def foo_pmap(tree_arg):
        #print(tree_arg)
        return jax.vmap(foo,in_axes=1)(tree_arg)
    
    divider=jax.device_count()
    n = len(xp)//divider
    mapped = []
    for i in range(divider):
       # print(i)
        mapped.append([xp[i*n:(i+1)*n],yp[i*n:(i+1)*n]])
        
    U_deltas = jax.pmap(foo_pmap)(jnp.array(mapped))
    
    return U_deltas.flatten()
