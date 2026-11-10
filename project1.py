from provided_functions import RoutingInstance
from provided_functions import load_instance
from provided_functions import cheapest_grid_routes
from provided_functions import plot_grid_routes
import numpy as np
import matplotlib.pyplot as plt


def calc_delta(PCB):
    '''
    Brief: Calculates delta plus and delta minus for all nodes.

    Parameters: 
        RoutingInstance PCB

    Returns:
        dictionary(tuples | list[tuples]) delta_plus, delta_minus

    '''
    delta_plus  = dict()
    delta_minus = dict()

    # z = 0
    for x in range(PCB.width):
        for y in range(PCB.height):
            delta_plus_iter = [(x,y,1)]
            delta_minus_iter = [(x,y,1)]

            if x < PCB.width-1:
                delta_plus_iter.append((x+1,y,0))
                delta_minus_iter.append((x+1,y,0))
            if x > 0:
                delta_plus_iter.append((x-1,y,0))
                delta_minus_iter.append((x-1,y,0))

            delta_plus[(x,y,0)] = delta_plus_iter
            delta_minus[(x,y,0)] = delta_minus_iter
    # z = 1
    for x in range(PCB.width):
        for y in range(PCB.height):
            delta_plus_iter = [(x,y,0)]
            delta_minus_iter = [(x,y,0)]

            if y < PCB.height-1:
                delta_plus_iter.append((x,y+1,1))
                delta_minus_iter.append((x,y+1,1))
            if y > 0:
                delta_plus_iter.append((x,y-1,1))
                delta_minus_iter.append((x,y-1,1))

            delta_plus[(x,y,1)] = delta_plus_iter
            delta_minus[(x,y,1)] = delta_minus_iter

    # append contact pairs
    for contact_pair in PCB.contact_pairs:
        #
        end   = contact_pair[1]
        end = (end[0],end[1],end[2])
        start = contact_pair[0]
        start = (start[0],start[1],start[2])

        delta_plus[end].append(start) # fix
        delta_plus[start].append(end)

    return delta_plus, delta_minus


def h(u, routes):
    '''
        Brief: Calculate h(u) given the solution to the Lagrangean subproblems for each l=0..k
    '''
    
    subproblem_sum = 0
    for route_k in routes: # For each k the only x's set to 1 are the ones in the route, all other are zero

        if route_k is None: #If no route for this l exist this term is zero
            continue
        
        subproblem_sum +=  1 # Q? Is it always route xor no rote
        for vertex_i in route_k:
            subproblem_sum -= u[vertex_i]*1

        #print(route_k)  
    #print(np.sum(u) + subproblem_sum)

    return np.sum(u) + subproblem_sum


def calculate_subgradient(routes,width,height):
    '''
    Brief: Calculate subgradient at step t
    '''
    search_direction = np.ones((width, height, 2))

    for route_k in routes: # For each k the only x's set to 1 are the ones in the route, all other are zero
    
            if route_k is None: #If no route for this l exist this term is zero
                continue
    
            for vertex_i in route_k:
                search_direction[vertex_i] -= 1

    return search_direction


def update_u(u,routes,k,width,height,theta):
    search_direction = calculate_subgradient(routes,width,height)

    # Perform projection, if u_i = 0 project search direction onto the positive real line
    tol = 1e-5
    for x in range(width):
        for y in range(height):
            for z in range(2):
                if ((u[(x,y,z)] < tol) and (search_direction[(x,y,z)] > 0)):
                    search_direction[(x,y,z)] =  0

    #Polyak step length
    lower_bound_h = 6.2 #Q? Bad lower boudn but should always be the case
    hu_t = h(u, routes)
    alpha_t = theta * (hu_t - lower_bound_h)/np.sum(search_direction*search_direction)      # Denominator se 5.2 b

    #Half step and then positive projection again
    u_updated =  u - alpha_t*search_direction
    for x in range(width):
        for y in range(height):
            for z in range(2):
                if u_updated [(x,y,z)] <= 0:
                    u_updated [(x,y,z)] =  0

    return u_updated, hu_t
    

def subgradient_algorithm(instance_name, max_iter=1000):
    '''
    Brief: full subgradient algorithm

    '''
    PCB = load_instance("instances/"+instance_name)
    width = PCB.width
    height = PCB.height
    contact_pairs = PCB.contact_pairs
    k = contact_pairs.shape[0]

    delta_plus, delta_minus = calc_delta(PCB)

    #x = [np.zeros((width, height, 2)) for _ in range(k)]


    u = np.ones((width, height, 2))/100
    theta = 1.9
    for i in range(max_iter):
        
        routes  = cheapest_grid_routes(width, height, u, contact_pairs)

        u, bound_for_primal_t = update_u(u,routes,k,width,height,theta,)


        #if bound_for_primal_t < k - 1e-9:
         #   break
        if (i + 1) % 10 == 0:
            theta *= 0.95
            print(
                f"Iteration {i}: "
                f"h(u) = {bound_for_primal_t:.6f}, "
                f"u_max = {np.max(u):.6f}"
            )
    plot_grid_routes(width,height,routes,dual_values=u)
    plt.show()
    return 0

iters = 1000
subgradient_algorithm("VLSI_8_6_7.json",iters)
#subgradient_algorithm("VLSI_35_12_18.json",iters)


'''
Remains: 
    Polyak step length setting lower_bound_h approprietly
    Should return best 'bound_for_primal_t' not last bound
'''
