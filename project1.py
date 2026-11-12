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

        delta_plus[end].append(start)
        delta_minus[start].append(end)

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

def heuristic_cut_first_intersection(routes,k):
    '''
        Brief: The naivest heuristic, remove a route if it intersects any other set and repeat till no intersections
    '''
    feasible_routes = []
    for i in range(k):
        if routes[i] is None:
            feasible_routes.append(None)
        else:
            for j in range(i+1,k):
                if routes[j] is None:
                    if_cut = 0
                    pass
                else:
                    if_cut = len(list(set(routes[i]).intersection(routes[j])))
                    if if_cut > 0:
                        break
            if if_cut > 0:
                feasible_routes.append(None)
            else:
                feasible_routes.append(routes[i])
    return feasible_routes

def heuristic_cut_largest_intersection(routes,k):
    '''
        Brief: Less naive heuristic, find all intersections and remove the route with the most.

        Note: The optimal problem of finding the maximal independent subset is NP-hard so this is only an approximate best subset to remove
    '''
    intersections_where  = np.zeros((k,k))
    intersections_number = np.zeros(k)

    # Which routes do we have?
    index_list = []
    for l in range(k):
        if routes[l] is not None:
            index_list.append(l)

    #For all routes we have find which intersects which?
    for i in index_list:
        for j in index_list: 
            if i < j:
                intersections_ij = len(list(set(routes[i]).intersection(routes[j])))

                if intersections_ij > 0:
                    intersections_where[i,j] += 1
                    intersections_where[j,i] += 1

                    intersections_number[i] += 1
                    intersections_number[j] += 1

    # While we have intersections, remove the one with the most itnersections
    max_intersections_index = np.argmax(intersections_number)
    while intersections_number[max_intersections_index] > 0:

        for i in index_list:
            if intersections_where[max_intersections_index, i] > 0:
                intersections_number[i] -= 1

                intersections_where[max_intersections_index, i] = 0
                intersections_where[i, max_intersections_index] = 0
        
        intersections_number[max_intersections_index] = 0
        index_list.remove(max_intersections_index)
        
        max_intersections_index = np.argmax(intersections_number)

    #Create the set of feasible routes from the indexes left.
    feasible_routes = []
    for i in range(k):
        if i in index_list:
            feasible_routes.append(routes[i])
        else:
            feasible_routes.append(None)
        
    return feasible_routes


def update_u(u,routes,width,height,theta,feasible_primal_solution):
    '''
        Brief: Update u one step using calculate_subgradient() based on Polyak step length with a feasible primal solution
    
    '''

    #Find seach direction
    search_direction = calculate_subgradient(routes,width,height)

    # Perform projection, if u_i = 0 project search direction onto the positive real line
    tol = 1e-5
    for x in range(width):
        for y in range(height):
            for z in range(2):
                if ((u[(x,y,z)] < tol) and (search_direction[(x,y,z)] > 0)):
                    search_direction[(x,y,z)] =  0

    #Polyak step length, using 
    lower_bound_h = feasible_primal_solution
    hu_t = h(u, routes)
    alpha_t = theta * (hu_t - lower_bound_h)/np.sum(search_direction*search_direction)      # Denominator se 5.2 b

    #Half step and then positive projection again
    u_halfstep =  u - alpha_t*search_direction
    u_updated = np.maximum(u_halfstep, 0)

    return u_updated, hu_t
    

def subgradient_algorithm(instance_name, max_iter=1000):
    '''
    Brief: full subgradient algorithm

    '''

    # Load the PCB we wish to calculate from
    PCB = load_instance("instances/"+instance_name)
    
    #Define basic variables and copydown attributes for cleaner notation
    width = PCB.width
    height = PCB.height
    contact_pairs = PCB.contact_pairs
    k = contact_pairs.shape[0]

    #delta_plus, delta_minus = calc_delta(PCB)

    # Initial u and theta values
    u = np.ones((width, height, 2))/100
    theta = 1.9

    # Save best feasible 
    best_feasible_primal_solution = 0
    best_feasible_routes   = []
    best_infeasible_routes = []

    # Main loop over some max iterations
    for t in range(max_iter):
        #Find routes using given mathod
        routes_t  = cheapest_grid_routes(width, height, u, contact_pairs)

        # Using heuristic to find feasible route with corresponding primal solution (number of connected contact pairs)
        feasible_routes_t = heuristic_cut_largest_intersection(routes_t,k)
        feasible_primal_solution_t = len([route for route in feasible_routes_t if route is not None])

        #If we get a better primal solution, update the current best set of solutions
        if feasible_primal_solution_t > best_feasible_primal_solution:
            best_feasible_primal_solution = feasible_primal_solution_t
            best_feasible_routes          = feasible_routes_t
            best_infeasible_routes        = routes_t

        #Update u
        u, hu_t = update_u(u,routes_t,width,height,theta,best_feasible_primal_solution)
            
        #Print progress and decrese theta
        if (t + 1) % 10 == 0:
            theta *= 0.95
            print(
                f"Iteration {t}: "
                f"h(u) = {hu_t:.6f}, "
                f"hbar = {best_feasible_primal_solution:.6f}"
            )

        # Duality gap is less than 1, the best feasible primal has been achieved (since it is an integer)
        if hu_t - best_feasible_primal_solution < 1:
            break

        
        
    plot_grid_routes(width,height,best_feasible_routes,dual_values=u)
    plt.show()
    plot_grid_routes(width,height,best_infeasible_routes,dual_values=u)
    plt.show()
    return 0

iters = 2000
#subgradient_algorithm("VLSI_8_6_7.json",iters)
subgradient_algorithm("VLSI_30_10_15.json",iters)


'''
Remains: 
    Polyak step length setting lower_bound_h approprietly
    Should return best 'bound_for_primal_t' not last bound
'''
