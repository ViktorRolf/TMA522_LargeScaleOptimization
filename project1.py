from provided_functions import RoutingInstance
from provided_functions import load_instance
from provided_functions import cheapest_grid_routes
from provided_functions import plot_grid_routes
import numpy as np
import matplotlib.pyplot as plt
import time


def calc_delta(PCB: RoutingInstance):
    '''
    Brief: Calculates delta plus and delta minus for all nodes.

    Parameters: 
        RoutingInstance PCB

    Returns:
        dictionary(tuples -> list[tuples]) delta_plus, delta_minus : Essentially 'lookup' table for which nodes are reachable from current node.

    '''
    #NOTE: For this function, some things are superflous and we could get by with less, we do this for rigor as it doesn't impact performence a lot anyway

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

        #delta_plus[end].append(start)     #REMOVED, while not correct in the mathematical sense we don't want graphs between contact pairs to find ways 'across' this imagined line.
        #delta_minus[start].append(end)

    return delta_plus, delta_minus


def heuristic_cut_first_intersection(routes,k):
    '''
    Brief: The naivest heuristic, remove a route if it intersects any other set and repeat till no intersections

    Parameters: 
        list[list[tuples] | None] routes : List of routes found in the subproblem. 
        int k : Number of contact pairs (length of routes)
        
    Returns:
        list[list[tuples] | None] feasible_routes : A list of feasable routes, produced from routes         
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
        
    Parameters: 
        list[list[tuples] | None] routes : List of routes found in the subproblem. 
        int k : Number of contact pairs (length of routes)
        
    Returns:
        list[list[tuples] | None] feasible_routes : A list of feasable routes, produced from routes     
        
    '''
    #NOTE: The optimal problem of finding the maximal independent subset is NP-hard so this is only an approximate best subset to remove
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

    # While we have intersections, remove the one with the most intersections
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


def heuristic_cut_and_add(routes,k, contact_pairs, delta_plus):
    '''
        Brief: Best heuristic we found, remove intersecting routes using 'heuristic_cut_largest_intersection' then using graph search add in more routes on unused contact pairs.
        
        Parameters: 
            list[list[tuples] | None] routes : List of routes found in the subproblem. 
            int k : Number of contact pairs (length of routes)
            dictionary(tuples -> list[tuples]) delta_plus : Lookup table to check where we have connections between nodes
        
        Returns:
            list[list[tuples] | None] feasible_routes : A list of feasable routes, produced from routes     
        
    '''
    # Make the solutio nfeasable through removing routes
    routes = heuristic_cut_largest_intersection(routes,k)

    # Find out where we should search for new routes (for which contact pairs) and gather all used nodes in a set.
    used_nodes = set()
    contact_pairs_without_route = []
    for i in range(k):
        if routes[i] is None:
            contact_pairs_without_route.append(i)
        else:
            used_nodes = used_nodes | set(routes[i])


    # Subfunction to recusivly (via DFS) find the shortest route for all a contact pair.
    def graph_search_helper(current_route):
        #find adjecent nodes and remove all that have been seen/used
        nodes_to_search = set(delta_plus[current_route[-1]]) - visited_nodes_this_contact_pair
    
        if not nodes_to_search:
            # Set empty, nothing further to search
            return None
        elif end_node in nodes_to_search:
            # Found a route
            return current_route+[end_node]
        else:
            # No stopping condition found, continue searching (depth first)
            best_route_length = 100000
            best_route = None
            for node in nodes_to_search:
                visited_nodes_this_contact_pair.add(node)
                possible_route = graph_search_helper(current_route+[node])

                if possible_route is not None and len(possible_route) < best_route_length:
                    best_route = possible_route

            # return the best route (or None fi no route was found)
            return best_route
    

    # For each contact pair (that does not have a path), start the recursive loop with only the initial node
    for i in contact_pairs_without_route:
        visited_nodes_this_contact_pair = used_nodes.copy()
        start_node = tuple(contact_pairs[i][0])
        end_node   = tuple(contact_pairs[i][1])

        # For each unused contact pair find a route (if one exists)
        visited_nodes_this_contact_pair.add(start_node)
        possible_route = graph_search_helper([start_node])
        
        # If a route is found add it, then repeat.
        if possible_route is not None:
            routes[i] = possible_route
            used_nodes = used_nodes | set(routes[i])

    return routes


def h(u, routes):
    '''
    Brief: Calculate h(u) given the solution to the Lagrangean subproblems for each l=0..k
        
    Parameters: 
        ArrayLike u : The specific set of dual-vartiables we calculate from
        list[list[tuples] | None] routes : List of routes found in the subproblem, given the above u 

    Returns:
        The value of h(u) given the subproblem solution.

    '''
    
    subproblem_sum = 0
    for route_k in routes: # For each k the only x's set to 1 are the ones in the route, all other are zero

        if route_k is None: #If no route for this l exist this term is zero
            continue
        
        subproblem_sum +=  1
        for vertex_i in route_k:
            subproblem_sum -= u[vertex_i]*1


    return np.sum(u) + subproblem_sum


def calculate_subgradient(routes,width,height):
    '''
    Brief: Calculate subgradient at step t
    
    Parameters: 
        list[list[tuples] | None] routes : List of routes found in the subproblem, given the above u 
        int width, height : size of the grid

    Returns:
        ArrayLike search_direction: A subdirection
    
    '''
    search_direction = np.ones((width, height, 2))

    for route_k in routes: # For each k the only x's set to 1 are the ones in the route, all other are zero
    
            if route_k is None: #If no route for this l exist this term is zero
                continue
    
            for vertex_i in route_k:
                search_direction[vertex_i] -= 1

    return search_direction


def update_u(u,routes,width,height,theta,feasible_primal_solution):
    '''
    Brief: Update u one step using calculate_subgradient() based on Polyak step length with a feasible primal solution
    
    Parameters: 
        ArrayLike u : The specific set of dual-vartiables we calculate from
        list[list[tuples] | None] routes : List of routes found in the subproblem, given the above u 
        int width, height : size of the grid
        float theta : Decresing parameter of 'steplength' in Polyak step-length formula
        int feasible_primal_solution : A lower bound on h(u) given by a feasable primal solution

    Returns:
        ArrayLike u_updated: The new u value
        hu_t : The value of h(u)
    
    '''

    #Find search direction
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
    

def subgradient_algorithm(instance_name, max_iter=1000, save_all_hdata = False, timing_iterations=False):
    '''
    Brief: full subgradient algorithm

    Parameters: 
        string instance_name : What instance do we run it on, give only name not path. The below structure is assumed.
        int max_iter : How many iterations before automatic termination, default = 1000.
        bool save_all_hdata : If we keep all data for hbar and h(u), for plot construction then return this data. 
        bool timing_iterations : Time iterations and print an average (per iter) compute time.

    Returns:
        algorithm_results : Structure documented in code below
    '''
    # Storage for results
    algorithm_results = ()
    if save_all_hdata:
        hdata=([],[],[])
    else:
        hdata = None

    # Load the PCB we wish to calculate from
    PCB = load_instance("instances/"+instance_name)
    
    #Define basic variables and copydown attributes for cleaner notation
    width = PCB.width
    height = PCB.height
    contact_pairs = PCB.contact_pairs
    k = contact_pairs.shape[0]

    delta_plus, delta_minus = calc_delta(PCB)

    # Initial u and theta values
    u = np.ones((width, height, 2))/100
    theta = 1.9

    # Save best feasible 
    best_feasible_primal_solution = 0
    best_feasible_routes   = []
    best_infeasible_routes = []

    #if timing, set start time
    if timing_iterations==True:
        start_time = time.perf_counter()


    # Main loop over some max iterations
    for t in range(max_iter):
        #Find routes using given mathod
        routes_t  = cheapest_grid_routes(width, height, u, contact_pairs)

        # Using heuristic to find feasible route with corresponding primal solution (number of connected contact pairs)
        feasible_routes_t = heuristic_cut_and_add(routes_t, k, contact_pairs, delta_plus) #feasible_routes_t = heuristic_cut_largest_intersection(routes_t,k)
        feasible_primal_solution_t = len([route for route in feasible_routes_t if route is not None]) 

        #If we get a better primal solution, update the current best set of solutions
        if feasible_primal_solution_t > best_feasible_primal_solution:
            best_feasible_primal_solution = feasible_primal_solution_t
            best_feasible_routes          = feasible_routes_t
            best_infeasible_routes        = routes_t

        #Update u, note hu_t  is h(u) of the previous u
        u_new, hu_t = update_u(u,routes_t,width,height,theta,best_feasible_primal_solution)
        
        #decrese theta
        if (t + 1) % 10 == 0:
            theta *= 0.95

        #Print progress
        if t % 100 == 0:
            print(instance_name+f"Iteration {t} : h(u) = {hu_t:.6f} : hbar = {best_feasible_primal_solution:.6f}")

        #saving hdata
        if save_all_hdata:
            hdata[0].append(best_feasible_primal_solution) #hbar
            hdata[1].append(hu_t)                          #hu
            hdata[2].append(t)                             #iteration

        # Duality gap is less than 1, the best feasible primal has been achieved (since it is an integer)
        duality_gap = hu_t - best_feasible_primal_solution
        if duality_gap < 1-1e-6:
            algorithm_results = (t,                                        # Number of iterations
                                 hu_t,                                     # Value of dual function
                                 int(best_feasible_primal_solution),       # The best feasable value found by heuristic, that is hbar
                                 duality_gap,                              # Duality gap, difference of above
                                 (width,height,best_feasible_routes, u),   # Data needed to use ' plot_grid_routes(width,height,best_feasible_routes,dual_values=u) '
                                 hdata)
            break

        #Update u for clarity and then repeat loop
        u = u_new

        #Failed to converge
        if t == max_iter-1:
            algorithm_results = None
        
    '''   
    plot_grid_routes(width,height,best_feasible_routes,dual_values=u)
    plt.show()
    plot_grid_routes(width,height,best_infeasible_routes,dual_values=u)
    plt.show()
    '''

    #if timing, set end time
    if timing_iterations==True:
        end_time = time.perf_counter()
        per_iteration_time = round((end_time - start_time) * 1e3/t,5)
        print("Execution time (per iteration):",per_iteration_time  , "milliseconds")
        print("")

    return algorithm_results

# THE TESTS WE DO:
iters = 10000
instance_names = ["VLSI_8_6_7.json",
                  "VLSI_30_10_15.json",
                  "VLSI_30_11_15.json",
                  "VLSI_35_12_18.json",
                  "VLSI_35_13_17.json",
                  "VLSI_35_14_16.json"]

printed_strings = []
plotting_data   = []
h_data          = []

for instance_name in instance_names:
    algorithm_results = subgradient_algorithm(instance_name,iters,True)
    iterations_inst    = algorithm_results[0]
    hu_inst            = algorithm_results[1]
    hbar_inst          = algorithm_results[2]
    duality_gap_inst   = algorithm_results[3]

    plotting_data.append(algorithm_results[4])
    h_data.append(algorithm_results[5])

    printed_strings.append(instance_name+f" completed in {iterations_inst} iterations, h(u)={hu_inst}, hbar={hbar_inst} giving duality gap {duality_gap_inst}")

# Print the results
print("")
for string_t in printed_strings:
    print(string_t)

#plot convergence
fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharex=False, sharey=True)
axs = axs.flatten()

colors = ['red', 'blue', 'orange', 'green', 'purple', 'yellow']

for k in range(len(instance_names)):
    ax = axs[k]

    hbar       = h_data[k][0]
    hu         = h_data[k][1]
    iterations = h_data[k][2]

    ax.plot(iterations, hu, color=colors[k], label=r'$h(u)$')
    ax.plot(iterations, hbar, '--', color=colors[k], label=r'$\bar{h}$')
    ax.axvline(x=iterations[-1], color=colors[k],
               linestyle=':', linewidth=2, alpha=0.8)

    ax.set_title(instance_names[k])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("h-values")
    ax.legend()

fig.suptitle(
    "Convergence of all instances\n"
    r"$h(u)$ as solid line and $\bar{h}$ as dashed line",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()



# plot all wiring
for data in plotting_data:
    plot_grid_routes(data[0],data[1],data[2],dual_values=data[3])
    plt.show()


#print iterations times
print("")
for instance_name in instance_names:
    print(instance_name+" (timing): ")
    subgradient_algorithm(instance_name,iters,save_all_hdata=False,timing_iterations=True)

