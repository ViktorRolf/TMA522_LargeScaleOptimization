from provided_functions import RoutingInstance
from provided_functions import load_instance
from provided_functions import cheapest_grid_routes
import numpy as np

def h(u):
    return 0

def calc_delta(PCB):
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

            delta_plus[(x,y,0)] = delta_plus_iter
            delta_minus[(x,y,0)] = delta_minus_iter

    # append contact pairs
    for contact_pair in PCB.contact_pairs:
        #
        end   = contact_pair[1]
        end = (end[0],end[1],end[2])
        start = contact_pair[0]
        start = (start[0],start[1],start[2])

        delta_plus[end]    = delta_plus[end].append(start)
        delta_minus[start] = delta_plus[start].append(end)

    return delta_plus, delta_minus

def subgradient_algo(instance_name, max_iter=1000):
    PCB = load_instance("instances/"+instance_name)
    width = PCB.width
    height = PCB.height
    contact_pairs = PCB.contact_pairs
    k = contact_pairs.shape[0]

    delta_plus, delta_minus = calc_delta(PCB)

    x = [np.zeros((width, height, 2)) for _ in range(k)]

    u = np.zeros((width, height, 2))
    theta = 1.9
    for i in range(max_iter):
        routes = cheapest_grid_routes(width, height, u, contact_pairs)

subgradient_algo("VLSI_8_6_7.json")