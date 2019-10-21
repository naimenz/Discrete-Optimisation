import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq as hq

# Initialise seed for reproduciblity
np.random.seed(2)

##############################################
############# SET-UP FUNCTIONS ###############
##############################################

"""
calculate the distance between two nodes
"""
def calc_dist(pos1, pos2):
    dist = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    return dist

"""
simple wrapper function to extract weight of an edge
"""
def get_weight(graph, edge, attr="weight"):
    return graph.edges[edge][attr]

"""
For a complete graph on N nodes and a list of N positions,
assign each node a position and then add the distances
to the edges between them all.
"""
def assign_positions_and_distances(graph, positions):

    # shorter aliases for graph objects
    nodes = graph.nodes
    edges = graph.edges

    # assign positions
    for node, position in zip(nodes, positions):
        nodes[node]["position"] = position

    # assign distances to the edges
    for edge in edges:
        # getting the nodes in the edge
        node1 = nodes[edge[0]]
        node2 = nodes[edge[1]]
        edges[edge]["weight"] = calc_dist(node1["position"], node2["position"])

"""
calculate the total length of the route (including looping back)
"""
def calc_route_length(graph, route):
    # number of stops in the route
    N = len(route)
    length = 0
    # sum the weights of all edges in the closed route
    # note that we wrap around at the end to close the loop
    for i in range(N):
        edge = (route[i], route[(i+1)%N])
        length += get_weight(graph, edge)

    return length

"""A function to take a set of edges that constitute a cycle and construct a list of the
cities visited in order
nb. will throw an error if the edges do not precisely constitute a cycle"""

def construct_route_from_edges(input_edges):
    N = len(input_edges)
    # making a copy of the edges so we can modify it
    edges = input_edges[:]
    # getting a starting edge to begin the route
    edge0 = edges.pop(0)
    route = [edge0[0], edge0[1]]
    # loop that will be broken when the loop is completed or 
    # if it takes too long and fails
    for i in range(N - 2):
        # note that there will only be one edge that now contains the city
        # because we remove the other
        previous_city = route[-1]
        next_edge = [e for e in edges if previous_city in e][0] 

        # remove the edge from the list
        edges.remove(next_edge)

        next_city = next_edge[(next_edge.index(previous_city) + 1) % 2]
        route.append(next_city)

    return route
    
"""
draw the cities using the routes to specify order
Nb. takes a list and draws them as subplots
"""
def draw_cities(graph, routes):
    N = len(routes)
    plt.figure(figsize=(20,5))
    for i in range(N):
        plt.subplot(1, N, i+1)
        
        route = routes[i][0]
        plt.title(routes[i][1])
        
        x, y = [],[]
        for stop in route:
            pos = graph.nodes[stop]["position"]
            x.append(pos[0])
            y.append(pos[1])

        # adding the first element to the end to close the loop
        x.append(x[0])
        y.append(y[0])

        # plotting all possible connections
        for node in graph.nodes:
            nodepos = graph.nodes[node]["position"]
            for neighbour in graph.neighbors(node):
                neighpos = graph.nodes[neighbour]["position"]
                plt.plot([nodepos[0], neighpos[0]], [nodepos[1], neighpos[1]], color="grey", alpha=0.5, linewidth=0.5)

        
        plt.plot(x,y, linewidth=3)
        plt.scatter(x,y, color="black", zorder=100, s=80)
    plt.show()

##############################################
############ SIMULATED ANNEALING #############
##############################################

"""
An operation to create a new route based on an old one.
Can no longer calculate dEs though so may be slower.
"""
def two_opt(li, i, j):
    return np.concatenate(( li[:i], list(reversed(li[i:j])), li[j:])).astype(int)

"""
calculate the change in length ("energy") that would arise from
swapping the ith city with the i+1th city
"""
def calc_dE(graph, route, i):
    dE = 0
    N = len(route)
    # first we calculate the current length contributed
    # by the two nodes in their current positions
    # note the two nodes will still be connected so no need to include them
    edge1 = (route[(i-1)%N], route[i])
    edge2 = (route[(i+1)%N], route[(i+2)%N])
    current = get_weight(graph, edge1) + get_weight(graph, edge2)

    # now calculate what would happen after switching
    edge3 = (route[i], route[(i+2)%N])
    edge4 = (route[(i-1)%N], route[(i+1)%N])
    future = get_weight(graph, edge3) + get_weight(graph, edge4)
    dE = future - current 
    return dE

"""
Creates a list of the changes that would result from flipping any element of the route.
"""
def generate_dE_list(graph, route):
    N = len(route)
    dEs = [0]*N

    for i in range(N):
        dEs[i] = calc_dE(graph, route, i)

    return dEs


"""
Updates the dEs of the neighbors of node and itself.
Used after changing the place of two cities
Nb. modifies dEs in place
"""
def update_neighbors(graph, route, dEs, i):
    N = len(route)
    # update dE of ith city itself and two to either side
    for j in range(i-2, i+3):
        dEs[j%N] = calc_dE(graph, route, j%N)

"""
Calculates the temperature on an exponential scale based on the starting 
two temperatures and the current stage in the iteration. 
"""
def T(n, T0, T1):
    if n == 0:
        return T0
    elif n == 1:
        return T1
    else:
        return (T1/T0)**n * T0


"""
returns True or False
if True, accept the spin flip
if False, reject it
"""
def accept_change(dE, temp, k):
    if dE < 0:
        return True
    else: 
        if temp <= 0:
            return False
        else:
            # print(np.exp(-dE /( k*(temp))))
            return np.exp(-dE /( k*(temp))) >= np.random.rand()

"""
Function to try making a change to the route, and accept it with 
a probability calculated in accept_change. 
"""
def attempt_flip(graph, route, i, j, old_energy,  temp, k):
    # get the change in length from two_opting at i,j
    new_route = two_opt(route, i, j)
    new_energy = calc_route_length(graph, new_route)
    dE = new_energy - old_energy

    # if we should accept the change, keep the two_opted route
    # and update the dEs of it and its neighbours
    # then return True to indicate success
    if accept_change(dE, temp, k):
        return new_energy, new_route, True
    # if we shouldn't accept the change, do nothing and return False to indicate failure
    else:
        return 1, route, False

"""
Function to run the annealing algorithm, pulling together all of the above functions.
Has a stopping condition of 3 successive temperatures with no change in route length,
and a failsafe of 500 loops.
"""
def anneal(graph, orig_route, T0, T1, k):
    # making a copy of the original route so as not to destroy it
    route = orig_route.copy()
    # number of cities (for scale)
    N = len(route)
    # target for number of successful and attempted flips respectively
    size = np.sqrt(N)
    starget = 10*N
    atarget = 3*N*N
    # starting energy
    energy = calc_route_length(graph, route)

    # initialise loop variables
    n = 0
    fails = 0

    while n < 500:
        temp = T(n, T0, T1)

        successes = 0
        attempted = 0

        while attempted < atarget and successes < starget:
            # pick a random node
            i, j = np.random.randint(N, size=2)
            # make sure i <= j or it breaks
            if i > j:
                i, j = j, i
            new_energy, new_route, outcome = attempt_flip(graph, route, i, j, energy, temp, k)

            # if succesful, update energy and add a success
            if outcome:
                energy = new_energy
                route = new_route
                successes += 1

            attempted += 1
        # if the successful flip criterion is not met, 
        # increase the fails counter and if this reaches 3, 
        # end the process
        if attempted >= atarget:
            fails += 1
            if fails == 3:
                break

        # otherwise if enough successful flips, then
        # reset the fails counter
        else:
            fails = 0

        n += 1
        # move to the next n, and hence temperature

        # Progression print
        if n > 1:
            #cursor up one line
            sys.stdout.write('\x1b[1A')
            #delete last line
            sys.stdout.write('\x1b[2K')

        # temperature for printing
        temp0 = np.round(temp, 4)
        print(f"Step number: {n}, Temperature: {temp0}, Consecutive fails: {fails}")

    # adding a new line to the output
    sys.stdout.write("\n")

    return route
 

##############################################
############## BRANCH AND BOUND ##############
##############################################

# Implementing Branch-and-Bound tailored for TSP

"""If sol can be extended by adding edge to Y or N (determined by 'rule')
 returns new_sol, True. Otherwise returns None, False.

nb. edge is a tuple (node1, node2) and rule is:
True for adding 'edge' to Y,
False for adding 'edge' to N"""
def is_feasible(graph, sol, edge, rule):
    Y, N = sol

    # this branch is adding an edge that must be included
    # here we have to check if it requires too many edges for a node
    # or if it creates a cycle
    # Note we can assume Y contains no cycles since it was deemed feasible before
    if rule:
        # checking if the additional edge creates a cycle.
        # to do this we make a subgraph with Y and the new edge and then search
        # for a cycle from one of the new edge's nodes
        Y_extended = list(Y | {edge} ) 
        sg = graph.edge_subgraph(Y_extended)
        
        # using try, except to check for cycles because the function returns an error if there
        # aren't any
        try:
            cycle = nx.find_cycle(sg, edge[0])
        except:
            cycle = None

        if cycle and len(cycle) <= len(graph.nodes) - 1:
            return None, False

        # checking if any node now has too many edges
        # we only have to check the nodes in the subgraph made above,
        # as all other nodes have no required edges.
        for node in list(sg.nodes):
            if sg.degree(node) > 2:
                return None, False

        # if all of that works out fine, then it is feasible so return the solution
        return (Y | {edge}, N), True
        

    # this branch is adding an edge that cannot be included
    # all we have to check here is that each node has at least 2 edges left allowed
    else:
        allowed_edges = set(graph.edges) - (N | {edge})
        # making a subgraph that contains all remaining nodes
        sg = graph.edge_subgraph(allowed_edges)
        # first of all, if any nodes have been lost then clearly the solution isn't feasible
        if len(sg.nodes) != len(graph.nodes): 
            return None, False

        # now check each node and see if they have high enough degree in the remaining edges
        for node in list(sg.nodes):
            if sg.degree(node) < 2:
                return None, False

        # if all of that works out fine, then it is feasible so return the solution
        return (Y, N | {edge}), True
    

def all_extensions(graph, sol):
    Y, N = sol
    # first we find all the edges that are not already allowed/disallowed by the solution,
    # and branch on one of those (if there are any)
    possible_edges = set(graph.edges) - (Y | N)

    rv = []

    if possible_edges:
        edge = possible_edges.pop()
        
        # checking the feasibility of each extension, and keeping track of additional restrictions 
        sol1, flag1 = is_feasible(graph, sol, edge, True)
        if flag1:
            rv.append(sol1)

        sol2, flag2 = is_feasible(graph, sol, edge, False)
        if flag2: 
            rv.append(sol2)
    return rv

def push_extensions(heap, extensions):
    if extensions:
        for ext in extensions:
            Y, N = ext
            # the length is the number of nodes that are required to be in the solution
            # we negate it to perform depth-first search
            l = -len(Y) 

            # we then push these onto the heap, along with a unique id to avoid the heap trying to compare
            # incomparable things
            hq.heappush(heap, (l, id(ext), ext))
            

"""
Calculates a lower bound for the length of the optimal hamiltonian
tour given a graph 'graph', possibly with reduced edge set

Nb. returns a pair (length, success), where success is only True
if sol is a feasible solution
"""
def calc_lb(graph, sol):
    Y, N = sol
    # aliases for graph objects
    nodes = graph.nodes
    edges = graph.edges

    # initialise min length
    length = 0

    # for each node, we add the two smallest weighted edges
    for node in nodes:

        # for each node, we start by including the compulsory edges (i.e. those in Y)
        compulsory_edges = []
        for e in graph.edges.data():
            # Checking if the current node belongs to this edge.
            # Checking if the edge is in the compulsory set of edges 
            # Checking if adding the next edge to the edges up to this point makes a cycke
            if node in e and e[0:2] in Y:
                compulsory_edges.append(e)

        incident = [e for e in graph.edges.data() if node in e and e[0:2] not in N]
        
        # now we know the solution is feasible, so we want the 0, 1, or 2 smallest elements left to fill
        # the node with edges
        sol_edges = hq.nsmallest(2 - len(compulsory_edges), incident, key=lambda x:x[2]["weight"])
        
        # add the compulsory edges to the solution edges
        sol_edges.extend(compulsory_edges)

        # add up the lengths to get the lower bound
        length += sol_edges[0][2]["weight"] + sol_edges[1][2]["weight"]

    return length/2

""" 
Function to get a starting upper bound (e.g. from a heuristic method)
Here we just take a random route to start off with.
"""
def get_ub(graph):
    return calc_route_length(graph, np.random.permutation(graph.nodes))

def branchandbound(graph, upper_bound=None):
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    
    # initialise heap with empty solution
    heap = []
    sol0 = (set(), set())
    hq.heappush(heap, (0, id(sol0), sol0))

    # initialise best solution as nothing and upper bound with function get_ub if none is supplied
    if upper_bound:
        best, ub = [], upper_bound
    else:
        best, ub = [], get_ub(graph)

    # while there are still states to explore:
    while len(heap) > 0:
        # popping (negative) length of solution, id, and sol (which is Y, N))
        l, _, sol = hq.heappop(heap)
        lb = calc_lb(graph, sol)
        
        # if the lower bound for this solution is lower than the upper bound on
        # solution length (plus some constant to account for imprecise float
        # arithmetic), then continue and branch on this node
        if lb < ub + 0.0001:
            Y, N = sol 
            # if we have a complete solution, then update the best so far
            if len(Y) == len(nodes):
                best, ub = Y, lb
            extensions = all_extensions(graph, sol)
            # push any extensions to the heap
            push_extensions(heap, extensions)

    # before returning the route we convert it to a list of cities rather than a set of edges
    best = construct_route_from_edges(list(best))
    return best, ub

##############################################
################# HELD-KARP ##################
##############################################

"""Function to implement the Held-Karp algorithm for finding
exact solutions to the Traveling Salesman Problem"""
def held_karp(graph):
    cities = list(graph.nodes)
    # picking the starting city to be the beginning
    start = cities[0]
    length, route =  D(cities, start, start)
    
    # add starting city to the route
    # we add at the end because it is built backwards anyway
    route.append(start)
    return length, route

"""
Returns the weight of the shortest path between 0 and c through S,
and the path that achieves it
Nb. S is a set for faster removal of elements
"""
def D(S, c, start):

    # if only one thing in S, it must be c
    # so return its distance and the node
    if len(S) == 1:
        return get_weight(graph, (start, c)), []

    # otherwise we find the minimum of all paths up to c
    else:

        # we remove c from S and repeat the problem
        S.remove(c)
        # M holds minimum and best holds best route
        M, best = np.inf, []

        #looping over all nodes in the set
        for x in S:
            # we give the recursive call a copy of S so as not to destroy the input during iteration
            dist, path = D(S.copy(), x, start)
            # adding the distance from x to c separately so I can get the path out
            dist += get_weight(graph, (x, c))

            if dist < M:
                path.append(x)
                M, best = dist, path
        
        return M, best


##############################################
########### RUNNING THE ALGORITHMS ###########
##############################################

""" 
There are not as many interesting cases to test for the Travelling Salesman
Problem as there were for the ising spin problem. Here we simply try the three
methods on a few smallish (so that the exact methods can terminate) sample
problems and see how well annealing compares to the exact solutions.

For a plot of the temperature curve, see the ising spin file.

We try a completely random map, and then a couple of circles. The first circle
is small so the exact solvers can work it out, and the second is larger to show
that annealing is still relatively fast on a larger example (that we can check by eye).
"""

print(
"""
 Ten Randomly Placed Cities
 ==========================
"""
)

N = 10
graph = nx.complete_graph(N)
# Boltzmann factor k to scale temperatures
k = 1

# create and assign positions to the nodes of graph
positions = np.random.uniform(0, N, (N,2))

assign_positions_and_distances(graph, positions)

# generate a random path as a possible candidate for the shortest route
candidate_route = np.random.permutation(graph.nodes)
candidate_length = calc_route_length(graph, candidate_route)
print("Length of random route:", candidate_length)

# running the annealing algorithm
annealed_route = anneal(graph, candidate_route, 20, 19.5, k)
annealed_length = calc_route_length(graph, annealed_route)
print("Length of annealed route:", annealed_length)

# running branch and bound with the annealed solution as an upper bound
bab_route, bab_length = branchandbound(graph, annealed_length)
print("Length of Branch and Bound route:", bab_length)

# running Held-Karp as a final check
hk_length, hk_route = held_karp(graph)
print("Length of Held-Karp route:", hk_length)

draw_cities(graph, [(candidate_route, "Random route"),
                    (annealed_route, "Annealed route"),
                    (bab_route, "Branch-and-Bound route"),
                    (hk_route, "Held-Karp route")])

print(
"""
 A circle of cities
 ==================
"""
)

N = 10
graph = nx.complete_graph(N)
# Boltzmann factor k to scale temperatures
k = 1

# create and assign positions to the nodes of graph
positions = []
for i in range(N):
    positions.append((N*np.cos(2*np.pi * i / N), N*np.sin(2*np.pi * i / N)))

assign_positions_and_distances(graph, positions)

# generate a random path as a possible candidate for the shortest route
candidate_route = np.random.permutation(graph.nodes)
candidate_length = calc_route_length(graph, candidate_route)
print("Length of random route:", candidate_length)

# running the annealing algorithm
annealed_route = anneal(graph, candidate_route, 20, 19.5, k)
annealed_length = calc_route_length(graph, annealed_route)
print("Length of annealed route:", annealed_length)

# running branch and bound with the annealed solution as an upper bound
bab_route, bab_length = branchandbound(graph, annealed_length)
print("Length of Branch and Bound route:", bab_length)

# running Held-Karp as a final check
hk_length, hk_route = held_karp(graph)
print("Length of Held-Karp route:", hk_length)

draw_cities(graph, [(candidate_route, "Random route"),
                    (annealed_route, "Annealed route"),
                    (bab_route, "Branch-and-Bound route"),
                    (hk_route, "Held-Karp route")])


print(
"""
 A bigger circle of cities
 =========================
"""
)

N = 40
graph = nx.complete_graph(N)
# Boltzmann factor k to scale temperatures
k = 1

# create and assign positions to the nodes of graph
positions = []
for i in range(N):
    positions.append((N*np.cos(2*np.pi * i / N), N*np.sin(2*np.pi * i / N)))

assign_positions_and_distances(graph, positions)

# generate a random path as a possible candidate for the shortest route
candidate_route = np.random.permutation(graph.nodes)
candidate_length = calc_route_length(graph, candidate_route)
print("Length of random route:", candidate_length)

# running the annealing algorithm
annealed_route = anneal(graph, candidate_route, 20, 19.5, k)
annealed_length = calc_route_length(graph, annealed_route)
print("Length of annealed route:", annealed_length)

draw_cities(graph, [(candidate_route, "Random route"),
                    (annealed_route, "Annealed route")])
