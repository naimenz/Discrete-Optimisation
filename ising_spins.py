import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
from heapq import heappush, heappop
from random import choice, seed

# setting the seeds for reproducibility
np.random.seed(3)
seed(3)

##############################################
############# SET-UP FUNCTIONS ###############
##############################################

# Functions for building matrices of J (interaction) values where the i,jth element is the interaction
# between spin i and spin j.

"""
Function to make a matrix of J values for N particles on an m*n grid
Nb. behaviour only guaranteed for m,n>=3
"""
def make_square_J_matrix(Jij, m, n):
    N = m*n
    J = np.zeros((N,N))
    for x in range(N):
        # We split the number x into a base (number of rows down)
        # and an offset (how far along each row)
        base, offset = x // n, x % n
        
        # then we change each by one to get the neighbours in the grid,
        # and use modulo arithmetic to make sure we stay inside the bounds
        # and convert back to a number
        neighbours = []
        neighbours.append(((base-1)%m) * n + offset)
        neighbours.append(((base+1)%m) * n + offset)
        neighbours.append(base * n + (offset-1)%n)
        neighbours.append(base * n + (offset+1)%n)
        # looping over each neighbour and changing its entry in the matrix
        # note we only include if it is in the upper triangle, i.e. neighbour > x
        for neighbour in neighbours:
            if neighbour > x:
                J[x, neighbour] = Jij
       
    return J

"""
Function to make a matrix of J values for N particles on an m*n grid
using random Gaussians between lo and hi for each Jij
Nb. behaviour guaranteed for m,n>=3
"""
def make_square_normal_J_matrix(lo, hi, m, n):
    N = m*n
    J = np.zeros((N,N))
    for x in range(N):
        # We split the number x into a base (number of rows down)
        # and an offset (how far along each row)
        base, offset = x // n, x % n
        
        # then we change each by one to get the neighbours in the grid,
        # and use modulo arithmetic to make sure we stay inside the bounds
        # and convert back to a number
        neighbours = []
        neighbours.append(((base-1)%m) * n + offset)
        neighbours.append(((base+1)%m) * n + offset)
        neighbours.append(base * n + (offset-1)%n)
        neighbours.append(base * n + (offset+1)%n)
        # looping over each neighbour and changing its entry in the matrix
        # note we only include if it is in the upper triangle, i.e. neighbour > x
        for neighbour in neighbours:
            if neighbour > x:
                J[x, neighbour] = np.random.normal(lo, hi)
       
    return J

"""
Function to make a matrix of J values for N particles on an m*n triangular grid
Nb. behaviour guaranteed for m,n>=3
"""
def make_triangular_J_matrix(Jij, m, n):
    N = m*n
    J = np.zeros((N,N))
    for x in range(N):
        # We split the number x into a base (number of rows down)
        # and an offset (how far along each row)
        base, offset = x // n, x % n
        
        # then we change each by one to get the neighbours in the grid,
        # and use modulo arithmetic to make sure we stay inside the bounds
        # and convert back to a number
        neighbours = []
        neighbours.append(((base-1)%m) * n + offset)
        neighbours.append(((base+1)%m) * n + offset)
        neighbours.append(base * n + (offset-1)%n)
        neighbours.append(base * n + (offset+1)%n)

        #triangular neighbours
        neighbours.append(((base-1)%m) * n + (offset-1)%n)
        neighbours.append(((base+1)%m) * n + (offset+1)%n)


        # looping over each neighbour and changing its entry in the matrix
        # note we only include if it is in the upper triangle, i.e. neighbour > x
        for neighbour in neighbours:
            if neighbour > x:
                J[x, neighbour] = Jij
       
    return J

"""
Creates a dictionary from a matrix of J values and assigns
them to the graph 'lattice'
"""
def assign_Jdict_from_matrix(lattice, J):
    Jdict = {edge: J[edge[0], edge[1]] for edge in lattice.edges}
    nx.set_edge_attributes(lattice, Jdict, "J")
    
"""
Assigning spins to the lattice, based on the value of 'spin'

if spin = 0, assigns randomly between 1 and -1
if spin = 1, assign all 1
if spin = -1, assign all -1
"""
def assign_spins(lattice, spin):

    if spin == 0:
        # a dictionary holding random spins to assign to the nodes of the lattice
        spins = {node: np.random.choice([-1,1]) for node in lattice.nodes}
    else:
        spins = {node: spin for node in lattice.nodes}

    nx.set_node_attributes(lattice, spins, "spin") # assign spins to nodes

"""
Function to apply magnetic field to a lattice, pulling random gaussian floats between lo and hi
"""
def assign_h_field(lattice, lo, hi):
    if lo == hi:
        hs = {node: lo for node in lattice.nodes}

    else:
        hs = {node: np.random.normal(lo, hi)  for node in lattice.nodes}

    nx.set_node_attributes(lattice, hs, "h")


# Function for drawing the spins

"""
Draws the spin lattice with nodes coloured by spin and 
edge (i,j) labelled by Jij
"""
def draw_lattice(lattice, m, n, title=None):
    # create colourmap from the spins
    cmap = [spin for node,spin in nx.get_node_attributes(lattice, "spin").items()]
    # creating positions for drawing
    pos = {node: (node//n,node%n) for node in lattice.nodes}
    # pos = dict(zip(lattice, lattice))
    nx.draw_networkx_nodes(lattice,pos,node_color=cmap, with_labels=True)
    edgelist = [edge for edge in lattice.edges if lattice.edges[edge]["J"] != 0]
    nx.draw_networkx_edges(lattice, pos, edgelist)
    
    #TEST
    # if a title is supplied, add it to the plot
    if title:
        plt.title(title)
    


##############################################
############ SIMULATED ANNEALING #############
##############################################

# Functions to implement Simulated Annealing on ising spins.
# Uses networkx to implement the system, first done with numpy arrays but
# turns out to be simpler to do with graphs.

"""Calculate the energy of the current lattice state

Nb. note the - sign before J; we follow the convention of the hamiltonian consisting of -Jij SiSj
"""
def H(lattice):
    # get the spins for each node 
    spins = nx.get_node_attributes(lattice, "spin")

    #initialise energy
    energy = 0

    # loop over all nodes
    for node in lattice.nodes:

        # first add contribution from magnetic field (note negative by convention)
        energy += -lattice.nodes[node]["h"] * spins[node]

        for neighbor, attrs in lattice[node].items():
            # we only use the edge if node < neighbor so that we avoid
            # double counting edges
            if node < neighbor:
                energy += -attrs["J"] * spins[node] * spins[neighbor]

    return energy

"""Calculates the change in energy that would occur by flipping the spin
of 'node'
Nb. note the - sign before J; we follow the convention of the hamiltonian consisting of -Jij SiSj
"""
def calc_dE(lattice, node):
    dE = 0
    # get the spins for each node 
    spins = nx.get_node_attributes(lattice, "spin")

    # first energy contribution is from the magnetic field on this node
    dE -= 2 * -lattice.nodes[node]["h"] * spins[node]

    # for each neighbor, we subtract twice the current energy contribution
    # to see what would happen if we flipped it
    for neighbor, attrs in lattice[node].items():
        dE -= 2 * -attrs["J"] * spins[node] * spins[neighbor]

    return dE

"""
Creates a dictionary of the form {node: dE}
and assigns it to the nodes of the lattice
"""
def assign_dEs(lattice):
    dEs = {} 

    # loop over all nodes
    for node in lattice.nodes:
        dEs[node] = calc_dE(lattice, node)

    nx.set_node_attributes(lattice, dEs, "dE")


"""
Updates the dEs of the neighbors of node and itself.
Used after flipping the spin of node
"""
def update_neighbors(lattice, node):
    # update node itself and then each neighbor
    lattice.nodes[node]["dE"] = calc_dE(lattice, node)
    for neighbor in lattice.neighbors(node):
        if lattice.edges[(node,neighbor)]["J"] != 0:
            lattice.nodes[neighbor]["dE"] = calc_dE(lattice, neighbor)


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
def accept_change(dE, T):
    if dE < 0:
        return True
    # # just for now, handle no change in energy separately (otherwise accepted every time)
    # elif delta_E == 0:
    #     return False
    else: 
        if T <= 0:
            return False
        else:
            return (np.exp(-dE / (T)) >= np.random.rand())


"""
Function to try making a change to the route, and accept it with 
a probability calculated in accept_change. 
"""
def attempt_flip(lattice, node, T):
    dE = lattice.nodes[node]["dE"]

    # if we should accept the change, flip the sign of sigma[i,j]
    # and update the dEs of it and its neighbours
    # then return True to indicate success
    if accept_change(dE, T):
        # flip the spin attribute of node
        lattice.nodes[node]["spin"] *= -1
        # update the dEs of node and its neighbors
        update_neighbors(lattice, node)
        return dE, True
    # if we shouldn't accept the change, do nothing and return False to indicate failure
    else:
        return 1, False

"""
Function to run the annealing algorithm, pulling together all of the above functions.
Has a stopping condition of 3 successive temperatures with no change in energy,
and a failsafe of 500 loops.
"""
def anneal(orig_lattice, T0, T1):
    # creating a copy so as not to destroy original lattice
    lattice = orig_lattice.copy()
    # root of number of points in the lattice (as a sense of lattice size)
    N = np.sqrt(len(lattice.nodes))
    # target for number of successful and attempted flips respectively
    starget = 1*N
    atarget = 100*N
    # starting energy
    energy = H(lattice)
    # assigning the changes in energy to the lattice
    assign_dEs(lattice)

    # initialise loop variables
    n = 0
    fails = 0

    while n < 500:
        temp = T(n, T0, T1)

        successes = 0
        attempted = 0

        while attempted < atarget and successes < starget:
            # pick a random node
            node = choice(list(lattice.nodes))
            dE, outcome = attempt_flip(lattice, node, temp)

            # if succesful, update energy and add a success
            if outcome:
                energy += dE
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
        # if you want to see all of the output at once, simply remove this if block
        if n > 1:
            #cursor up one line
            sys.stdout.write('\x1b[1A')

            #delete last line
            sys.stdout.write('\x1b[2K')

        # rounded temperature for printing
        temp0 = np.round(temp, 4)
        print(f"Step number: {n}, Temperature: {temp0}, Consecutive fails: {fails}")


    # adding a new line to the output
    sys.stdout.write("\n")

    return lattice


##############################################
############## BRANCH AND BOUND ##############
##############################################

# Implementing the Branch-and-Bound algorithm for ising spins

"""
Function to calculate a lower bound for the energy of a given partial state (node on the graph of
partial states).
"""
def get_El(l, partial=None, Eprev=None):
   # l: current branch level, 0 indexed
   # Eprev: energy of previous node
   # J: 2d array of interaction energies

   J = list(partial.edges.data("J"))
   h = list(partial.nodes.data("h"))
   spins = np.array(partial.nodes.data("spin"))

   # if l is 1, this is the starting energy
   if l == 1:
       # summing over the J values and adding the h values
       El = np.sum(-np.abs([edge[2] for edge in J]))+np.sum(-np.abs([node[1] for node in h]))
       
       # note the first spin might have h increasing the energy
       if spins[l-1][1] == -np.sign(h[0][1]):
           El += 2*np.abs(h[0][1])
       return El


   El = Eprev
   # sum over all spins parallel to the current spin in the partial
   # solution
   for i in range(l-1):
       # If the spins align and J for that edge is negative,
       # then add it to El (as we would not get that contribution, under
       # convention that the hamiltonian is -Jij)
       # Similarly, if the spins differ and J is positive, add it
       Jij = partial.edges[(i,l-1)]["J"]
       if spins[i][1] * spins[l-1][1] == -np.sign(Jij):
           El += 2*np.abs(Jij)

   # finally, add the contribution from the magnetic field (if the new
   # spin increases the energy)
   hi = h[l-1][1]
   if spins[l-1][1] == -np.sign(hi):
       El += 2*np.abs(hi)
   return El


"""
Return both possible extensions of a partial solution.
Note partial solution is a complete graph on all N points
but is missing spins for some nodes
"""
def extend(l, sol):
    up, down = sol.copy(), sol.copy()
    up.nodes[l]["spin"] = 1
    down.nodes[l]["spin"] = -1
    return up, down

"""
Calculate an upper bound for the given lattice.
(Simply takes the energy of the current spin state in the lattice)
"""
def get_upper_bound(lattice):
    return H(lattice)

"""Function to add a state to the heap"""
def add_sol(heap, l, sol, Eprev):
    # calculate the energy of sol and add it to the heap
    E = get_El(l, sol, Eprev)
    # we add the id of the object to break ties so heapq doesnt
    # try to compare graphs (which throws an error)
    heappush(heap, (E, l, id(sol), sol))

"""Branch on a given node"""
def branch(l, old_sol, Eprev, heap):
    extensions = extend(l, old_sol)
    for sol in extensions:
        add_sol(heap, l+1, sol, Eprev)

"""
Run the branch and bound algorithm for a given lattice.
Either takes an upper bound as input, or calculates one from get_upper_bound
"""
def bb(input_lattice, upper_bound=None):
    # we make a copy of the original lattice to preserve J and h
    lattice = input_lattice.copy()
    N = len(lattice.nodes)
    # create heap
    heap = []
    # initialising upper bound
    if upper_bound:
        ub = upper_bound
    else:
        ub = get_upper_bound(lattice)

    # wiping the spins and setting the first spin up/down
    nx.set_node_attributes(lattice, 0, "spin")
    start_up = lattice.copy()
    start_down = lattice.copy()
    start_up.nodes[0]["spin"] = 1
    start_down.nodes[0]["spin"] = -1


    # add first solution to the heap
    add_sol(heap, 1, start_up, 0)
    add_sol(heap, 1, start_down, 0)

    #creating a variable to hold the current best full solution
    best = (np.inf, None)

    # iterate until heap is empty
    while len(heap) > 0:
        # unpack element on heap
        Eprev, l, _,  old_sol = heappop(heap)
        
        # if we have reached the correct length, compare solutions
        if l == N:
            if Eprev < best[0]:
                best = (Eprev, old_sol)
        # adding a small constant to avoid float rounding errors causing it to fail
        elif Eprev <= ub + 0.0001:
            branch(l, old_sol, Eprev, heap)

    return best

##############################################
########### RUNNING THE ALGORITHMS ###########
##############################################

""" 
Here we run all of the functions on some example systems.

Before running the algorithms, we plot the temperature curve that is used as
the annealing progresses.

We start with simple ferromagnetic and antiferrogmagnetic systems with a square
lattice.  These systems have no frustration, i.e. there's not competition
between adjacent spins, they all "want to help" each other.

Then we implement a uniform external magnetic field on the antiferromagnetic
case.  Provided we pick a strong enough magnetic field, this should overcome
the tendency of an antiferromagnetic system for spins to anti-align with their
neighbours, and we should get a uniform state.

After that, we see what happens when we use a triangular lattice structure instead
of square with antiferromagnetic interactions. Here there should be
frustration, as any 3 spins cannot all anti-align as they would like to.
For the triangular lattice we use a smaller system as annealing doesn't give such
a good upper bound, and so Branch and Bound can take a long time.

We then try a completely random square system, with Gaussian random numbers
between -1 and 1 for spin interactions, and the same for the magnetic field
felt by each spin. This also uses a relatively small number of spins so that
Branch and Bound can finish and we can see how well annealing did.  

The final example is a large square lattice antiferromagnetic example. Here we
only run annealing, as the exact solvers would take too long, and we know what
the ground state should be (spins anti-aligned with their neighbours). This
shows that annealing is much quicker than the exact solvers and can still come
up with very good solutions.
"""

print(
"""
 TEMPERATURE CURVE 
 =================
"""
)

# plotting the temperature curve against step in the process
ns = np.array(range(500))
Ts = [T(n, 20, 19.5) for n in ns]
plt.plot(ns, Ts)
plt.title("Temperature curve over 500 iterations")
plt.xlabel("Step number")
plt.ylabel("Temperature")
plt.show()

print(
"""
 FERROMAGNETIC EXAMPLE 
 =====================
"""
)

# First we pick a lattice size and initialise it as a complete graph
# rows, cols
m, n = 4,4
# Number of particles
N = m*n

lattice = nx.complete_graph(N)
# making a ferromagnetic square lattice (i.e. spins interact above, below, and to left and right)
J_ferro = make_square_J_matrix(1, m, n)
assign_Jdict_from_matrix(lattice, J_ferro)

# Starting with a uniform J and no magnetic field
assign_spins(lattice, 0)
assign_h_field(lattice, 0, 0)

# Plotting all three on same figure
plt.figure(figsize=(14,4))
plt.subplot(131)

print("Energy of the lattice BEFORE annealing: ",H(lattice))
draw_lattice(lattice, m, n, "Random spin arrangement")

# Run annealing on the above defined lattice
annealed_sol = anneal(lattice, 20, 19.5)
annealed_energy = H(annealed_sol)
print("Energy of the lattice AFTER annealing: ", annealed_energy,)

plt.subplot(132)
draw_lattice(annealed_sol, m, n,  "Annealing of a square\n ferromagnetic system")

# Running branch and bound on the above lattice, using the annealed energy as an upper bound
bb_energy, bb_sol = bb(lattice, annealed_energy)
print("Energy of the lattice AFTER Branch and Bound: ", bb_energy)

plt.subplot(133)
draw_lattice(bb_sol, m, n, "Exact solution of square\n ferromagnetic system from BaB")

plt.show()

print(
"""
 ANTIFERROMAGNETIC EXAMPLE
 =========================
 """
 )

# First we pick a lattice size and initialise it as a complete graph
# rows, cols
m, n = 4,4
# Number of particles
N = m*n

lattice = nx.complete_graph(N)
# making an antiferromagnetic square lattice (i.e. spins interact above, below, and to left and right)
J_antiferro = make_square_J_matrix(-1, m, n)
assign_Jdict_from_matrix(lattice, J_antiferro)

# Starting with no magnetic field
assign_spins(lattice, 0)
assign_h_field(lattice, 0, 0)

# Plotting all three on the same figure
plt.figure(figsize=(14,4))
plt.subplot(131)

print("Energy of the lattice BEFORE annealing: ",H(lattice))
draw_lattice(lattice, m, n, "Random spin arrangement")

# Run annealing on the above defined lattice
annealed_sol = anneal(lattice, 20, 19.5)
annealed_energy = H(annealed_sol)
print("Energy of the lattice AFTER annealing: ", annealed_energy)

plt.subplot(132)
draw_lattice(annealed_sol, m, n,  "Annealing of a square\n antiferromagnetic system")

# Running branch and bound on the above lattice, using the annealed energy as an upper bound
bb_energy, bb_sol = bb(lattice, annealed_energy)
print("Energy of the lattice AFTER Branch and Bound: ", bb_energy)

plt.subplot(133)
draw_lattice(bb_sol, m, n, "Exact solution of square\n antiferromagnetic system from BaB")

plt.show()

print(
"""
 STRONG MAGNETIC FIELD EXAMPLE
 =============================
"""
)

# First we pick a lattice size and initialise it as a complete graph
# rows, cols
m, n = 4,4 
# Number of particles
N = m*n

lattice = nx.complete_graph(N)
assign_spins(lattice, 0)

# making an antiferromagnetic square lattice (i.e. spins interact above, below, and to left and right)
J_antiferro = make_square_J_matrix(-1, m, n)
assign_Jdict_from_matrix(lattice, J_antiferro)

# Adding a strong external magnetic field 
assign_h_field(lattice, 50, 50)


# Plotting all three on the same figure
plt.figure(figsize=(14,4))
plt.subplot(131)

print("Energy of the lattice BEFORE annealing: ",H(lattice))
draw_lattice(lattice, m, n, "Random spin arrangement")

# Run annealing on the above defined lattice
annealed_sol = anneal(lattice, 20, 19.5)
annealed_energy = H(annealed_sol)
print("Energy of the lattice AFTER annealing: ", annealed_energy)

plt.subplot(132)
draw_lattice(annealed_sol, m, n,  "Annealing of a square antiferro system\n with a strong magnetic field")

# Running branch and bound on the above lattice, using the annealed energy as an upper bound
bb_energy, bb_sol = bb(lattice, annealed_energy)
print("Energy of the lattice AFTER Branch and Bound: ", bb_energy)

plt.subplot(133)
draw_lattice(bb_sol, m, n, "Exact solution of square antiferro\n system with strong magnetic field from BaB ")

plt.show()

print(
"""
 TRIANGULAR ANTIFERROMAGNETIC EXAMPLE
 ====================================
"""
)

# First we pick a lattice size and initialise it as a complete graph
# rows, cols
m, n = 4, 4
# Number of particles
N = m*n

lattice = nx.complete_graph(N)
assign_spins(lattice, 0)

# making an antiferromagnetic triangular lattice (each spin has 3 neighbours instead of 4)
J_antiferro = make_triangular_J_matrix(-1, m, n)
assign_Jdict_from_matrix(lattice, J_antiferro)

# No external field
assign_h_field(lattice, 0, 0)


# Plotting all three on the same figure
plt.figure(figsize=(14,4))
plt.subplot(131)

print("Energy of the lattice BEFORE annealing: ",H(lattice))
draw_lattice(lattice, m, n, "Random spin arrangement")

# Run annealing on the above defined lattice
annealed_sol = anneal(lattice, 20, 19.5)
annealed_energy = H(annealed_sol)
print("Energy of the lattice AFTER annealing: ", annealed_energy)

plt.subplot(132)
draw_lattice(annealed_sol, m, n,  "Annealing of a triangular\n antiferromagnetic system")

# Running branch and bound on the above lattice, using the annealed energy as an upper bound
bb_energy, bb_sol = bb(lattice, annealed_energy)
print("Energy of the lattice AFTER Branch and Bound: ", bb_energy)

plt.subplot(133)
draw_lattice(bb_sol, m, n, "Exact solution of triangular\n antiferromagnetic system from BaB")

plt.show()

print(
"""
 RANDOM SQUARE EXAMPLE
 =====================
"""
)

# First we pick a lattice size and initialise it as a complete graph
# rows, cols
m, n = 4,4
# Number of particles
N = m*n

lattice = nx.complete_graph(N)
assign_spins(lattice, 0)

# making a random square lattice
J = make_square_normal_J_matrix(-1, 1, m, n)
assign_Jdict_from_matrix(lattice, J)

# Random external field at each point
assign_h_field(lattice, -1, 1)

# Plotting all three on the same figure
plt.figure(figsize=(14,4))
plt.subplot(131)

print("Energy of the lattice BEFORE annealing: ",H(lattice))
draw_lattice(lattice, m, n, "Random spin arrangement")

# Run annealing on the above defined lattice
annealed_sol = anneal(lattice, 20, 19.5)
annealed_energy = H(annealed_sol)
print("Energy of the lattice AFTER annealing: ", annealed_energy)

plt.subplot(132)
draw_lattice(annealed_sol, m, n,  "Annealing of a random\n system on a square lattice")

# Running branch and bound on the above lattice, using the annealed energy as an upper bound
bb_energy, bb_sol = bb(lattice, annealed_energy)
print("Energy of the lattice AFTER Branch and Bound: ", bb_energy)

plt.subplot(133)
draw_lattice(bb_sol, m, n, "Exact solution of random system\n on square lattice from BaB")

plt.show()


print(
"""
 LARGE FERROMAGNETIC EXAMPLE
 ===========================
"""
)


# First we pick a lattice size and initialise it as a complete graph
# rows, cols
m, n = 12, 12
# Number of particles
N = m*n

lattice = nx.complete_graph(N)
assign_spins(lattice, 0)

# making a square ferromagnetic lattice
J = make_square_J_matrix(-1, m, n)
assign_Jdict_from_matrix(lattice, J)

# No external field
assign_h_field(lattice, 0, 0)

# Plotting all three on the same figure
plt.figure(figsize=(14,4))
plt.subplot(121)

print("Energy of the lattice BEFORE annealing: ",H(lattice))
draw_lattice(lattice, m, n, "Random spin arrangement")

# Run annealing on the above defined lattice
annealed_sol = anneal(lattice, 20, 19.5)
annealed_energy = H(annealed_sol)
print("Energy of the lattice AFTER annealing: ", annealed_energy)

plt.subplot(122)
draw_lattice(annealed_sol, m, n,  "Annealing of a random\n system on a square lattice")

plt.show()
