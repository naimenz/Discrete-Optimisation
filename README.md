# Discrete-Optimisation
Code written for my MT5611 Symbolic Computation project at St Andrews. Simulated annealing and branch-and-bound solvers applied to TSP and ising spin model.

# Discrete Optimisation with Exact and Heuristic Methods

---

## Introduction

This project started with the intention of implementing the Simulated Annealing
(SA) algorithm[^SA-orig] on the Ising ground state problem. From there I got
interested in writing an exact solver that would work on small
examples, and decided to use the same methods on the famous Travelling
Salesman Problem (TSP) - plus one bonus exact solver.

## Ising Spin Ground States

This problem involves trying to find the ground state (lowest energy
configuration) of a set of spins, each of which can point either up or down.
The interaction of these spins, together with an external magnetic field, gives
rise to a Hamiltonian H(S) = - &Sigma;(J<sub>ij</sub> * S<sub>i</sub> *
S<sub>j</sub>) - &mu; &Sigma;(h<sub>j</sub> S<sub>j</sub>), where
J<sub>ij</sub> is the interaction energy between spins S<sub>i</sub> and
S<sub>j</sub>, and h<sub>j</sub> is the local magnetic field at spin
S<sub>j</sub>.  Minimising this is in general and NP hard problem, and so
heuristic methods are often employed.

- ### Simulated Annealing

A rough outline of the SA algorithm (which I used for both Ising and TSP) is:

1. Randomise a starting state, and calculate its energy.

2. Randomly create a new state by changing the current state slightly (e.g.
   flipping a few spins).

3. Calculate the change in energy that would result from moving from the old
   state to the new state.

4. If the change in energy is negative (e.g. closer to the ground state) then
   replace the old state with the new state. If the change is positive, accept
the change with some probability, depending on a parameter called
"temperature".

5. Repeat steps 2-4.

6. Gradually decrease the temperature, which decreases the probability of
   accepting a 'bad' change.

7. Stop the process when very few new changes are accepted (meaning we are
   probably in a relatively good state)

There are a lot of subjective words ('slightly', 'few', 'some'), and this is
deliberate. Many of the parameters are left almost to personal preference,
being different from problem to problem. I performed no rigorous analysis to
come up with these vague values, but simply tuned them until I was satisfied
that the algorithm worked.


- ### Branch-and-Bound

In order to see how well my SA heuristic performed, I needed to know what the
true answer was. All of the exact solvers I found online were way overkill for
my needs, and didn't accept inputs in the way I had developed for SA, so I
wrote my own using a Branch-and-Bound (BB) approach[^BB-ising]. This involves
building a graph of partial solutions where each node puts restrictions on what
the final solution may be (branching), and then discarding solutions that we can
discount as they will never be the best (bounding). This is much like the
strategy for graph colouring and graph isomorphisms encountered in the
lectures/problem sets. The general idea is to:

0. If possible, get an upper bound from some other method (e.g. a heuristic like SA).

1. Start with an empty solution with no restrictions as the start of the graph,
   and calculate a lower bound for the problem.

2. Branch from the starting node (e.g. to two solutions, one with the first
   spin up and one with the first spin down) and calculate lower bounds for
each of these.

3. Repeatedly branch from each node, discarding any solutions whose lower
   bounds are higher than the current upper bound.

4. Whenever a full solution is made, check if it beats the current upper bound.
   If so, update the upper bound and keep the solution as the current best.

5. When there are no nodes left to be branched, terminate the algorithm. The
   optimum solution should have been found.

This algorithm works quickly on small problems (~25 spins), but soon fails to
finish in a reasonable amount of time due to the huge number of possible
states.


## Travelling Salesman Problem (TSP)

The TSP is a famous NP-hard optimisation problem that involves finding the
shortest path between N cities (or reformulated using graph theory, the minimum
weight Hamiltonian cycle in a weighted complete graph on N points). 

- ### Simulated Annealing

SA is again effective here, when using 'two-opt' as the operation[^two-opt]. The two-opt
starts as follows: pick two edges in the cycle, (i j) and (k l) . You then
remove these two edges and add the edges (i k) and (j l) to the cycle. Note
that this always gives you another Hamiltonian cycle.

- ### Held-Karp

The Held-Karp algorithm is an exact solver for the TSP. It relies on the fact that

> Every subpath of a path of minimum distance is itself of minimum distance

The above quote as well as the algorithm itself was found on wikipedia[^HK] and
implemented directly from the description provided there - it is a fairly
simple algorithm to implement recursively.

- ### Branch-and-Bound

I also used BaB for a second time on the TSP. Here the partial solutions
consist of sets of edges that must be included and those that must be excluded.
This was probably the hardest part of the whole project, as implementing all
the necessary restrictions and checks that solutions were still feasible was
quite involved. I took the rough structure of my solution from Section 3 of
[^msword] and from [^nostyle], and the way of thinking about allowed and disallowed edges
as sets Y and N from [^YN].


[^SA-orig]: <http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/TemperAnneal/KirkpatrickAnnealScience1983.pdf>
[^BB-ising]: <http://www.theorie.physik.uni-goettingen.de/~hartmann/nwgruppe/talks/kobe_new.pdf>
[^msword]: <http://www.jot.fm/issues/issue_2003_03/column7.pdf>
[^nostyle]: <http://lcm.csa.iisc.ernet.in/dsa/node187.html>
[^YN]: <http://www.enseignement.polytechnique.fr/informatique/INF431/X09-2010-2011/AmphiLL/branch_and_bound_for_TSP-notes.pdf>
[^two-opt]: <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.7639&rep=rep1&type=pdf>
[^HK]: <https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm#Algorithm>
