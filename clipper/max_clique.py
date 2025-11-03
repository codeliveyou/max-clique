from typing import List
import numpy as np
# import sys

# sys.path.insert(0, '/home/client_2762_9/clipper/build/bindings/python')
# import clipperpy
from build.bindings.python import clipperpy

def maximum_clique_algorithm(num_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    # Build the affinity matrix M (adjacency)
    M = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, neighbors in enumerate(adjacency_list):
        for j in neighbors:
            M[i, j] = 1.0
            M[j, i] = 1.0

    # Build the constraint matrix C - for max clique usually same as M
    C = np.array(M, copy=True)
    np.fill_diagonal(M, 0)
    np.fill_diagonal(C, 0)
    M = np.ascontiguousarray(M, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)

    params = clipperpy.Params()
    invariant = clipperpy.invariants.PairwiseInvariant()
    clipper_instance = clipperpy.CLIPPER(invariant, params)
    clipper_instance.set_matrix_data(M, C)
    clipper_instance.solve_as_maximum_clique()
    result = clipper_instance.get_solution()
    return result

# Example test
num_nodes = 5
adj_list = [
    [1, 2],
    [0, 2],
    [0, 1, 3],
    [2, 4],
    [3]
]

max_clique_sol = maximum_clique_algorithm(num_nodes, adj_list)
print("Maximum Clique found:", max_clique_sol.nodes)
