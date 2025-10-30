import numpy as np
from clipper.build.bindings.python import clipperpy

# def maximum_clique_algorithm(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
#     # Create adjacency matrix from adjacency_list
#     adj_matrix = np.zeros((number_of_nodes, number_of_nodes), dtype=np.bool_)
#     for i, neighbors in enumerate(adjacency_list):
#         for j in neighbors:
#             adj_matrix[i, j] = True
#             adj_matrix[j, i] = True  # assuming undirected graph

#     # Convert numpy matrix to Eigen dense matrix in clipperpy
#     adj_matrix_double = adj_matrix.astype(np.float64)

#     # Set default solver parameters
#     params = clipperpy.maxclique.Params()
#     params.verbose = False
#     params.method = clipperpy.maxclique.Method.EXACT  # EXACT, HEU, KCORE, etc.
#     params.threads = 1
#     params.time_limit = 60.0

#     # Solve max clique
#     max_clique = clipperpy.maxclique.solve(adj_matrix_double, params)

#     return max_clique


# # Test example: graph with 5 nodes and edges forming a max clique of size 3
# num_nodes = 5
# adj_list = [
#     [1, 2],    # node 0 connected to 1, 2
#     [0, 2],    # node 1 connected to 0, 2
#     [0, 1, 3], # node 2 connected to 0, 1, 3
#     [2, 4],    # node 3 connected to 2, 4
#     [3]        # node 4 connected to 3
# ]

# result = maximum_clique_algorithm(num_nodes, adj_list)
# print("Maximum Clique found:", result)


print(dir(clipperpy))
