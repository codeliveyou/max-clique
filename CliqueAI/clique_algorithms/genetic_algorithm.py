import gaclique

solver = gaclique.GACliqueSolver()
solver.set_graph([
    [1, 2],
    [0, 2],
    [0, 1]
])  # Example adjacency list

max_clique = solver.run(10000)
print("Maximum clique found:", max_clique)


def genetic_algorithm(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
    max_clique = []
    
    return max_clique