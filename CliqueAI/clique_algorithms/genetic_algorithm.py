import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gaclique

def save_clq_file(number_of_nodes, adjacency_list, folder="saved_graph"):
    # Get the directory of the current script (genetic_algorithm.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"graph_{timestamp}.clq"
    full_path = os.path.join(folder_path, filename)

    # Extract edges as unordered pairs (1-based node ids)
    edges = set()
    for i, neighbors in enumerate(adjacency_list):
        for j in neighbors:
            u, v = i+1, j+1  # clq format is 1-based
            if u != v:
                edge = tuple(sorted((u, v)))
                edges.add(edge)

    with open(full_path, "w") as f:
        f.write(f"p edge {number_of_nodes} {len(edges)}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    return full_path


def genetic_algorithm(number_of_nodes: int, adjacency_list: list[list[int]], generations=500) -> list[int]:
    clq_filename = save_clq_file(number_of_nodes, adjacency_list)
    max_clique = gaclique.run_max_clique(clq_filename, generations)
    print("Genetic algorithm result: ", len(max_clique))
    return [(x - 1) for x in max_clique]

# nodes = 3
# adj_list = [[1, 2], [0, 2], [0, 1]]
# print(genetic_algorithm(nodes, adj_list, generations=1000))
