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


def genetic_algorithm(number_of_nodes: int, adjacency_list: list[list[int]], generations=500, populationNum=10, localImprovement=10, mutations=1, uniqueIterations=100, shuffleTolerance=10) -> list[int]:
    clq_filename = save_clq_file(number_of_nodes, adjacency_list)
    max_clique = []
    parameters = [
        [500, 10, 10, 1, 100, 10],
        [500, 5, 10, 1, 100, 10],
        [500, 15, 10, 1, 100, 10],
        [500, 20, 10, 1, 100, 10],
        [500, 10, 10, 5, 100, 10],
        [500, 10, 10, 10, 100, 10],
        [500, 10, 10, 1, 50, 10],
        [500, 10, 10, 1, 100, 5],
        [500, 10, 10, 15, 100, 10],
        # [500, 10, 10, 20, 100, 10]
    ]
    for parameter in parameters:
        try:
            tmp_max_clique = gaclique.run_max_clique(clq_filename, parameter[0], parameter[1], parameter[2], parameter[3], parameter[4], parameter[5])
            if len(tmp_max_clique) > len(max_clique):
                max_clique = tmp_max_clique.copy()
            # print(f"{parameter}: {len(tmp_max_clique)}")
        except Exception as e:
            pass
            # print(f"Error running with parameters {parameter}: {e}")

    # print("Mutations", end='')
    # for i in range(1, 11):
    #     tmp_max_clique = gaclique.run_max_clique(clq_filename, generations, populationNum, localImprovement, i, uniqueIterations, shuffleTolerance)
    #     if len(tmp_max_clique) > len(max_clique):
    #         max_clique = tmp_max_clique.copy()
    #     print(f", {i}: {len(tmp_max_clique)}", end='')
    # print("")

    # print("Populations", end='')
    # for i in range(5, 16):
    #     tmp_max_clique = gaclique.run_max_clique(clq_filename, generations, i, localImprovement, mutations, uniqueIterations, shuffleTolerance)
    #     if len(tmp_max_clique) > len(max_clique):
    #         max_clique = tmp_max_clique.copy()
    #     print(f", {i}: {len(tmp_max_clique)}", end='')
    # print("")

    # print("LocalImprovement", end='')
    # for i in range(5, 16):
    #     tmp_max_clique = gaclique.run_max_clique(clq_filename, generations, populationNum, i, mutations, uniqueIterations, shuffleTolerance)
    #     if len(tmp_max_clique) > len(max_clique):
    #         max_clique = tmp_max_clique.copy()
    #     print(f", {i}: {len(tmp_max_clique)}", end='')
    # print("")

    # print("UniqueIterations", end='')
    # for i in range(50, 151, 10):
    #     tmp_max_clique = gaclique.run_max_clique(clq_filename, generations, populationNum, localImprovement, mutations, i, shuffleTolerance)
    #     if len(tmp_max_clique) > len(max_clique):
    #         max_clique = tmp_max_clique.copy()
    #     print(f", {i}: {len(tmp_max_clique)}", end='')
    # print("")

    # print("ShuffleTolerance", end='')
    # for i in range(5, 16):
    #     tmp_max_clique = gaclique.run_max_clique(clq_filename, generations, populationNum, localImprovement, mutations, uniqueIterations, i)
    #     if len(tmp_max_clique) > len(max_clique):
    #         max_clique = tmp_max_clique.copy()
    #     print(f", {i}: {len(tmp_max_clique)}", end='')
    # print("")

    # print("Genetic algorithm result: ", len(max_clique))
    return [(x - 1) for x in max_clique]

# c++ -I/home/user/project/include -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) gaClique.cpp gaclique_pybind.cpp -o gaclique$(python3-config --extension-suffix)
