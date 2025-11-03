from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

from .gnn_models import SC_MODEL

import requests
import time


def scattering_clique_algorithm(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
    def extend_to_maximal_clique(
        adjacency_list: list[list[int]], clique: list[int]
    ) -> list[int]:
        clique_set = set(clique)
        n = len(adjacency_list)
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if v in clique_set:
                    continue
                neighbors = set(adjacency_list[v])
                if clique_set.issubset(neighbors):
                    clique_set.add(v)
                    changed = True
                    break
        return list(clique_set)

    num_nodes = number_of_nodes
    adjacency_list = adjacency_list
    
    ####################################
    total_edge = int(sum([len(x) for x in adjacency_list]) / 2)
    print(f"Nodes: {num_nodes}, Edges: {total_edge}")
    # print(int(num_nodes * (num_nodes - 1) / 2))
    # print(f"Edge persent: {total_edge / (num_nodes * (num_nodes - 1) / 2)}")
    ####################################

    maximum_clique: list[int] = []

    start = time.time()

    if num_nodes < 400:
        try:
            response = requests.post(
                "http://localhost:8008/max_clique",
                json={"num_nodes": num_nodes, "adjacency_list": adjacency_list}
            )
            response.raise_for_status()
            maximum_clique = response.json()["max_clique"]
        except Exception as e:
            print("Failed to get maximum clique from CLIPPER API:", e)
            maximum_clique = []
    
    if len(maximum_clique) == 0:
        for clique in SC_MODEL.predict_iter(num_nodes, adjacency_list):
            clique = extend_to_maximal_clique(adjacency_list, list(map(int, clique)))
            if len(clique) > len(maximum_clique):
                maximum_clique = clique
    
    end = time.time()
    print(f"Clipper runtime: {end - start:.4f} seconds")


    return maximum_clique