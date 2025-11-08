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
    # print(f"Nodes: {num_nodes}, Edges: {total_edge}")
    # print(int(num_nodes * (num_nodes - 1) / 2))
    # print(f"Edge persent: {total_edge / (num_nodes * (num_nodes - 1) / 2)}")
    ####################################

    maximum_clique: list[int] = []

    start = time.time()

    iner_adjacency_list = []
    over_adjacency_list = []
    iner_num_nodes = 0
    over_num_nodes = 0
    over_limit = 300
    for vtx in range(num_nodes):
        if vtx < over_limit:
            iner_adjacency_list.append([])
            iner_num_nodes += 1
            for u in adjacency_list[vtx]:
                if u < over_limit:
                    iner_adjacency_list[-1].append(u)
        # else:
        #     over_adjacency_list.append([])
        #     over_num_nodes += 1
        #     for u in adjacency_list[vtx]:
        #         if u >= over_limit:
        #             over_adjacency_list[-1].append(u - over_limit)
    
    try:
        response = requests.post(
            "http://localhost:8008/max_clique",
            json={"num_nodes": iner_num_nodes, "adjacency_list": iner_adjacency_list}
        )
        response.raise_for_status()
        maximum_clique = response.json()["max_clique"]
        if num_nodes > over_limit:
            over_good_nodes = []
            for vtx in range(over_limit, num_nodes):
                is_good = True
                for v in maximum_clique:
                    if v not in adjacency_list[vtx]:
                        is_good = False
                        break
                if is_good:
                    over_good_nodes.append(vtx)
            if len(over_good_nodes) > 0:
                # print("Over good nodes: ", len(over_good_nodes), over_good_nodes)
                over_adjacency_list = [[] for _ in range(len(over_good_nodes))]
                for i in range(len(over_good_nodes)):
                    over_num_nodes += 1
                    for j in range(len(over_good_nodes)):
                        if over_good_nodes[j] in adjacency_list[over_good_nodes[i]]:
                            over_adjacency_list[i].append(j)
                # print("Over Adj: ", over_adjacency_list)
                over_response = requests.post(
                    "http://localhost:8008/max_clique",
                    json={"num_nodes": over_num_nodes, "adjacency_list": over_adjacency_list}
                )
                over_response.raise_for_status()
                over_maximum_clique = over_response.json()["max_clique"]
                # print("Over maximum clique: ", len(over_maximum_clique), over_maximum_clique)
                for i in over_maximum_clique:
                    maximum_clique.append(over_good_nodes[i])
                # print("Added over_max clique")

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