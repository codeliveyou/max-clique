import json
import os
import shutil
import csv

from CliqueAI.clique_algorithms import (networkx_algorithm,
                                        scattering_clique_algorithm,
                                        clipper,
                                        ant_colony_algorithm,
                                        mcp,
                                        genetic_algorithm)
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

data_paths = [
    "test_data/general_0.1.json",
    "test_data/general_0.2.json",
    "test_data/general_0.4.json",
]

directory_paths = [
    "CliqueAI/clique_algorithms/saved_graph/0.1",
    "CliqueAI/clique_algorithms/saved_graph/0.2",
    "CliqueAI/clique_algorithms/saved_graph/0.4",
]

def get_test_data(data_path: str) -> MaximumCliqueOfLambdaGraph:
    with open(data_path, "r") as f:
        data = json.load(f)
    synapse = MaximumCliqueOfLambdaGraph.model_validate(data)
    return synapse


def get_test_data_from_clq(data_path: str) -> MaximumCliqueOfLambdaGraph:
    number_of_nodes = None
    number_of_edges = None
    adjacency = []

    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue  # skip comments or empty lines
            if line.startswith("p edge"):
                # Example: p edge 100 200
                _, _, nodes, edges = line.split()
                number_of_nodes = int(nodes)
                number_of_edges = int(edges)
                # Initialize adjacency list for all nodes (1-based)
                for i in range(1, number_of_nodes + 1):
                    adjacency.append([])
            elif line.startswith("e"):
                _, u, v = line.split()
                u, v = int(u) - 1, int(v) - 1
                while len(adjacency) <= u:
                    adjacency.append([])
                adjacency[u].append(v)
                while len(adjacency) <= v:
                    adjacency.append([])
                adjacency[v].append(u)

    data = {
        "name": "MaximumCliqueOfLambdaGraph",
        "timeout": 30.0,
        "total_size": 0,
        "header_size": 0,
        "dendrite": {
            "status_code": None,
            "status_message": None,
            "process_time": None,
            "ip": None,
            "port": None,
            "version": None,
            "nonce": None,
            "uuid": None,
            "hotkey": None,
            "signature": None
        },
        "axon": {
            "status_code": None,
            "status_message": None,
            "process_time": None,
            "ip": None,
            "port": None,
            "version": None,
            "nonce": None,
            "uuid": None,
            "hotkey": None,
            "signature": None
        },
        "computed_body_hash": "",
        "uuid": "5ce9af7e-3cbb-422d-9768-ff139214dcfe",
        "label": "general",
        "number_of_nodes": number_of_nodes,
        "adjacency_list": adjacency,
        "maximum_clique": []
    }
    
    synapse = MaximumCliqueOfLambdaGraph.model_validate(data)
    return synapse


def check_clique(adjacency_list: list[list[int]], clique: list[int]) -> bool:
    clique_set = set(clique)
    for i in range(len(clique)):
        node = clique[i]
        neighbors = set(adjacency_list[node])
        if not clique_set.issubset(neighbors.union({node})):
            return False
    for v in range(len(adjacency_list)):
        if v in clique_set:
            continue
        if all(v in adjacency_list[node] for node in clique):
            return False
    return True


def run(algorithm, synapse: MaximumCliqueOfLambdaGraph):
    maximum_clique = algorithm(synapse.number_of_nodes, synapse.adjacency_list)
    clique_check = check_clique(synapse.adjacency_list, maximum_clique)
    if not clique_check:
        print("Invalid clique found by algorithm!")
    else:
        print(f"Clique size: {len(maximum_clique)}")
    return len(maximum_clique)

def main():
    difficulty_list = ["0.1", "0.2", "0.4"]
    difficulty = 1
    csv_filename = f"clipper-{difficulty_list[difficulty]}.csv"
    with open(csv_filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "node_count", "genetic_count", "aca_count", "gnn_count", "networkx_count"])

        for fname in os.listdir(directory_paths[difficulty]):
            if not fname.endswith(".clq"):
                continue
            fpath = os.path.join(directory_paths[difficulty], fname)
            synapse = get_test_data_from_clq(fpath)
            print(f"Testing data from {fpath} with {synapse.number_of_nodes} nodes")
            genetic_count = run(genetic_algorithm, synapse)
            # aca_count = run(ant_colony_algorithm, synapse)
            # gnn_count = run(scattering_clique_algorithm, synapse)
            # networkx_count = run(networkx_algorithm, synapse)
            clipper_count = run(clipper, synapse)
            print(f"G&C: {genetic_count}, {clipper_count}")
            writer.writerow([fname, synapse.number_of_nodes, genetic_count, clipper_count])
            csvfile.flush()
    
    # for data_path in data_paths:
    #     synapse = get_test_data(data_path)
    #     print(f"Testing data from {data_path} with {synapse.number_of_nodes} nodes")
    #     # put your algorithm here
    #     run(genetic_algorithm, synapse)
    #     # run(ant_colony_algorithm, synapse)
        


if __name__ == "__main__":
    main()

# [3, 9, 11, 13, 24, 30, 32, 37, 41, 42, 57, 63, 68, 70, 72, 76, 77, 80, 82]
# [1, 2, 8, 11, 19, 20, 26, 32, 41, 42, 44, 46, 51, 65, 70, 73, 75, 79, 81]