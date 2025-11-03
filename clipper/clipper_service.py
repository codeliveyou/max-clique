# clipper_service.py (Python 3.8 environment)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from build.bindings.python import clipperpy

class GraphRequest(BaseModel):
    num_nodes: int
    adjacency_list: List[List[int]]

app = FastAPI()

def maximum_clique_algorithm(num_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    M = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, neighbors in enumerate(adjacency_list):
        for j in neighbors:
            M[i, j] = 1.0
            M[j, i] = 1.0
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
    return list(result.nodes)  # Ensure it's a list for JSON serialization

@app.post("/max_clique")
def max_clique_endpoint(graph: GraphRequest):
    max_clique = maximum_clique_algorithm(graph.num_nodes, graph.adjacency_list)
    return {"max_clique": max_clique}

# uvicorn clipper_service:app --host 0.0.0.0 --port 8008
