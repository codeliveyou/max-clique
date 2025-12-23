from typing import List
import os

# Import the compiled module
from . import max_clique_cpp1 as _m

def max_clique_portfolio_cpp1(number_of_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    # Keep it aligned to your harness's 30s timeout
    # Use your actual CPU count (you said 8)
    return _m.max_clique_cpp1(number_of_nodes, adjacency_list, 29.5, 8)
