from __future__ import annotations

import os
import time
import multiprocessing as mp
from typing import List

import fastclique  # the pybind module you built


def _worker(n: int, adj: List[List[int]], tl: float, seed: int, q: mp.Queue) -> None:
    res = fastclique.run(n, adj, tl, seed)
    q.put(res)


def genie_clique_cpp(number_of_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    """
    8-core portfolio:
      run 8 independent C++ searches with different seeds in parallel
      return the best clique found within ~29.3s total wall clock.
    """
    n = number_of_nodes
    if n <= 0:
        return []

    # You said cpu_count is 8
    procs = 8
    ctx = mp.get_context("fork" if os.name == "posix" else "spawn")
    q: mp.Queue = ctx.Queue()

    start = time.time()
    time_limit = 29.3
    deadline = start + time_limit

    # Each process uses almost full time_limit (they run in parallel)
    # Keep a tiny safety margin for join/IPC
    per_proc_tl = max(1.0, time_limit - 0.3)

    ps = []
    base_seed = int(start * 1e6) & 0xFFFFFFFF
    for i in range(procs):
        seed = (base_seed + i * 1000003) & 0xFFFFFFFF
        p = ctx.Process(target=_worker, args=(n, adjacency_list, per_proc_tl, seed, q))
        p.daemon = True
        p.start()
        ps.append(p)

    for p in ps:
        p.join(timeout=max(0.0, deadline - time.time()))

    best = []
    best_len = -1
    while not q.empty():
        clq = q.get()
        if len(clq) > best_len:
            best_len = len(clq)
            best = clq

    return best
