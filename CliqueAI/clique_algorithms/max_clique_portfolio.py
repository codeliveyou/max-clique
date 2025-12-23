from __future__ import annotations

import os
import time
import multiprocessing as mp
from random import Random
from typing import List, Tuple

# Optional: your GA backend
try:
    import gaclique  # type: ignore
except Exception:
    gaclique = None


# ----------------------------
# Bitset helpers
# ----------------------------
def _bits_to_list(bits: int) -> List[int]:
    out = []
    while bits:
        lsb = bits & -bits
        out.append(lsb.bit_length() - 1)
        bits ^= lsb
    return out


def _bits_to_sorted_list(bits: int) -> List[int]:
    out = _bits_to_list(bits)
    out.sort()
    return out


def _make_maximal(adj: List[int], clique_bits: int) -> int:
    """Greedily extend clique to a maximal clique."""
    # candidates are common neighbors of clique vertices
    if clique_bits == 0:
        return 0
    cand = (1 << len(adj)) - 1
    tmp = clique_bits
    while tmp:
        lsb = tmp & -tmp
        v = lsb.bit_length() - 1
        cand &= adj[v]
        tmp ^= lsb
    cand &= ~clique_bits
    while cand:
        lsb = cand & -cand
        v = lsb.bit_length() - 1
        clique_bits |= lsb
        cand &= adj[v]
    return clique_bits


def _common_neighbors(adj: List[int], clique_bits: int, all_bits: int) -> int:
    cand = all_bits
    tmp = clique_bits
    while tmp:
        lsb = tmp & -tmp
        v = lsb.bit_length() - 1
        cand &= adj[v]
        tmp ^= lsb
    return cand & ~clique_bits


# ----------------------------
# Shared best clique (fast, no Manager)
# Store clique as 8x uint64 chunks (enough for n<=512)
# ----------------------------
def _read_shared_bits(shared_chunks) -> int:
    bits = 0
    for i in range(len(shared_chunks)):
        bits |= int(shared_chunks[i]) << (64 * i)
    return bits


def _write_shared_bits(shared_chunks, bits: int) -> None:
    for i in range(len(shared_chunks)):
        shared_chunks[i] = (bits >> (64 * i)) & ((1 << 64) - 1)


def _try_publish(best_size_val, shared_chunks, lock, clique_bits: int) -> None:
    s = clique_bits.bit_count()
    with lock:
        if s > best_size_val.value:
            best_size_val.value = s
            _write_shared_bits(shared_chunks, clique_bits)


# ----------------------------
# Worker 1: GA seed (gaclique) for ~4-6 seconds
# ----------------------------
def _ga_seed_worker(
    n: int,
    adjacency_list: List[List[int]],
    adj_bits: List[int],
    deadline: float,
    best_size_val,
    shared_chunks,
    lock,
) -> None:
    if gaclique is None:
        return

    # spend only a slice, not whole 30s
    seed_deadline = min(deadline, time.time() + 6.0)

    # write temp .clq in /tmp
    import tempfile
    import os as _os

    fd, path = tempfile.mkstemp(suffix=".clq", prefix="seed_", text=True)
    _os.close(fd)

    try:
        # count edges i<j
        m_edges = 0
        for i, neigh in enumerate(adjacency_list):
            for j in neigh:
                if j > i:
                    m_edges += 1

        with open(path, "w") as f:
            f.write(f"p edge {n} {m_edges}\n")
            for i, neigh in enumerate(adjacency_list):
                u = i + 1
                for j in neigh:
                    if j > i:
                        f.write(f"e {u} {j+1}\n")

        params_list = [
            (900, 20, 18, 2, 160, 10),
            (1100, 25, 20, 3, 180, 10),
            (1300, 30, 22, 4, 200, 10),
            (1100, 25, 20, 8, 180, 10),
            (1100, 25, 20, 12, 180, 10),
        ]

        all_bits = (1 << n) - 1
        idx = 0
        while time.time() < seed_deadline:
            params = params_list[idx % len(params_list)]
            idx += 1
            try:
                sol = gaclique.run_max_clique(path, *params)  # 1-based list
                if not sol:
                    continue
                b = 0
                for x in sol:
                    v = int(x) - 1
                    if 0 <= v < n:
                        b |= 1 << v

                # make maximal in the true graph (free improvement)
                b = _make_maximal(adj_bits, b)
                _try_publish(best_size_val, shared_chunks, lock, b)
            except Exception:
                continue
    finally:
        try:
            _os.remove(path)
        except Exception:
            pass


# ----------------------------
# Worker 2..12: DLS-MC style local search (incremental miss_count)
# This is the important "new method".
# ----------------------------
def _dls_mc_worker(
    wid: int,
    n: int,
    adj: List[int],
    deg: List[int],
    deadline: float,
    best_size_val,
    shared_chunks,
    lock,
) -> None:
    rng = Random(((int(time.time() * 1e6) & 0xFFFFFFFF) ^ (wid * 0x9E3779B1)) & 0xFFFFFFFF)
    all_bits = (1 << n) - 1

    # Tuning for dense graphs (your data is often 0.7~0.94)
    tabu_tenure = 10
    penalty_delay = 200
    sample_cap_add = 220   # score up to 220 add candidates
    sample_cap_swap = 320  # score up to 320 swap candidates

    penalties = [0] * n
    tabu_until = [0] * n
    step = 0

    # workspace arrays reused per restart
    miss = [0] * n  # number of clique vertices that are NOT neighbors of v
    in_clique = [False] * n

    def start_clique() -> int:
        # choose a start vertex biased by degree
        top = 160 if n >= 450 else 120
        # build top list once per restart (cheap at n=500)
        top_vertices = sorted(range(n), key=lambda x: deg[x], reverse=True)[:top]
        v0 = top_vertices[int((rng.random() ** 2) * len(top_vertices))]

        # build maximal clique greedily from v0 using add-candidates intersection
        C = 1 << v0
        P = adj[v0]
        while P:
            # pick best v in P by (neighbors inside P) - penalty
            # sample for speed
            candidates = []
            tmp = P
            for _ in range(sample_cap_add):
                if not tmp:
                    break
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                candidates.append(v)
                tmp ^= lsb
            if not candidates:
                break
            rng.shuffle(candidates)

            best_v = -1
            best_score = -10**18
            for v in candidates:
                sc = (adj[v] & P).bit_count() * 1000 - penalties[v] * 9 + rng.randrange(0, 13)
                if sc > best_score:
                    best_score = sc
                    best_v = v
            if best_v < 0:
                break
            C |= 1 << best_v
            P &= adj[best_v]

        return _make_maximal(adj, C)

    def init_state(C_bits: int) -> None:
        # reset membership
        for i in range(n):
            in_clique[i] = False
            miss[i] = 0

        C_list = _bits_to_list(C_bits)
        for v in C_list:
            in_clique[v] = True

        # compute miss[v] = number of clique vertices not connected to v
        # miss for clique members is 0 by definition (ignored)
        for v in range(n):
            if in_clique[v]:
                miss[v] = 0
            else:
                # count clique vertices not adjacent to v
                # miss = |C| - neighbors_in_C
                miss[v] = len(C_list) - (adj[v] & C_bits).bit_count()

    def add_vertex(C_bits: int, v: int) -> int:
        # add v to clique, update miss[] for outside vertices
        in_clique[v] = True
        v_bit = 1 << v
        C_bits |= v_bit

        # update miss for vertices not in clique:
        # For any u outside clique: if u is NOT neighbor of v -> miss[u]++
        not_neighbors = (~adj[v]) & ((1 << n) - 1)
        outside = (~C_bits) & ((1 << n) - 1)
        affected = outside & not_neighbors
        tmp = affected
        while tmp:
            lsb = tmp & -tmp
            u = lsb.bit_length() - 1
            miss[u] += 1
            tmp ^= lsb

        # v itself no longer outside
        miss[v] = 0
        return C_bits

    def remove_vertex(C_bits: int, v: int) -> int:
        # remove v from clique, update miss[] for outside vertices
        in_clique[v] = False
        v_bit = 1 << v
        C_bits &= ~v_bit

        # For any u outside clique: if u is NOT neighbor of v -> miss[u]-- (because clique got smaller)
        not_neighbors = (~adj[v]) & ((1 << n) - 1)
        outside = (~C_bits) & ((1 << n) - 1)
        affected = outside & not_neighbors
        tmp = affected
        while tmp:
            lsb = tmp & -tmp
            u = lsb.bit_length() - 1
            miss[u] -= 1
            tmp ^= lsb

        return C_bits

    def current_candidates_add(C_bits: int, C_size: int) -> int:
        # vertices with miss==0 and not in clique => can be added
        # (use bitset scan)
        cand = 0
        for v in range(n):
            if (not in_clique[v]) and miss[v] == 0:
                cand |= 1 << v
        return cand

    def current_candidates_swap(C_bits: int) -> List[Tuple[int, int]]:
        # swap candidates: miss==1; for each u we need to know which clique vertex is missing.
        # We'll find missing vertex by scanning clique members (clique size small-ish, ok).
        C_list = _bits_to_list(C_bits)
        out = []
        # build list of u with miss==1
        # sample later; keep cheap
        for u in range(n):
            if (not in_clique[u]) and miss[u] == 1:
                # find the unique clique vertex not adjacent to u
                # (the one to drop)
                for v in C_list:
                    if ((adj[u] >> v) & 1) == 0:
                        out.append((u, v))
                        break
        return out

    def pick_best_add(cand_bits: int) -> int:
        # score candidates; sample subset
        candidates = []
        tmp = cand_bits
        for _ in range(sample_cap_add):
            if not tmp:
                break
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            candidates.append(v)
            tmp ^= lsb
        if not candidates:
            return -1
        rng.shuffle(candidates)

        best_v = -1
        best_score = -10**18
        for v in candidates:
            if tabu_until[v] > step:
                continue
            # prefer high degree, low penalty
            sc = deg[v] * 1000 - penalties[v] * 10 + rng.randrange(0, 19)
            if sc > best_score:
                best_score = sc
                best_v = v
        if best_v == -1:
            best_v = candidates[0]
        return best_v

    def pick_best_swap(swaps: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not swaps:
            return -1, -1
        # sample swaps
        if len(swaps) > sample_cap_swap:
            rng.shuffle(swaps)
            swaps = swaps[:sample_cap_swap]

        best_u = -1
        best_drop = -1
        best_score = -10**18
        for u, drop in swaps:
            if tabu_until[u] > step:
                continue
            # prefer adding high degree u and dropping high-penalty vertex
            sc = deg[u] * 1000 - penalties[u] * 12 + penalties[drop] * 18 + rng.randrange(0, 19)
            if sc > best_score:
                best_score = sc
                best_u = u
                best_drop = drop

        if best_u == -1:
            best_u, best_drop = swaps[0]
        return best_u, best_drop

    # ---- main loop (restarts) ----
    best_local_bits = 0
    best_local_sz = 0

    while time.time() < deadline:
        C_bits = start_clique()
        C_bits = _make_maximal(adj, C_bits)
        C_size = C_bits.bit_count()

        init_state(C_bits)

        if C_size > best_local_sz:
            best_local_sz = C_size
            best_local_bits = C_bits
            _try_publish(best_size_val, shared_chunks, lock, C_bits)

        stagnation = 0

        while time.time() < deadline and stagnation < 6000:
            step += 1
            incumbent = best_size_val.value

            # quick pruning: if even adding all perfect candidates can't beat incumbent, restart
            # (cheap upper estimate: C_size + count(miss==0))
            # if you're already behind, restart sooner
            if C_size + 5 <= incumbent and stagnation > 900:
                break

            cand_add = current_candidates_add(C_bits, C_size)
            if cand_add:
                v = pick_best_add(cand_add)
                if v >= 0:
                    C_bits = add_vertex(C_bits, v)
                    C_bits = _make_maximal(adj, C_bits)
                    C_size = C_bits.bit_count()
                    init_state(C_bits)  # recompute miss fast enough at n=500; keeps correctness clean

                    if C_size > best_local_sz:
                        best_local_sz = C_size
                        best_local_bits = C_bits
                        _try_publish(best_size_val, shared_chunks, lock, C_bits)
                        stagnation = 0
                    else:
                        stagnation += 1
                    continue

            # plateau / swap move (miss==1)
            swaps = current_candidates_swap(C_bits)
            if swaps:
                u, drop = pick_best_swap(swaps)
                if u >= 0 and drop >= 0:
                    C_bits = remove_vertex(C_bits, drop)
                    tabu_until[drop] = step + tabu_tenure + rng.randrange(0, tabu_tenure)
                    C_bits = add_vertex(C_bits, u)
                    C_bits = _make_maximal(adj, C_bits)
                    C_size = C_bits.bit_count()
                    init_state(C_bits)

                    if C_size > best_local_sz:
                        best_local_sz = C_size
                        best_local_bits = C_bits
                        _try_publish(best_size_val, shared_chunks, lock, C_bits)
                        stagnation = 0
                    else:
                        stagnation += 2
                    continue

            # stuck: apply DLS penalties + kick
            C_list = _bits_to_list(C_bits)
            for v in C_list:
                penalties[v] += 1

            if step % penalty_delay == 0:
                # smooth penalties
                for i in range(n):
                    if penalties[i] > 0:
                        penalties[i] -= 1

            # kick: remove 1-3 vertices with highest penalties
            if C_list:
                C_list.sort(key=lambda x: penalties[x], reverse=True)
                k = 1 if rng.random() < 0.55 else (2 if rng.random() < 0.85 else 3)
                for j in range(min(k, len(C_list))):
                    v = C_list[j]
                    C_bits = remove_vertex(C_bits, v)
                    tabu_until[v] = step + tabu_tenure + rng.randrange(0, tabu_tenure)

                C_bits = _make_maximal(adj, C_bits)
                C_size = C_bits.bit_count()
                init_state(C_bits)

            stagnation += 15

    # publish local best at end
    _try_publish(best_size_val, shared_chunks, lock, best_local_bits)


# ----------------------------
# Public API
# ----------------------------
def max_clique_portfolio(number_of_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    """
    Portfolio designed to beat GA-only on dense graphs within 30s:
      - 1x GA seeder (gaclique) for a few seconds
      - 11x DLS-MC local search workers (strong plateau+penalty method)
    Uses fork on Linux; no pickling issues.
    """
    n = number_of_nodes
    if n <= 0:
        return []

    start = time.time()
    deadline = start + 29.3

    # Build symmetric adjacency bitsets
    adj = [0] * n
    for u in range(n):
        bits = 0
        if u < len(adjacency_list):
            for v in adjacency_list[u]:
                if 0 <= v < n and v != u:
                    bits |= 1 << v
        adj[u] = bits
    for u in range(n):
        tmp = adj[u]
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            adj[v] |= 1 << u
            tmp ^= lsb

    deg = [adj[i].bit_count() for i in range(n)]

    # multiprocessing context
    ctx = mp.get_context("fork" if os.name == "posix" else "spawn")

    best_size_val = ctx.Value("i", 0)
    lock = ctx.Lock()

    # chunks for n<=512: 8 x uint64
    shared_chunks = ctx.Array("Q", 8, lock=False)  # unsigned long long
    _write_shared_bits(shared_chunks, 0)

    # quick greedy seed in main process
    # start from top-degree vertex and greedily build maximal clique
    top = sorted(range(n), key=lambda x: deg[x], reverse=True)[:200]
    seed_bits = 0
    best_seed_sz = 0
    for v0 in top[: min(120, len(top))]:
        if time.time() > deadline:
            break
        C = 1 << v0
        P = adj[v0]
        while P:
            lsb = P & -P
            v = lsb.bit_length() - 1
            C |= lsb
            P &= adj[v]
        C = _make_maximal(adj, C)
        s = C.bit_count()
        if s > best_seed_sz:
            best_seed_sz = s
            seed_bits = C

    with lock:
        best_size_val.value = best_seed_sz
        _write_shared_bits(shared_chunks, seed_bits)

    # start workers
    procs = []
    total_cores = min(12, os.cpu_count() or 12)

    # 1 GA seeder if available, else all DLS
    if gaclique is not None and time.time() < deadline - 10:
        p = ctx.Process(
            target=_ga_seed_worker,
            args=(n, adjacency_list, adj, deadline, best_size_val, shared_chunks, lock),
        )
        p.daemon = True
        p.start()
        procs.append(p)
        dls_workers = total_cores - 1
    else:
        dls_workers = total_cores

    for wid in range(dls_workers):
        p = ctx.Process(
            target=_dls_mc_worker,
            args=(wid, n, adj, deg, deadline, best_size_val, shared_chunks, lock),
        )
        p.daemon = True
        p.start()
        procs.append(p)

    # wait until deadline
    for p in procs:
        timeout = max(0.0, deadline - time.time())
        p.join(timeout=timeout)

    best_bits = _read_shared_bits(shared_chunks)
    best_bits = _make_maximal(adj, best_bits)
    return _bits_to_sorted_list(best_bits)
