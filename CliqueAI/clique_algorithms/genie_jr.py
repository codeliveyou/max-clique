from time import time
from random import randrange, random
from typing import List, Tuple


def genie_clique_jr(number_of_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    """
    Hybrid maximum clique solver optimized for dense graphs (~500 nodes) under ~30s.

    - Phase 1: heavy multi-start greedy + iterated drop/regrow local search (fast lower bound)
    - Phase 2: exact branch & bound booster (BBMC/Tomita style) on a safely reduced candidate set
              (never returns worse than Phase 1)

    Signature:
        genie_clique(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]
    """

    n = number_of_nodes
    if n <= 0:
        return []

    start = time()
    time_limit = 29.3  # your harness uses 30; keep a little headroom

    # -------------------------
    # Build adjacency bitsets
    # -------------------------
    adj = [0] * n
    for u in range(n):
        m = 0
        if u < len(adjacency_list):
            for v in adjacency_list[u]:
                if 0 <= v < n and v != u:
                    m |= (1 << v)
        adj[u] = m

    ALL = (1 << n) - 1

    # degrees + ordering
    deg = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=deg.__getitem__, reverse=True)

    # -------------------------
    # Bit helpers
    # -------------------------
    def bits_iter(bits: int):
        while bits:
            lsb = bits & -bits
            yield lsb.bit_length() - 1
            bits ^= lsb

    def bits_to_list(bits: int) -> List[int]:
        out = []
        while bits:
            lsb = bits & -bits
            out.append(lsb.bit_length() - 1)
            bits ^= lsb
        out.sort()
        return out

    def cand_from_clique(clq_bits: int) -> int:
        """Candidates that connect to all vertices in clq_bits."""
        cand = ALL
        tmp = clq_bits
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            cand &= adj[v]
            tmp ^= lsb
        return cand & ~clq_bits

    # -------------------------
    # Best tracking
    # -------------------------
    best_bits = 0
    best_size = 0

    def update_best(clq_bits: int):
        nonlocal best_bits, best_size
        s = clq_bits.bit_count()
        if s > best_size:
            best_size = s
            best_bits = clq_bits

    # -------------------------
    # Phase 1: very strong heuristic
    # -------------------------
    # Pick vertex in candidate set: internal degree + noise (for exploration)
    def pick_vertex(cand_bits: int, scan_cap: int, noise: float) -> int:
        best_v = -1
        best_score = -1.0
        tmp = cand_bits
        scanned = 0

        # scan limited subset for speed
        while tmp and scanned < scan_cap:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            tmp ^= lsb
            scanned += 1

            score = (adj[v] & cand_bits).bit_count()
            score = score + noise * random()
            if score > best_score:
                best_score = score
                best_v = v

        return best_v

    def greedy_expand(clq_bits: int, cand_bits: int, scan_cap: int, noise: float) -> int:
        while cand_bits:
            v = pick_vertex(cand_bits, scan_cap, noise)
            if v < 0:
                break
            clq_bits |= (1 << v)
            cand_bits &= adj[v]
        return clq_bits

    # Local improvement: drop k nodes from current best and regrow
    def improve_drop_regrow(base_bits: int, drop_k: int, tries: int, scan_cap: int) -> int:
        nonlocal best_bits, best_size
        verts = [v for v in bits_iter(base_bits)]
        L = len(verts)
        if L <= drop_k:
            return base_bits

        for t in range(tries):
            if time() - start > time_limit * 0.88:
                break

            # choose drop_k vertices (biased but cheap and diverse)
            dropped = set()
            seed = (t * 2654435761) & 0xFFFFFFFF
            for i in range(drop_k):
                idx = (seed + i * 1640531527) % L
                dropped.add(verts[idx])

            new_bits = 0
            for v in verts:
                if v not in dropped:
                    new_bits |= (1 << v)

            cand = cand_from_clique(new_bits)
            grown = greedy_expand(new_bits, cand, scan_cap=scan_cap, noise=0.30)
            if grown.bit_count() > base_bits.bit_count():
                base_bits = grown
                update_best(base_bits)
                verts = [v for v in bits_iter(base_bits)]
                L = len(verts)
                if L <= drop_k:
                    break

        return base_bits

    # Heuristic tuning for dense graphs
    scan_cap = 260 if n >= 450 else 200

    # Seed with top-degree vertices (deterministic)
    for v in order[:min(n, 80)]:
        if time() - start > time_limit * 0.30:
            break
        clq = 1 << v
        clq = greedy_expand(clq, adj[v], scan_cap=scan_cap, noise=0.0)
        update_best(clq)

    # Many randomized restarts
    while time() - start < time_limit * 0.65:
        # pick among top-degree nodes (biased random)
        v = order[randrange(min(n, 220))]
        clq = 1 << v
        clq = greedy_expand(clq, adj[v], scan_cap=scan_cap, noise=0.20)
        update_best(clq)

        # quick small improvement sometimes
        if best_size >= 10 and random() < 0.15:
            _ = improve_drop_regrow(best_bits, drop_k=1, tries=6, scan_cap=scan_cap)

    # Strong local improvement loop
    current = best_bits
    while time() - start < time_limit * 0.85:
        before = current.bit_count()
        current = improve_drop_regrow(current, drop_k=1, tries=22, scan_cap=scan_cap)
        current = improve_drop_regrow(current, drop_k=2, tries=14, scan_cap=scan_cap)
        # occasional deeper shake
        if random() < 0.25:
            current = improve_drop_regrow(current, drop_k=3, tries=6, scan_cap=scan_cap)
        if current.bit_count() == before:
            # restart to escape plateau
            v = order[randrange(min(n, 240))]
            trial = greedy_expand(1 << v, adj[v], scan_cap=scan_cap, noise=0.35)
            update_best(trial)
            current = best_bits

    # -------------------------
    # Phase 2: Exact booster (BBMC/Tomita) with fast coloring + pivot
    # -------------------------
    # BBMC greedy coloring (bit-only), returns verts + colors (color is bound)
    def color_sort(Pbits: int) -> Tuple[List[int], List[int]]:
        verts = []
        colors = []
        color = 0
        U = Pbits
        while U:
            color += 1
            Q = U
            while Q:
                lsb = Q & -Q
                v = lsb.bit_length() - 1
                Q ^= lsb
                U &= ~lsb
                verts.append(v)
                colors.append(color)
                Q &= ~adj[v]  # keep independent set for this color
        return verts, colors

    def pick_pivot(Pbits: int) -> int:
        pivot = -1
        best = -1
        tmp = Pbits
        while tmp:
            lsb = tmp & -tmp
            u = lsb.bit_length() - 1
            tmp ^= lsb
            cnt = (Pbits & adj[u]).bit_count()
            if cnt > best:
                best = cnt
                pivot = u
        return pivot

    def expand(Rbits: int, Rsize: int, Pbits: int):
        nonlocal best_bits, best_size
        if time() - start > time_limit:
            return

        if not Pbits:
            if Rsize > best_size:
                best_size = Rsize
                best_bits = Rbits
            return

        # trivial bound
        if Rsize + Pbits.bit_count() <= best_size:
            return

        verts, cols = color_sort(Pbits)

        # pivot
        u = pick_pivot(Pbits)
        branch = Pbits & ~adj[u] if u >= 0 else Pbits

        # explore from end (higher color later) → raise best early
        for i in range(len(verts) - 1, -1, -1):
            if time() - start > time_limit:
                return

            v = verts[i]
            vb = 1 << v

            # coloring bound prune
            if Rsize + cols[i] <= best_size:
                return

            # pivot rule
            if not (branch & vb):
                continue

            newP = Pbits & adj[v]
            expand(Rbits | vb, Rsize + 1, newP)

            # remove v and continue
            Pbits &= ~vb
            branch &= ~vb

            if Rsize + Pbits.bit_count() <= best_size:
                return

    # Safe candidate reduction for improvement:
    # any vertex in a clique of size > best_size must have degree >= best_size
    # (actually >= best_size, since in clique size k each vertex degree >= k-1)
    remaining = time_limit - (time() - start)
    if remaining > 0.7:
        threshold = max(0, best_size - 1)
        eligible = 0
        for v in range(n):
            if deg[v] >= threshold:
                eligible |= (1 << v)

        # additionally restrict to vertices that are "promising": in dense graphs this helps
        # keep it safe-ish: still includes all vertices that could be in a larger clique
        expand(0, 0, eligible)

    return bits_to_list(best_bits)
