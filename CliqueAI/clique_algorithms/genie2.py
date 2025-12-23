# from random  import randint, random, sample
# from itertools import combinations
# from copy import deepcopy
# from time import time 
# from _thread import start_new_thread

# global startTime, timeLimit, bestLength, bestNodes, setLocalBest, visited
# startTime = time()
# timeLimit = 29.3
# bestLength, bestNodes = 0, set()
# setLocalBest = []
# visited ={}

# combLimit = 15
# numNodes = (100, 300, 500)
# goodLimits = (200, 100, 50) # Control Parameter
# deltas = (1, 1, 1)
# sleeptime = 3.7

# test = 2
# rate = 800
# graphSize = numNodes[test] - randint(0, 10)
# goodLimit = goodLimits[test]
# delta = deltas[test]

# def geneRandomGraph(graphSize, rate):
#     edges =[]
#     for u in range(graphSize):
#         for v in range(u+1, graphSize):
#             if random() * 1000 <= rate:
#                 edges.append([u, v])
#     return edges

# def getMatrix(graphSize, edges):
#     ans = [set() for _ in range(graphSize)]
#     for edge in edges:
#         ans[edge[0]].add(edge[1])
#         ans[edge[1]].add(edge[0])
#     return ans

# def getLinkNodes(subNodes, graph):
#     i, ans = 0, set()
#     for link in graph:
#         if subNodes <=link:
#             ans.add(i)
#         i += 1
#     return ans

# def addBestSet(subNodes, source):
#     global setLocalBest, bestLength, bestNodes
#     if subNodes not in setLocalBest:
#         setLocalBest.append(subNodes)
#         start_new_thread(additionalFind, (subNodes, graph, delta))
#         if len(subNodes) > bestLength:
#             bestLength = len(subNodes)
#             bestNodes = subNodes
#             # print( bestLength, i, source, check_clique(graph, list(bestNodes)), time() - startTime)
#             return True
#     return False

# def localBest(subNodes, linkNodes, graph, source):
#     global bestLength
#     if not linkNodes:
#         print("one best")
#         return addBestSet(subNodes, source)
#     for size in range(len(linkNodes), 0, -1):
#         if size + len(subNodes) < bestLength:
#             return False
#         for nodes in combinations(linkNodes, size):
#             if all(set(nodes) - {node} <= graph[node] for node in nodes):
#                 newNodes = subNodes.union(nodes)
#                 addBestSet(newNodes, source)
#     return False

# def mainFind(solutions, graph, goodLimit, source):
#     while solutions:
#         if time() - startTime > timeLimit:
#             return
#         # print(i , len(solutions[0][0]), len(solutions))
#         maxLinkLen, buf = 0, []
#         for solution in solutions:
#             lenNode, lenLink = len(solution[0]), len (solution[1])
#             if lenNode + lenLink >= bestLength:
#                 if lenLink > combLimit:
#                     linkInfo = sorted([[node, solution[1].intersection(graph[node])]for node in solution[1]], key = lambda x:len(x[1]), reverse = True)
#                     if len(linkInfo[0][1]) > maxLinkLen:
#                         maxLinkLen = len(linkInfo[0][1])
#                         buf = []
#                     for link in linkInfo:
#                         if time() - startTime > timeLimit:
#                             return
#                         if len(link[1]) == maxLinkLen:
#                             try: 
#                                 newNodes = solution[0].union({link[0]})
#                                 if newNodes not in visited[maxLinkLen]:
#                                     buf.append([newNodes, link[1]])
#                                     visited[maxLinkLen].append(newNodes)
#                             except KeyError:
#                                 buf.append([newNodes, link[1]])
#                                 visited[maxLinkLen] = [newNodes]
#                         else:
#                             break
#                 else:
#                     localBest(solution[0], solution[1], graph, source)
#         solutions = buf if len(buf) < goodLimit else sample(buf, goodLimit)

# def additionalFind(subNodes, graph, delta):
#     for i in range(1, delta + 1):
#         for nodes in combinations(subNodes, len(subNodes) - i):
#             if len(subNodes) < bestLength or time() - startTime > timeLimit:
#                 return
#             nodes = set(nodes)
#             link = getLinkNodes(nodes, graph)
#             if len(nodes) + len(link) > bestLength:
#                 mainFind([[nodes, getLinkNodes(nodes, graph)]], graph, goodLimit, "thread")

# def check_clique(adjucency_list, clique):
#     clique_set = set(clique)
#     for i in range(len(clique)):
#         node = clique[i]
#         neighbours = set(adjucency_list[node])
#         if not clique_set.issubset(neighbours. union([node])):
#             print("no connect")
#             return False
#     for v in range(len(adjucency_list)):
#         if v in clique_set:
#             continue
#         if all(v in adjucency_list[node] for node in clique):
#             print("sub net")
#             return False
#     return True

# edges = geneRandomGraph(graphSize, rate)
# graph = getMatrix(graphSize, edges)

# population = sorted([[{node}, link] for node, link in enumerate(graph)], key = lambda x:len(x[1]), reverse = True)
# i = 0
# for solution in population:
#     i += 1
#     if time() - startTime > timeLimit:
#         break
#     mainFind([solution], graph, goodLimit, "main")
#     # print(i, time() - startTime)
# # print("localbest", len(setLocalBest), time() - startTime)
# # print("best", check_clique(graph, list(bestNodes)), len(bestNodes))

# # while True:
# #     if time() - startTime > 29.6:
# print("Result: ", time() - startTime, list(bestNodes))
#         # break

from time import time
from typing import List

def genie_clique(number_of_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    """
    Maximum Clique (exact) - optimized pure Python bitset solver.
    BBMC/Tomita style: greedy coloring upper bound + pivot branching.

    Args:
        number_of_nodes: int
        adjacency_list: list[list[int]] (0-based neighbors)
    Returns:
        list[int] maximum clique found (sorted)
    """
    n = number_of_nodes
    if n <= 0:
        return []

    start = time()
    time_limit = 29.3  # align with your 30s harness

    # ---- build adjacency bitsets ----
    adj = [0] * n
    for u in range(n):
        m = 0
        if u < len(adjacency_list):
            for v in adjacency_list[u]:
                if 0 <= v < n and v != u:
                    m |= (1 << v)
        adj[u] = m

    # If your input can be non-symmetric, uncomment to symmetrize:
    # for u in range(n):
    #     bits = adj[u]
    #     while bits:
    #         lsb = bits & -bits
    #         v = lsb.bit_length() - 1
    #         adj[v] |= (1 << u)
    #         bits ^= lsb

    # ---- degrees + initial ordering ----
    deg = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=deg.__getitem__, reverse=True)

    # ---- greedy seed clique to set a strong lower bound ----
    best_bits = 0
    best_size = 0
    P = (1 << n) - 1
    seed_bits = 0
    for v in order:
        vb = 1 << v
        if P & vb:
            seed_bits |= vb
            P &= adj[v]
    best_bits = seed_bits
    best_size = seed_bits.bit_count()

    # ---- helpers ----
    def pick_pivot(Pbits: int) -> int:
        """Choose pivot u in P maximizing |P ∩ N(u)| (scan bits in P)."""
        # In dense graphs, pivoting helps a lot.
        max_cnt = -1
        pivot = -1
        tmp = Pbits
        while tmp:
            lsb = tmp & -tmp
            u = lsb.bit_length() - 1
            cnt = (Pbits & adj[u]).bit_count()
            if cnt > max_cnt:
                max_cnt = cnt
                pivot = u
            tmp ^= lsb
        return pivot

    def color_bound(Pbits: int):
        """
        Greedy coloring using only bit ops.
        Returns (vertices, bounds) where bounds[i] is an upper bound for clique extension
        from vertices[0..i] (BBMC style used in reverse iteration).
        """
        verts = []
        bounds = []
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
                bounds.append(color)
                Q &= ~adj[v]
        return verts, bounds

    # ---- main branch & bound ----
    def expand(Rbits: int, Rsize: int, Pbits: int):
        nonlocal best_bits, best_size

        # timeout
        if time() - start > time_limit:
            return

        if not Pbits:
            if Rsize > best_size:
                best_size = Rsize
                best_bits = Rbits
            return

        # coloring upper bound
        verts, bounds = color_bound(Pbits)

        # pivot: branch only on vertices not adjacent to pivot
        u = pick_pivot(Pbits)
        if u >= 0:
            branch_bits = Pbits & ~adj[u]
        else:
            branch_bits = Pbits

        # iterate candidates (prefer last colored / higher bound in reverse)
        # We want to explore promising vertices first to raise best_size early.
        i = len(verts) - 1
        while i >= 0:
            v = verts[i]
            vb = 1 << v

            # prune by coloring bound
            if Rsize + bounds[i] <= best_size:
                return

            # only branch if v is in branch_bits; otherwise skip (pivot rule)
            if branch_bits & vb:
                newRbits = Rbits | vb
                newPbits = Pbits & adj[v]
                expand(newRbits, Rsize + 1, newPbits)

                # remove v from P
                Pbits &= ~vb
                if not Pbits:
                    break

                # also update branch_bits (since P changed)
                branch_bits &= ~vb

                # quick prune: even if we take all remaining P, cannot beat best
                if Rsize + Pbits.bit_count() <= best_size:
                    return

            i -= 1

    # Start from all vertices
    all_bits = (1 << n) - 1
    expand(0, 0, all_bits)

    # convert best_bits -> list
    res = []
    bits = best_bits
    while bits:
        lsb = bits & -bits
        res.append(lsb.bit_length() - 1)
        bits ^= lsb
    res.sort()
    return res
