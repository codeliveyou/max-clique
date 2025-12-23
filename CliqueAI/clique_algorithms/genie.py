from itertools import combinations
from random import sample
from time import time
from typing import List, Set, Dict, Tuple, Optional


def genie_clique(number_of_nodes: int, adjacency_list: List[List[int]]) -> List[int]:
    """
    Heuristic maximum clique finder (based on your original code structure).

    Args:
        number_of_nodes: total number of nodes in the graph (0..n-1)
        adjacency_list: adjacency list where adjacency_list[u] is a list of neighbors of u

    Returns:
        A list of node indices representing the best clique found (bestNodes).
    """

    # -----------------------------
    # Reset "global" state per call
    # -----------------------------
    startTime = time()
    timeLimit = 29.3

    bestLength: int = 0
    bestNodes: Set[int] = set()
    setLocalBest: List[Set[int]] = []
    visited: Dict[int, List[Set[int]]] = {}

    # -----------------------------
    # Tunables (same idea as yours)
    # -----------------------------
    combLimit = 15
    # you can adjust these per n; keeping your original selection logic
    numNodes = (100, 300, 500)
    goodLimits = (200, 100, 50)  # Control Parameter
    deltas = (1, 1, 1)

    # choose bucket by size (simple mapping similar to your original "test")
    if number_of_nodes <= 120:
        test = 0
    elif number_of_nodes <= 380:
        test = 1
    else:
        test = 2

    goodLimit = goodLimits[test]
    delta = deltas[test]

    # ------------------------------------
    # Normalize adjacency to list[set[int]]
    # (makes subset/intersection checks fast)
    # ------------------------------------
    # We also clamp to number_of_nodes and remove self loops defensively.
    graph: List[Set[int]] = []
    for u in range(number_of_nodes):
        neigh = set(adjacency_list[u]) if u < len(adjacency_list) else set()
        neigh.discard(u)
        neigh = {v for v in neigh if 0 <= v < number_of_nodes}
        graph.append(neigh)

    # -----------------------------
    # Helper functions (closures)
    # -----------------------------
    def getLinkNodes(subNodes: Set[int]) -> Set[int]:
        """Return nodes that connect to ALL nodes in subNodes."""
        ans: Set[int] = set()
        for i, link in enumerate(graph):
            if subNodes <= link:
                ans.add(i)
        return ans

    def addBestSet(subNodes: Set[int], source: str) -> bool:
        nonlocal bestLength, bestNodes, setLocalBest
        if subNodes not in setLocalBest:
            setLocalBest.append(subNodes)
            if len(subNodes) > bestLength:
                bestLength = len(subNodes)
                bestNodes = subNodes
                return True
        return False

    def localBest(subNodes: Set[int], linkNodes: Set[int], source: str) -> bool:
        nonlocal bestLength
        if not linkNodes:
            return addBestSet(subNodes, source)

        linkNodes_list = list(linkNodes)
        for size in range(len(linkNodes_list), 0, -1):
            if size + len(subNodes) < bestLength:
                return False
            for nodes in combinations(linkNodes_list, size):
                # Check that nodes themselves form a clique
                if all(set(nodes) - {node} <= graph[node] for node in nodes):
                    newNodes = subNodes.union(nodes)
                    addBestSet(newNodes, source)
        return False

    def mainFind(solutions: List[Tuple[Set[int], Set[int]]], source: str) -> None:
        nonlocal visited, bestLength

        while solutions:
            if time() - startTime > timeLimit:
                return

            maxLinkLen = 0
            buf: List[Tuple[Set[int], Set[int]]] = []

            for subNodes, linkSet in solutions:
                lenNode = len(subNodes)
                lenLink = len(linkSet)

                if lenNode + lenLink < bestLength:
                    continue

                if lenLink > combLimit:
                    # For each candidate node in linkSet, compute its reduced link set
                    linkInfo = sorted(
                        [(node, linkSet.intersection(graph[node])) for node in linkSet],
                        key=lambda x: len(x[1]),
                        reverse=True,
                    )

                    if not linkInfo:
                        continue

                    top_len = len(linkInfo[0][1])
                    if top_len > maxLinkLen:
                        maxLinkLen = top_len
                        buf = []

                    for node, reduced_links in linkInfo:
                        if time() - startTime > timeLimit:
                            return
                        if len(reduced_links) != maxLinkLen:
                            break

                        newNodes = subNodes.union({node})

                        # visited buckets by maxLinkLen (same concept as your code)
                        if maxLinkLen not in visited:
                            visited[maxLinkLen] = [newNodes]
                            buf.append((newNodes, reduced_links))
                        else:
                            if newNodes not in visited[maxLinkLen]:
                                visited[maxLinkLen].append(newNodes)
                                buf.append((newNodes, reduced_links))
                else:
                    localBest(subNodes, linkSet, source)

            # reduce branching
            if not buf:
                return
            solutions = buf if len(buf) < goodLimit else sample(buf, goodLimit)

    def additionalFind(subNodes: Set[int]) -> None:
        nonlocal bestLength

        # try dropping 1..delta nodes from a found clique-candidate to explore neighbors
        for i in range(1, delta + 1):
            if time() - startTime > timeLimit:
                return
            if len(subNodes) - i <= 0:
                continue

            for nodes in combinations(subNodes, len(subNodes) - i):
                if len(subNodes) < bestLength or time() - startTime > timeLimit:
                    return
                nodes_set = set(nodes)
                link = getLinkNodes(nodes_set)
                if len(nodes_set) + len(link) > bestLength:
                    # Start a focused search from this reduced set
                    mainFind([(nodes_set, link)], "thread")

    # -----------------------------
    # Main run (same flow as yours)
    # -----------------------------
    population: List[Tuple[Set[int], Set[int]]] = sorted(
        [({node}, graph[node].copy()) for node in range(number_of_nodes)],
        key=lambda x: len(x[1]),
        reverse=True,
    )

    # for solution in population[:300]:
    for solution in population:
        if time() - startTime > timeLimit:
            break
        mainFind([solution], "main")

    setLocalBest.sort(key=lambda x: len(x))
    for sol in setLocalBest:
        if time() - startTime > timeLimit:
            break
        additionalFind(sol)

    return list(sorted(bestNodes))


# # -----------------------------
# # Example usage (remove in prod)
# # -----------------------------
# if __name__ == "__main__":
#     # Small demo graph: triangle (0,1,2) plus extra node 3 connected to 0 only
#     adj = [
#         [1, 2, 3],  # 0
#         [0, 2],     # 1
#         [0, 1],     # 2
#         [0],        # 3
#     ]
#     print(genie_clique(4, adj))  # likely [0,1,2]
