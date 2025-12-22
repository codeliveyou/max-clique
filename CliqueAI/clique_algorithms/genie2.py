from random  import randint, random, sample
from itertools import combinations
from copy import deepcopy
from time import time 
from _thread import start_new_thread

global startTime, timeLimit, bestLength, bestNodes, setLocalBest, visited
startTime = time()
timeLimit = 29.3
bestLength, bestNodes = 0, set()
setLocalBest = []
visited ={}

combLimit = 15
numNodes = (100, 300, 500)
goodLimits = (200, 100, 50) # Control Parameter
deltas = (1, 1, 1)
sleeptime = 3.7

test = 2
rate = 990
graphSize = numNodes[test] - randint(0, 10)
goodLimit = goodLimits[test]
delta = deltas[test]

def geneRandomGraph(graphSize, rate):
    edges =[]
    for u in range(graphSize):
        for v in range(u+1, graphSize):
            if random() * 1000 <= rate:
                edges.append([u, v])
    return edges

def getMatrix(graphSize, edges):
    ans = [set() for _ in range(graphSize)]
    for edge in edges:
        ans[edge[0]].add(edge[1])
        ans[edge[1]].add(edge[0])
    return ans

def getLinkNodes(subNodes, graph):
    i, ans = 0, set()
    for link in graph:
        if subNodes <=link:
            ans.add(i)
        i += 1
    return ans

def addBestSet(subNodes, source):
    global setLocalBest, bestLength, bestNodes
    if subNodes not in setLocalBest:
        setLocalBest.append(subNodes)
        start_new_thread(additionalFind, (subNodes, graph, delta))
        if len(subNodes) > bestLength:
            bestLength = len(subNodes)
            bestNodes = subNodes
            # print( bestLength, i, source, check_clique(graph, list(bestNodes)), time() - startTime)
            return True
    return False

def localBest(subNodes, linkNodes, graph, source):
    global bestLength
    if not linkNodes:
        print("one best")
        return addBestSet(subNodes, source)
    for size in range(len(linkNodes), 0, -1):
        if size + len(subNodes) < bestLength:
            return False
        for nodes in combinations(linkNodes, size):
            if all(set(nodes) - {nodes} <= graph[node] for node in nodes):
                newNodes = subNodes.union(nodes)
                addBestSet(newNodes, source)
    return False

def mainFind(solutions, graph, goodLimit, source):
    while solutions:
        if time() - startTime > timeLimit:
            return
        # print(i , len(solutions[0][0]), len(solutions))
        maxLinkLen, buf = 0, []
        for solution in solutions:
            lenNode, lenLink = len(solution[0]), len (solution[1])
            if lenNode + lenLink >= bestLength:
                if lenLink > combLimit:
                    linkInfo = sorted([[node, solution[1].intersection(graph[node])]for node in solution[1]], key = lambda x:len(x[1]), reverse = True)
                    if len(linkInfo[0][1]) > maxLinkLen:
                        maxLinkLen = len(linkInfo[0][1])
                        buf = []
                    for link in linkInfo:
                        if time() - startTime > timeLimit:
                            return
                        if len(link[1]) == maxLinkLen:
                            try: 
                                newNodes = solution[0].union({link[0]})
                                if newNodes not in visited[maxLinkLen]:
                                    buf.append([newNodes, link[1]])
                                    visited[maxLinkLen].append(newNodes)
                            except KeyError:
                                buf.append([newNodes, link[1]])
                                visited[maxLinkLen] = [newNodes]
                        else:
                            break
                else:
                    localBest(solution[0], solution[1], graph, source)
        solutions = buf if len(buf) < goodLimit else sample(buf, goodLimit)

def additionalFind(subNodes, graph, delta):
    for i in range(l, delta + 1):
        for nodes in combinations(subNodes, len(subNodes) - i):
            if len(subNodes) < bestLength or time() - startTime > timeLimit:
                return
            nodes = set(nodes)
            link = getLinkNodes(nodes, graph)
            if len(nodes) + len(link) > bestLength:
                mainFind([[nodes, getLinkNodes(nodes, graph)]], graph, goodLimit, "thread")

def check_clique(adjucency_list, clique):
    clique_set = set(clique)
    for i in range(len(clique)):
        node = clique[i]
        neighbours = set(adjucency_list[node])
        if not clique_set.issubset(neighbours. union([node])):
            print("no connect")
            return False
    for v in range(len(adjucency_list)):
        if v in clique_set:
            continue
        if all(v in adjucency_list[node] for node in clique):
            print("sub net")
            return False
    return True

edges = geneRandomGraph(graphSize, rate)
graph = getMatrix(graphSize, edges)

population = sorted([[{node}, link] for node, link in enumerate(graph)], key = lambda x:len(x[1]), reverse = True)
i = 0
for solution in population:
    i += 1
    if time() - startTime > timeLimit:
        break
    mainFind([solution], graph, goodLimit, "main")
    # print(i, time() - startTime)
# print("localbest", len(setLocalBest), time() - startTime)
# print("best", check_clique(graph, list(bestNodes)), len(bestNodes))

while True:
    if time() - startTime > 29.6:
        print(bestNodes)
        break