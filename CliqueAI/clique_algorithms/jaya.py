from random import randint, sample
from itertools import combinations
from copy import deepcopy
from collections import deque
import numpy as np 
import time, sys

global visited, bestLength, startTime, timeLimit, endedProgram, result
visited = {}
bestLength = 0
startTime = time.time()
timeLimit = 29
endedProgram = False
result = []

popuSizes = (300, 500, 700)
soluSizes = (25, 26, 26) # (25, 26, 26)
decRates = (3, 3.7, 6.5)
soluLimits = (13, 16, 17) # (16, 18, 19)
fitnessMAX = 1E5
fitnessLimits = (0.97, 0.975, 0.975)
combDeltas = (4, 4, 4)
iterations = (200, 200, 200)

def geneRandomGraph(graphSize):
    edges = []
    for u in range(graphSize):
        for v in range(u + 1, graphSize):
            if randint(0, 100) <= 80:
                edges.append([u, v])
    return edges

def getMatrix(graphSize, edges):
    ans = [set() for _ in range(graphSize)]
    for edge in edges:
        ans[edge[0]].add(edge[1])
        ans[edge[1]].add(edge[0])
    return ans

def getFitness(subNodes, graph):
    n = len(subNodes)
    ans = sum(len(subNodes.intersection(graph[node])) for node in subNodes)
    ans /= n * (n - 1)
    return int(ans * fitnessMAX)

def genePopulation(popuSize, soluSize, graph):
    ans = []
    graphSize = len(graph)
    for _ in range(popuSize):
        nodes = set()
        while len(nodes) < soluSize:
            nodes.add(randint(0, len(graph) - 1))
        fitness = getFitness(nodes, graph)
        ans.append([fitness, nodes])
    ans.sort(key = lambda x : x[0])
    return ans

def jaya(population, soluLimit, decRate, graph):
    i = 0
    for solution in population:
        n = len(solution[1])
        n = max(soluLimit, n - int(abs(np.random.randn() * decRate)))
        newNodes = solution[1].union(population[0][1], population[-1][1])
        newNodes = set(sample(list(newNodes), k = n))
        fitness = getFitness(newNodes, graph)
        if fitness > solution[0]:
            population[i] = [fitness, newNodes]
        # i += 1
    return sorted(population, key = lambda x : x[0])

def runJaya(population, soluLimit, decRate, iteration, fitnessLimit, graph):
    global starTime, timeLimit, endedProgram, result
    for i in range(iteration):
        if endedProgram:
            return (population[-1][1])
        population = jaya(population, soluLimit, decRate, graph)
        if population[-1][0] >= fitnessLimit:
            # print(i, population[-1][0], population[0][0], len(population[01][1])), len(population[0][1]))
            return (population[-1][1])
        if time.time() - startTime > timeLimit:
            print("Jaya Stop")
            if not endedProgram:
                endedProgram = True
                if len(visited):
                    result = list(visited[bestLength][0])
            return (population[-1][1])
    return set()

def getConnGraphs(subNodes, combDelta, graph):
    ans = deque()
    if not len(subNodes): return ans
    for nodes in combinations(subNodes, len(subNodes) - combDelta):
        if getFitness(set(nodes), graph) == fitnessMAX:
            ans.append(set(nodes))
    return ans

def getBest(conGraphs, graph):
    global visited, bestLength, startTime, timeLimit, endedProgram, result
    if not conGraphs:
        return
    
    n = len(conGraphs[0])
    
    # Process initial conGraphs safely
    while conGraphs and endedProgram == False:
        nodes = conGraphs.popleft()  # Use popleft() for efficiency
        
        # Add to visited
        if n in visited:
            if nodes not in visited[n]:
                visited[n].append(nodes)
        else:
            visited[n] = [nodes]
            bestLength = max(bestLength, n)
        
        # Generate new nodes by adding neighbors
        node = 0
        for link in graph[node:]:  # Optimize: only check relevant neighbors
            if nodes.issubset(link):  # Use issubset() instead of <=
                newNodes = nodes | {node}  # Cleaner set union
                new_n = len(newNodes)
                
                if new_n in visited:
                    if newNodes not in visited[new_n]:
                        visited[new_n].append(newNodes)
                        conGraphs.append(newNodes)
                else:
                    visited[new_n] = [newNodes]
                    conGraphs.append(newNodes)
                    bestLength = max(bestLength, new_n)
            
            node += 1
            
            if time.time() - startTime > timeLimit:
                endProgram()
                return
        
        if time.time() - startTime > timeLimit:
            endProgram()
            return


def endProgram():
    global visited, bestLength, endedProgram, result
    if not endedProgram:
        endedProgram = True
        # for x in visited.keys():
        #     print(x, len(visited[x]))
        if len(visited):
            result = list(visited[bestLength][0])



def jaya_main(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
    test = 0
    if number_of_nodes <= 100:
        test = 0
    elif number_of_nodes <= 300:
        test = 1
    else:
        test = 2
    graphSize = number_of_nodes
    popuSize = popuSizes[test]
    soluSize = soluSizes[test]
    decRate = decRates[test]
    soluLimit = soluLimits[test]
    fitnessLimit = int(fitnessLimits[test] * fitnessMAX)
    combDelta = combDeltas[test]
    iteration = iterations[test]

    global visited, bestLength, startTime, timeLimit, endedProgram, result
    visited = {}
    bestLength = 0
    startTime = time.time()
    timeLimit = 29
    endedProgram = False
    result = [0]

    i = 0
    while not endedProgram:
        i += 1
        # print("ppppp", i)
        population = genePopulation(popuSize, soluSize, adjacency_list)
        bestNodes = runJaya(population, soluLimit, decRate, iteration, fitnessLimit, adjacency_list)
        conGraphs = getConnGraphs(bestNodes, combDelta, adjacency_list)
        # print(len(conGraphs))
        if len(conGraphs):
            getBest(conGraphs, adjacency_list)
    print("Result: ", result)
    return result
