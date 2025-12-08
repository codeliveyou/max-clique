from random import randint, sample, random
from itertools import combinations
from copy import deepcopy
from collections import deque
import numpy as np 
import time, sys


numNodes = (100, 300, 500)
testLens = (70, 150, 200)
regressLimits = (63152.41, 80182.545, 63152.41)
regressAs = [(-3.02419774, -4.42951187e-04, 8.41150674e-09, 1.19008953e-13),
             (-2.25157184, -5.41284409e-04, 1.05787555e-08, 3.23290275e-13),
             (-0.78602445, -2.66241391e-04, 7.50477413e-09, 5.45673319e-13)]
popuSizes = (30, 30, 30)
soluLimits = (16, 18, 23)
fitnessMAX = 1E5
fitnessLimits = (1, 0.9999, 0.99993)
combDeltas = (0, 1, 1)
iterations = (70, 70, 70)

rate = 992

def geneRandomGraph(graphSize, rate):
    edges = []
    for u in range(graphSize):
        for v in range(u + 1, graphSize):
            if random() * 1000 <= rate:
                edges.append([u, v])
    return edges

def getMatrix(graphSize, edges):
    ans = [[] for _ in range(graphSize)]
    for edge in edges:
        ans[edge[0]].append(edge[1])
        ans[edge[1]].append(edge[0])
    return ans

def getFitness(subNodes, graph):
    n = len(subNodes)
    ans = sum(len(subNodes.intersection(graph[node])) for node in subNodes)
    ans /= n * (n - 1)
    return int(ans * fitnessMAX)

def genePopulation(nodes, popuSize, soluSize, graph):
    ans = []
    for _ in range(popuSize):
        subNodes = set(sample(nodes, soluSize))
        fitness = getFitness(subNodes, graph)
        ans.append([fitness, subNodes])
    ans.sort(key = lambda x : x[0])
    return ans

def jaya(population, soluLimit, graph):
    i = 0
    for solution in population:
        newNodes = set(sample(solution[1].union(population[0][1]), k = soluLimit))
        newNodes = set(sample(solution[1].union(population[-1][1]), k = soluLimit))
        fitness = getFitness(newNodes, graph)
        if fitness > solution[0]:
            population[i] = [fitness, newNodes]
        i += 1
    return sorted(population, key = lambda x : x[0])

def runJaya(population, soluLimit, iteration, fitnessLimit, graph):
    global starTime, timeLimit, endFlag
    for i in range(iteration):
        population = jaya(population, soluLimit, graph)
        if population[-1][0] >= fitnessLimit:
            # print(i, population[-1][0], population[0][0], len(population[01][1])), len(population[0][1]))
            return population[-1][1]
        if time.time() - startTime > timeLimit:
            # print("Jaya Stop")
            endFlag = True
            return population[-1][1]
    return population[-1][1]

def getGoodNodes(subNodes, graph):
    ans = deepcopy(subNodes)
    i = 0
    for link in graph:
        if subNodes <= set(link):
            ans.add(i)
        i += 1
    return ans

def getConnGraphs(subNodes, combDelta, goodNodes, findFirst, graph):
    global startTime, timeLimit, endFlag
    lenNodes = len(subNodes)
    if not lenNodes: return 0
    bestnum = 0
    subSize = max(lenNodes - combDelta, 2)
    for nodes in combinations(subNodes, subSize):
        if time.time() - startTime > timeLimit:
            endFlag = True
            return 0
        if getFitness(set(nodes), graph) == fitnessMAX:
            if not findFirst or (findFirst and not bestnum):
                getBest(deque([set(nodes),]), goodNodes, findFirst, graph)
            bestnum += 1
    return bestnum

def getBest(connGraphs, goodNodes, findFirst, graph):
    global visited, bestLength, startTime, timeLimit, endFlag
    n = len(connGraphs[0])
    for nodes in connGraphs:
        if n in visited:
            if nodes not in visited[n]:
                visited[n].append(nodes)
        else:
            visited[n] = [nodes, ]
            bestLength = max(bestLength, n)
    while len(connGraphs):
        nodes = connGraphs.pop()
        if findFirst and soluLimit < bestLength and len(nodes) < bestLength:
            return
        if len(nodes) < bestLength and time.time() - startTime > timeLimit:
            endFlag = True
            return
        selNodes = goodNodes.difference(nodes)
        for node in selNodes:
            if nodes <= set(graph[node]):
                newNodes = deepcopy(nodes)
                newNodes.add(node)
                n = len(newNodes)
                if n in visited:
                    if newNodes not in visited[n]:
                        visited[n].append(newNodes)
                        connGraphs.append(newNodes)
                else:
                    visited[n] = [newNodes, ]
                    connGraphs.append(newNodes)
                    bestLength = max(bestLength, n)
            node += 1

def endProgram():
    global visited, bestLength
    if len(visited):
        bestNodes = visited[bestLength][0]
    else:
        bestNodes = {0}
    print(bestNodes)
    return

def check_clique(adjucency_list, clique):
    clique_set = set(clique)
    for i in range(len(clique)):
        node = clique[i]
        neighbours = set(adjucency_list[node])
        if not clique_set.issubset(neighbours.union([node])):
            print("No connect")
            return False
    for v in range(len(adjucency_list)):
        if v in clique_set:
            continue
        if all(v in adjucency_list[node] for node in clique):
            print("sub net")
            return False
    return True

# global visited, bestLength, startTime, timeLimit, endFlag
# global graphSize, testLen, popuSize, regressLimit, regressA, soluLimit, fitnessLimit, combDelta, iteration

# def jaya_new_main(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
#     test = 0
#     if number_of_nodes <= 100:
#         test = 0
#     elif number_of_nodes <= 300:
#         test = 1
#     else:
#         test = 2
    
#     global graphSize, testLen, popuSize, regressLimit, regressA, soluLimit, fitnessLimit, combDelta, iteration
#     graphSize = number_of_nodes
#     testLen = testLens[test]
#     popuSize = popuSizes[test]
#     regressLimit = regressLimits[test]
#     regressA = regressAs[test]
#     soluLimit = soluLimits[test]
#     fitnessLimit = int(fitnessLimits[test] * fitnessMAX)
#     combDelta = combDeltas[test]
#     iteration = iterations[test]

#     global visited, bestLength, startTime, timeLimit, endFlag
#     visited = {}
#     bestLength = 0
#     startTime = time.time()
#     timeLimit = 29
    
#     edges = geneRandomGraph(graphSize, rate)
#     graph = getMatrix(graphSize, edges)
#     allNodes = {n for n in range(len(graph))}
#     population = genePopulation(allNodes, 200, testLen, graph)
#     fitness = sum(x[0] for x in population) / 200
#     if fitness > regressLimit:
#         soluLimit = round(regressA[0] + regressA[1] * fitness + regressA[2] * (fitness)**2 + regressA[3] * np.e**((fitness) / 3000))
#     else:
#         soluLimit = 2 + int(7 * fitness / regressLimit)
    
#     findFirst = True
#     endFlag = False
#     cnt = 0
#     while not endFlag:
#         cnt += 1
#         itersize = 12
#         soluLimit = max(bestLength, soluLimit)
#         goodNodes = {n for n in range(len(graph))}
#         for it in range(4, itersize):
#             solulen = max(soluLimit * it // itersize, 2)
#             population = genePopulation(goodNodes, popuSize, solulen, graph)
#             goodNodes = runJaya(population, solulen, iteration, fitnessMAX, graph)
#             goodNodes = getGoodNodes(goodNodes, graph)
#             if len(goodNodes) < soluLimit:
#                 break
#         if len(goodNodes) < soluLimit:
#             continue
#         population = genePopulation(goodNodes, popuSize, soluLimit, graph)
#         betterNodes = runJaya(population, soluLimit, iteration, fitnessLimit, graph)
#         bestNum = getConnGraphs(betterNodes, combDelta, goodNodes, findFirst, graph)
#         if findFirst and bestNum:
#             findFirst = False
#             itersize = 5
#             soluLimit1 = max(bestLength, soluLimit)
#             for it in range(1, itersize):
#                 solulen = max(soluLimit + (soluLimit1 - soluLimit) * it // itersize, 2)
#                 population = genePopulation(goodNodes, popuSize, solulen, graph)
#                 goodNodes = runJaya(population, solulen, iteration, fitnessMAX, graph)
#                 goodNodes = getGoodNodes(goodNodes, graph)
#                 if len(goodNodes) < soluLimit1:
#                     break
#             if len(goodNodes) < soluLimit1:
#                 continue
#             population = genePopulation(goodNodes, popuSize, bestLength, graph)
#             betterNodes = runJaya(population, bestLength, iteration, fitnessLimit, graph)
#             bestNum = getConnGraphs(betterNodes, combDelta, goodNodes, findFirst, graph)
#         if endFlag:
#             endProgram(graph)



graphSizes = [nn - randint(0, 10) for nn in numNodes]
test = 2

global graphSize, testLen, popuSize, regressLimit, regressA, soluLimit, fitnessLimit, combDelta, iteration
graphSize = graphSizes[test]
testLen = testLens[test]
popuSize = popuSizes[test]
regressLimit = regressLimits[test]
regressA = regressAs[test]
soluLimit = soluLimits[test]
fitnessLimit = int(fitnessLimits[test] * fitnessMAX)
combDelta = combDeltas[test]
iteration = iterations[test]

global visited, bestLength, startTime, timeLimit, endFlag
visited = {}
bestLength = 0
startTime = time.time()
timeLimit = 29

edges = geneRandomGraph(graphSize, rate)
graph = getMatrix(graphSize, edges)
allNodes = {n for n in range(len(graph))}
population = genePopulation(allNodes, 200, testLen, graph)
fitness = sum(x[0] for x in population) / 200
if fitness > regressLimit:
    soluLimit = round(regressA[0] + regressA[1] * fitness + regressA[2] * (fitness)**2 + regressA[3] * np.e**((fitness) / 3000))
else:
    soluLimit = 2 + int(7 * fitness / regressLimit)

findFirst = True
endFlag = False
cnt = 0
while not endFlag:
    cnt += 1
    itersize = 12
    soluLimit = max(bestLength, soluLimit)
    goodNodes = {n for n in range(len(graph))}
    for it in range(4, itersize):
        solulen = max(soluLimit * it // itersize, 2)
        population = genePopulation(goodNodes, popuSize, solulen, graph)
        goodNodes = runJaya(population, solulen, iteration, fitnessMAX, graph)
        goodNodes = getGoodNodes(goodNodes, graph)
        if len(goodNodes) < soluLimit:
            break
    if len(goodNodes) < soluLimit:
        continue
    population = genePopulation(goodNodes, popuSize, soluLimit, graph)
    betterNodes = runJaya(population, soluLimit, iteration, fitnessLimit, graph)
    bestNum = getConnGraphs(betterNodes, combDelta, goodNodes, findFirst, graph)
    if findFirst and bestNum:
        findFirst = False
        itersize = 5
        soluLimit1 = max(bestLength, soluLimit)
        for it in range(1, itersize):
            solulen = max(soluLimit + (soluLimit1 - soluLimit) * it // itersize, 2)
            population = genePopulation(goodNodes, popuSize, solulen, graph)
            goodNodes = runJaya(population, solulen, iteration, fitnessMAX, graph)
            goodNodes = getGoodNodes(goodNodes, graph)
            if len(goodNodes) < soluLimit1:
                break
        if len(goodNodes) < soluLimit1:
            continue
        population = genePopulation(goodNodes, popuSize, bestLength, graph)
        betterNodes = runJaya(population, bestLength, iteration, fitnessLimit, graph)
        bestNum = getConnGraphs(betterNodes, combDelta, goodNodes, findFirst, graph)
    if endFlag:
        endProgram(graph)