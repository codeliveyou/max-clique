#pragma once
#include <vector>
#include <string>
std::vector<int> run_max_clique(
    const std::string& filename, int generations, int populationNum,
    int localImprovementNum, int mutations, int uniqueIterations, int shuffleTolerance);
