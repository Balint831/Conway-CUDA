#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

// Generate a 0 or 1 randomly. 
// Seed is the current time, p1 is probability of choosing 1.
char rollCellState(double p1)
{
    double random_variable = static_cast<double>(std::rand()) / RAND_MAX;
    if (random_variable > p1) { return 0; }
    else return 1;
}

std::vector<char> fillVector(double p1, int N)
{
    std::vector<char> grid;
    for (int i = 0; i < N * N; i++)
    {
        grid.push_back(rollCellState(p1));
    }
    return grid;
}

void printGrid(std::vector<char>& A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << static_cast<int>(A[i * N + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}