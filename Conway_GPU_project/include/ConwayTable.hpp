#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

// Generate a 0 or 1 randomly. 
// Seed is the current time, p1 is probability of choosing 1.


std::vector<char> fillVector(double p1, int N)
{
    std::vector<char> grid(N * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d({ 1 - p1, 1 });

    std::generate(grid.begin(), grid.end(), [&] { return d(gen); });
    
    return grid;
}

void writeGrid(std::vector<char>& A, int N, std::ofstream& o)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            o << static_cast<int>(A[i * N + j]) << " ";
        }
        o << "\n";
    }
    o << "\n";
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