#include "ConwayTable.h"


// Generate a 0 or 1 randomly. 
// Seed is the current time, p1 is probability of choosing 1.

int rollCellState(double p1)
{    
    double random_variable = static_cast<double>(std::rand()) / RAND_MAX;
    if (random_variable > p1) { return 0; }
    else return 1;
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