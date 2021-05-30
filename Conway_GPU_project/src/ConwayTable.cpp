#include "ConwayTable.h"


// Generate a 0 or 1 randomly. 
// Seed is the current time, p1 is probability of choosing 1.

int rollCellState(double p1)
{    
    double random_variable = static_cast<double>(std::rand()) / RAND_MAX;
    if (random_variable > p1) { return 0; }
    else return 1;
}


ConwayTable::ConwayTable(int n, double p1)
{
    N = n;
    for (int i = 0; i < N * N; i++)
    {
        grid.push_back(rollCellState(0.5));
    }

    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            initNeigh(y, x);
        }
    }
}


ConwayTable::ConwayTable(int n, std::vector<char>& v)
{
    N = n;
    grid = v;
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            initNeigh(y, x);
        }
    }
}

void ConwayTable::printNeigh()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << neighGrid[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


void ConwayTable::initNeigh(int y, int x)
{
    char neighCount = 0;
    int xleft, xright, yabove, ybelow;
    //the reason for this section is that the modulo operation in C++ is a mess
    xleft = (x == 0) ? N - 1 : x - 1;
    xright = (x == (N - 1)) ? 0 : x + 1;

    yabove = (y == 0) ? N - 1 : y - 1;
    ybelow = (y == (N - 1)) ? 0 : y + 1;

    neighCount += (*this)(yabove, xleft);
    neighCount += (*this)(yabove, x);
    neighCount += (*this)(yabove, xright);

    neighCount += (*this)(y, xleft);
    neighCount += (*this)(y, xright);

    neighCount += (*this)(ybelow, xleft);
    neighCount += (*this)(ybelow, x);
    neighCount += (*this)(ybelow, xright);

    neighGrid.push_back(neighCount);
}