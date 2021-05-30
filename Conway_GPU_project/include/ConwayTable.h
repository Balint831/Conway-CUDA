#pragma once

#include <vector>
#include <fstream>
#include <iostream>

class ConwayTable
{
private:

    int N;
    std::vector<char> grid; //the cell states are registered on this grid
    std::vector<char> neighGrid; //the number of living neighbor cells
    std::vector<char> neighGrid2;
    

public:
    
    ConwayTable(int n, double p1); //constructor from 
    ConwayTable(int n, std::vector<char>& v); //constructor from vector
    void initNeigh(int y, int x); //initializing neighbour grid based on cell state grid
    void printNeigh(); //printing neighbour grid
    void increaseNeighbourCount(int y, int x);
    void decreaseNeighbourCount(int y, int x);
    void oneRow(int y, int k); //y - index of row
    void multiRow(int y_start, int y_end, int k);
    void oneStep(int k);

    char& operator()(int y, int x)
    {
        return grid[N * y + x];
    }

    //outputting cell state grid, converting cell states from char to int
    friend std::ostream& operator<<(std::ostream& o, ConwayTable& A)
    {
        for (int i = 0; i < A.N; i++)
        {
            for (int j = 0; j < A.N; j++)
            {
                o << static_cast<int>( A.grid[i * A.N + j] )<< " ";
            }
            o << "\n";
        }
        o << "\n";
        return o;
    }
};

