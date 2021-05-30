#include <iostream>
#include "cudaHelperFunctions.cuh"
#include "ConwayTable.cpp"


int main()
{

	std::vector<char> v = {0,0,0,0,0,0,
						   0,1,1,0,0,0,
						   0,1,1,0,0,0,
						   0,0,0,1,1,0,
						   0,0,0,1,1,0,
						   0,0,0,0,0,0};
	int n = 6;

	//Host side variables
	ConwayTable cnw(n, v);


	//Device side variables
	char* hostGrid = nullptr;
	char* grid = nullptr;

	char* neighGrid = nullptr;
	char* neighGrid2 = nullptr;

	
		





	return 0;
}