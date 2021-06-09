#include "device_launch_parameters.h" //bunch of define directives
#include "cuda_runtime_api.h"
#include "cuda.h"


__device__ void copyGrid(char* grid, char* grid2)
{

}


__global__ void oneCell(int N, char* grid, char* grid2)
{
	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < N && x < N) // I don't want to run more threads, than the size of the Conway table
	{

		char neighCount = 0;

		int xleft = (x == 0) ? N - 1 : x - 1;
		int xright = (x == (N - 1)) ? 0 : x + 1;

		int yabove = (y == 0) ? N - 1 : y - 1;
		int ybelow = (y == (N - 1)) ? 0 : y + 1;

		neighCount += grid[N * yabove + xleft];
		neighCount += grid[N * yabove + x];
		neighCount += grid[N * yabove + xright];

		neighCount += grid[N * y + xleft];
		neighCount += grid[N * y + xright];

		neighCount += grid[N * ybelow + xleft];
		neighCount += grid[N * ybelow + x];
		neighCount += grid[N * ybelow + xright];
	
	
		if ((grid[N * y + x] == 0) && (neighCount == 3)) { grid2[N * y + x] = 1; }

		if ((grid[N * y + x] == 1) && ((neighCount != 3) && (neighCount != 2))) { grid2[N * y + x] = 0; }

	}
}
