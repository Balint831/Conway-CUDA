#include "device_launch_parameters.h" //bunch of define directives
#include "cuda_runtime.h"
#include "cuda.h"

dim3 dimGrid(64, 64);
dim3 dimBlock(16, 16);


__device__ void increaseNeighbourCount(char* neighGrid2, int y, int x, int N)
{
	int xleft, xright, yabove, ybelow;

	xleft = (x == 0) ? N - 1 : x - 1;
	xright = (x == (N - 1)) ? 0 : x + 1;

	yabove = (y == 0) ? N - 1 : y - 1;
	ybelow = (y == (N - 1)) ? 0 : y + 1;

	neighGrid2[xleft + N * yabove]	+= 1;
	neighGrid2[x + N * yabove]		+= 1;
	neighGrid2[xright + N * yabove] += 1;

	neighGrid2[xleft + N * y]		+= 1;
	neighGrid2[xright + N * y]		+= 1;

	neighGrid2[xleft + N * ybelow]	+= 1;
	neighGrid2[x + N * ybelow]		+= 1;
	neighGrid2[xright + N * ybelow] += 1;
}


__device__ void decreaseNeighbourCount(char* neighGrid2, int y, int x, int N)
{
	int xleft, xright, yabove, ybelow;

	xleft = (x == 0) ? N - 1 : x - 1;
	xright = (x == (N - 1)) ? 0 : x + 1;

	yabove = (y == 0) ? N - 1 : y - 1;
	ybelow = (y == (N - 1)) ? 0 : y + 1;

	neighGrid2[xleft + N * yabove]	-= 1;
	neighGrid2[x + N * yabove]		-= 1;
	neighGrid2[xright + N * yabove] -= 1;

	neighGrid2[xleft + N * y]		-= 1;
	neighGrid2[xright + N * y]		-= 1;

	neighGrid2[xleft + N * ybelow]	-= 1;
	neighGrid2[x + N * ybelow]		-= 1;
	neighGrid2[xright + N * ybelow] -= 1;
}


__global__ void oneCell(int N, char* grid, char* neighGrid, char* neighGrid2)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (y < N && x < N) // I dont want to run more threads, than the size of the grid
	{
		if (grid[y * N + x] == 0) //check if the cell is dead
		{
			//if the cell is dead and it has 3 living neighbours, make it alive
			if (neighGrid[y * N + x] == 3)
			{
				grid[y * N + x] = 1;
				increaseNeighbourCount( neighGrid2, y, x, N );
			}
		}

		else //the cell is alive
		{
			// if the cell is alive and it has neither 2 nor 3 living neighbours, let it die
			if ((neighGrid[y * N + x] != 3) && (neighGrid[y * N + x] != 2))
			{
				grid[y * N + x] = 0;
				decreaseNeighbourCount( neighGrid2, y, x, N );
			}
		}
	}
}

void oneStep(int N, char* host_grid, char* grid, char* neighGrid, char* neighGrid2)
{
	oneCell<<<dimGrid, dimBlock>>>(N, grid, neighGrid, neighGrid2);

	auto error = cudaMemcpy(host_grid, grid, N * N * sizeof(char), cudaMemcpyDeviceToHost);
}
