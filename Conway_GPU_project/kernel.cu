#include "cudaHelperFunctions.cuh"
#include "ConwayTable.hpp"
#include "time_meas.hpp"


int main()
{
	//Computational grid parameters
	
	

	//Host side variables
	
	//Beacon n = 6
	/*std::vector<char> v = {0,0,0,0,0,0,
							 0,1,1,0,0,0,
						     0,1,1,0,0,0,
						     0,0,0,1,1,0,
						     0,0,0,1,1,0,
						     0,0,0,0,0,0};
	int n = 6;
	dim3 dimBlock(6,6);*/

	//Blinker n = 5
	/*std::vector<char> v = {0,0,0,0,0,
						   0,0,1,0,0,
						   0,0,1,0,0,
						   0,0,1,0,0,
						   0,0,0,0,0};
	int n = 5;
	dim3 dimBlock(5, 5);*/


	dim3 dimGrid(32, 32);
	dim3 dimBlock(32, 32);
	int n = 1024;
	std::vector<char> v = fillVector(0.5, n);
	
	
	
	//Device side variables
	char* grid = nullptr; //Conway table, which stores if a cell is alive or not

	char* neighGrid = nullptr; // A table that keeps the count of each cell's living neighbours - to read
	char* neighGrid2 = nullptr; // A table that keeps the count of each cell's living neighbours - to write
	// After one whole timestep, neighGrid2 is copied into neighGrid

	// Allocating memory on the device 
	auto err = cudaMalloc((void**)&grid, n * n * sizeof(char));
	if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	err = cudaMalloc((void**)&neighGrid, n * n * sizeof(char));
	if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	err = cudaMalloc((void**)&neighGrid2, n * n * sizeof(char));
	if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

	// Copying data from the host to the device
	err = cudaMemcpy(grid, v.data(), n * n * sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

	// Setting up a grid of thread blocks to initialize the neighbour count grid
	initNeigh<<<dimGrid, dimBlock >>>(n, grid, neighGrid);
	err = cudaGetLastError();
	if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }

	std::ofstream f_output("cnw_gpu_visu.txt");

	std::cout << "Now, we startin'" << std::endl;
	for (int a = 0; a < 10; ++a)
	{
		auto t1 = tmark();
		// Copying the neighGrid to the "writable" neighGrid2
		err = cudaMemcpy(neighGrid2, neighGrid, n * n * sizeof(char), cudaMemcpyDeviceToDevice);

		// Starting threads to step ahead in time 
		oneCell<<<dimGrid, dimBlock >>> (n, grid, neighGrid, neighGrid2);
		err = cudaGetLastError();
		if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
		
		// Copying the data back to the host
		err = cudaMemcpy(v.data(), grid, n * n * sizeof(char), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

		// Handing the new neighGrid to the "read only" variable
		err = cudaMemcpy(neighGrid, neighGrid2, n * n * sizeof(char), cudaMemcpyDeviceToDevice);

		//printGrid(v, n);
		auto t2 = tmark();
		std::cout << delta_time(t1, t2) << std::endl;
		
	}
	
	
	// Copying the data back to the host
	err = cudaMemcpy(v.data(), grid, n*n*sizeof(char), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

	// Freeing the device memory
	err = cudaFree(neighGrid);
	if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree(neighGrid2);
	if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree(grid);
	if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }


	return 0;
}