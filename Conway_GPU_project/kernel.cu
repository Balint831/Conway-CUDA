#include "cudaHelperFunctions.cuh"
#include "ConwayTable.cpp"


int main()
{
	//Host side variables
	std::vector<char> v = {0,0,0,0,0,0,
						   0,1,1,0,0,0,
						   0,1,1,0,0,0,
						   0,0,0,1,1,0,
						   0,0,0,1,1,0,
						   0,0,0,0,0,0}; 
	int n = 6;

	
	//Device side variables
	char* grid = nullptr;

	char* neighGrid = nullptr;
	char* neighGrid2 = nullptr;

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

	// Starting threads to initialize the neighbour count grid
	initNeigh<<<dimGrid, dimBlock >>>(n, grid, neighGrid);
	err = cudaGetLastError();
	if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }


	for (int a = 0; a < 10; ++a)
	{
		// Starting threads to step ahead in time 
		oneCell<<<dimGrid, dimBlock >>> (n, grid, neighGrid, neighGrid2);
		err = cudaGetLastError();
		if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
		
		// Copying the data back to the host
		err = cudaMemcpy(v.data(), grid, n * n * sizeof(char), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

		printGrid(v, n);
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

	printGrid(v, n);




	return 0;
}