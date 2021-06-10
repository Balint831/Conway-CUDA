#include "cudaHelperFunctions.cuh"
#include "ConwayTable.hpp"
#include "time_meas.hpp"


int main()
{
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


	// Gliders
	/*std::vector<char> v = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	};

	int n = 20;*/

	//std::ofstream f_output("cnw_gpu_visu.txt");
	std::ofstream times("gpu_runtimes.txt");
	

	std::vector<int> n_arr = {30, 50, 100, 150, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 2500, 3000 ,5000, 10000, 15000, 20000};
	for (auto n: n_arr)
	{
		std::vector<char> v = fillVector(0.5, n);
		times << n << ",";

		dim3 dimGrid(200, 200);
		dim3 dimBlock(32, 32);


		//Device side variables
		char* grid = nullptr; //Conway table, from which the actual state of the cell is read

		char* grid2 = nullptr; // A table where the new state of the cells is written


		// Allocating memory on the device 
		auto err = cudaMalloc((void**)&grid, n * n * sizeof(char));
		if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
		err = cudaMalloc((void**)&grid2, n * n * sizeof(char));
		if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }


		// Copying data from the host to the device
		err = cudaMemcpy(grid, v.data(), n * n * sizeof(char), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

		err = cudaMemcpy(grid2, grid, n * n * sizeof(char), cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) { std::cout << "Error copying memory on device: " << cudaGetErrorString(err) << "\n"; return -1; }



		

		std::cout << "Now, we startin'" << std::endl;

		// Creating CUDA events so that time can be measured on the device side
		cudaEvent_t evt[2];
		float milliseconds = 0; //elapsed time between two steps
		for (auto& e : evt)
		{
			auto cudaStatus = cudaEventCreate(&e);
			if (cudaStatus != cudaSuccess) { std::cout << "Error creating event: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
		}

		for (int a = 0; a < 60; ++a)
		{
			auto cudaStatus = cudaEventRecord(evt[0]);


			cudaEventRecord(evt[0]);

			// Starting threads to step ahead in time 
			oneCell << <dimGrid, dimBlock >> > (n, grid, grid2);
			err = cudaGetLastError();
			if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }


			// Handing the new grid to the "read only" variable
			err = cudaMemcpy(grid, grid2, n * n * sizeof(char), cudaMemcpyDeviceToDevice);
			if (err != cudaSuccess) { std::cout << "Error copying memory on device: " << cudaGetErrorString(err) << "\n"; return -1; }

			cudaEventRecord(evt[1]);


			// Copying the data back to the host
			err = cudaMemcpy(v.data(), grid2, n * n * sizeof(char), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

			cudaEventSynchronize(evt[1]);
			cudaEventElapsedTime(&milliseconds, evt[0], evt[1]);
			std::cout << milliseconds << std::endl;
			times << milliseconds << ",";
			//printGrid(v, n);
			//writeGrid(v, n, f_output);

			



		}
		times << std::endl;

		for (auto& e : evt) { cudaEventDestroy(e); }

		// Copying the data back to the host
		err = cudaMemcpy(v.data(), grid, n * n * sizeof(char), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

		// Freeing the device memory
		err = cudaFree(grid);
		if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

		err = cudaFree(grid2);
		if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }
	}

	times.close();

	return 0;
}