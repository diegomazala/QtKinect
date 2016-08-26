#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "RayBox.h"
#include "Timer.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include "helper_cuda.h"
#include "helper_image.h"
#include "KinectFusionKernels/KinectFusionKernels.h"

// Usage: ./Raycastingd.exe 3 1 1.5 1.5 -15 1.5 1.5 -10
int main(int argc, char **argv)
{
	int vol_size = atoi(argv[1]);
	int vx_size = atoi(argv[2]);

	Eigen::Vector3i voxel_count(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

	Eigen::Vector3f origin((float)atof(argv[3]), (float)atof(argv[4]), (float)atof(argv[5]));
	Eigen::Vector3f target((float)atof(argv[6]), (float)atof(argv[7]), (float)atof(argv[8]));
	Eigen::Vector3f direction = (target - origin).normalized();

	Timer t;
	t.start();
	raycast(origin.data(), direction.data(), voxel_count.data(), voxel_size.data());

	std::cout << "Time : " << t.diff_msec() << std::endl;


	return EXIT_SUCCESS;
}
