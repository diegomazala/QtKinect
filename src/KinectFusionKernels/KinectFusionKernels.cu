#ifndef _KINECT_CUDA_KERNELS_CU_
#define _KINECT_CUDA_KERNELS_CU_

#include "KinectFusionKernels.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


ushort* depth_buffer = 0;
size_t depth_pitch;
ushort depth_width;
ushort depth_height;


extern "C"
{
	void knt_cuda_allocate()
	{
		// allocate memory in gpu for depth buffer
		checkCudaErrors(
			cudaMallocPitch(
			&depth_buffer,
			&depth_pitch,
			sizeof(ushort) * depth_width,
			depth_height));



	}


	void knt_cuda_free()
	{
		checkCudaErrors(cudaFree(depth_buffer));
	}
}

#endif // #ifndef _KINECT_CUDA_KERNELS_CU_
