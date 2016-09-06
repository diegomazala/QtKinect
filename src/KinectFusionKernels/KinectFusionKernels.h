#ifndef _KINECT_FUSION_KERNELS_H_
#define _KINECT_FUSION_KERNELS_H_


#if _MSC_VER // this is defined when compiling with Visual Studio
#define CUDA_KERNELS_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define CUDA_KERNELS_API // XCode does not need annotating exported functions, so define is empty
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_runtime.h>


#ifndef ushort
typedef unsigned short ushort;
#endif
#ifndef uchar
typedef unsigned char uchar;
#endif
#ifndef uint
typedef unsigned int uint;
#endif
#ifndef ulong
typedef unsigned long ulong;
#endif


//static int iDivUp(int a, int b)
//{
//	return (a % b != 0) ? (a / b + 1) : (a / b);
//}

extern "C"
{
	void knt_cuda_setup(
		ushort vx_count,
		ushort vx_size,
		const float* grid_matrix_16f,
		const float* projection_matrix_16f,
		ushort depth_width, 
		ushort depth_height);

	void knt_cuda_allocate();
	void knt_cuda_free();

	void knt_cuda_update_grid(const float* view_matrix_16f);

	void knt_cuda_copy_host_to_device();
	void knt_cuda_copy_device_to_host();
	void knt_cuda_grid_params_copy_device_to_host(float* grid_params_2f);
};





#endif // _KINECT_FUSION_KERNELS_H_