#ifndef _RAYCAST_KERNELS_H_
#define _RAYCAST_KERNELS_H_


#if _MSC_VER // this is defined when compiling with Visual Studio
#define CUDA_KERNELS_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define CUDA_KERNELS_API // XCode does not need annotating exported functions, so define is empty
#endif

#ifdef _WIN32
#include <windows.h>
#endif



typedef unsigned char	uchar;
typedef unsigned int	uint;
typedef unsigned short	ushort;

extern "C"
{
	void raycast_one(
		const float* origin_float3, 
		const float* direction_float3, 
		const int* voxel_count_int3, 
		const int* voxel_size_int3);


	void raycast_box(
		void* image_rgb_output_uchar3, 
		uint width, 
		uint height,
		uint box_size,
		float fov,
		const float* camera_to_world_mat4f,
		const float* box_transf_mat4f);
};





#endif // _RAYCAST_KERNELS_H_