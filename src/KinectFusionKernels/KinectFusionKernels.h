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



typedef unsigned char	uchar;
typedef unsigned int	uint;
typedef unsigned short	ushort;
typedef float2 VolumeType;

static int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

extern "C"
{
	void volume_render_setup(void *h_volume, ushort volume_width, ushort volume_height, ushort volume_depth);
	void volume_render_cleanup();


	void knt_cuda_allocate();
	void knt_cuda_free();


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





#endif // _KINECT_FUSION_KERNELS_H_