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

};





#endif // _KINECT_FUSION_KERNELS_H_