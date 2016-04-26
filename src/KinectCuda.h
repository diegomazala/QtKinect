#ifndef __Q_KINECT_CUDA_H__
#define __Q_KINECT_CUDA_H__

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>

#ifndef ushort
typedef unsigned short ushort;
#endif





template<typename Type>
struct gpu_cpu_map
{
	Type* host;
	Type* dev;

	gpu_cpu_map() :host(nullptr), dev(nullptr){}
};



class KinectCuda
{
public:
	KinectCuda();
	virtual ~KinectCuda();

	void set_depth_buffer(
		ushort* h_depth_buffer, 
		ushort width, ushort height, 
		ushort min_distance, 
		ushort max_distance);

	void allocate();
	void free();
	void copyHostToDevice();
	void copyDeviceToHost();
	void runKernel();
	
	void get_vertex_data(float** vertex_array, size_t& vertex_count, size_t& tuple_size);
	void get_normal_data(float** normal_array, size_t& normal_count, size_t& tuple_size);

//private:
	gpu_cpu_map<ushort> depth_buffer;
	size_t depth_pitch;
	ushort depth_width;
	ushort depth_height;
	ushort depth_min_distance;
	ushort depth_max_distance;

	gpu_cpu_map<float4> vertex_buffer;
	size_t vertex_pitch;

	gpu_cpu_map<float4> normal_buffer;
	size_t normal_pitch;

	std::vector<float4> vertices;
	std::vector<float4> normals;
};

#endif // __Q_KINECT_GPU_H__