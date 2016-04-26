#include "KinectCuda.h"
#include "helper_cuda.h"
#include "helper_image.h"
#include "Eigen/Dense"
#include "Projection.h"
#include "KinectSpecs.h"
#include "cuda_kernels\cuda_kernels.h"


KinectCuda::KinectCuda()
{

}

KinectCuda::~KinectCuda()
{

}


void KinectCuda::set_depth_buffer(ushort* h_depth_buffer,
	ushort width, ushort height,
	ushort min_distance,
	ushort max_distance)
{
	depth_width = width;
	depth_height = height;
	depth_min_distance = min_distance;
	depth_max_distance = max_distance;
	depth_buffer.host = h_depth_buffer;
}


void KinectCuda::allocate()
{
	// allocate memory in gpu for depth buffer
	checkCudaErrors(
		cudaMallocPitch(
		&depth_buffer.dev, 
		&depth_pitch, 
		sizeof(ushort) * depth_width, 
		depth_height));

	// allocate memory in gpu for vertices
	checkCudaErrors(
		cudaMallocPitch(
		&vertex_buffer.dev,
		&vertex_pitch,
		depth_width * sizeof(float4),
		depth_height));


	// allocate memory in gpu for normals
	checkCudaErrors(
		cudaMallocPitch(
		&normal_buffer.dev,
		&normal_pitch,
		depth_width * sizeof(float4),
		depth_height));

}


void KinectCuda::free()
{
	checkCudaErrors(cudaFree(depth_buffer.dev));
	checkCudaErrors(cudaFree(vertex_buffer.dev));
	checkCudaErrors(cudaFree(normal_buffer.dev));
}


void KinectCuda::copyHostToDevice()
{
	checkCudaErrors(
		cudaMemcpy2D(
		depth_buffer.dev,
		depth_pitch,
		depth_buffer.host,
		sizeof(ushort) * depth_width,
		sizeof(ushort) * depth_width,
		depth_height,
		cudaMemcpyHostToDevice));
}





void KinectCuda::copyDeviceToHost()
{
	const int size = depth_width * depth_height;

	vertices.resize(size);
	normals.resize(size);

	cudaMemcpy2D(
		vertices.data(),
		sizeof(float4) * depth_width,
		vertex_buffer.dev,
		vertex_pitch,
		sizeof(float4) * depth_width,
		depth_height,
		cudaMemcpyDeviceToHost);

	cudaMemcpy2D(
		normals.data(),
		sizeof(float4) * depth_width,
		normal_buffer.dev,
		normal_pitch,
		sizeof(float4) * depth_width,
		depth_height,
		cudaMemcpyDeviceToHost);
}



void KinectCuda::runKernel()
{
	const float aspect_ratio = 
		static_cast<float>(depth_width) / 
		static_cast<float>(depth_height);


	Eigen::Matrix4f h_inverse_projection = perspective_matrix_inverse<float>(
		KINECT_V1_FOVY, 
		aspect_ratio, 
		static_cast<float>(depth_min_distance),
		static_cast<float>(depth_max_distance));

	back_projection_with_normal_estimation(
		vertex_buffer.dev,
		normal_buffer.dev,
		depth_buffer.dev,
		depth_width,
		depth_height,
		depth_max_distance,
		depth_pitch,
		normal_pitch,
		h_inverse_projection.data());
}


void KinectCuda::get_vertex_data(float** vertex_array, size_t& vertex_count, size_t& tuple_size)
{
	*vertex_array = &vertices.data()[0].x;
	vertex_count = vertices.size();
	tuple_size = 4;
}


void KinectCuda::get_normal_data(float** normal_array, size_t& normal_count, size_t& tuple_size)
{
	*normal_array = &normals.data()[0].x;
	normal_count = normals.size();
	tuple_size = 4;
}


