#ifndef _KINECT_CUDA_KERNELS_CU_
#define _KINECT_CUDA_KERNELS_CU_

#include "KinectFusionKernels.h"
#include <helper_cuda.h>
#include <helper_math.h>

#define MinTruncation 0.5f
#define MaxTruncation 1.1f
#define MaxWeight 10.0f

struct grid_3d
{
	ushort3 voxel_count;
	ushort3 voxel_size;
	grid_3d()
	{
		voxel_count = make_ushort3(3, 3, 3);
		voxel_size = make_ushort3(1, 1, 1);
	}

	ushort3 volume_size() const
	{
		return make_ushort3(
			voxel_count.x * voxel_size.x,
			voxel_count.y * voxel_size.y,
			voxel_count.z * voxel_size.z);
	}

	float3 half_volume_size() const
	{
		return make_float3(
			voxel_count.x * voxel_size.x * 0.5f,
			voxel_count.y * voxel_size.y * 0.5f,
			voxel_count.z * voxel_size.z * 0.5f);
	}

	ulong total_voxels() const
	{
		return voxel_count.x * voxel_count.y * voxel_count.z;
	}
};


struct buffer_2d
{
	ushort width;
	ushort height;
	size_t pitch;
	ushort* dev_ptr;

	buffer_2d() :dev_ptr(nullptr){}
};

static const float matrix_identity[16] = {
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1 };


buffer_2d	depth_buffer;
grid_3d		grid;


float* grid_params_dev_ptr			= nullptr;

float* grid_matrix_dev_ptr			= nullptr;
float* projection_matrix_dev_ptr	= nullptr;
float* view_matrix_dev_ptr			= nullptr;

float grid_matrix_host[16];
float projection_matrix_host[16];

__global__ void grid_update_kernel(
	float* grid_voxels_params_2f,
	ushort vx_count,
	ushort vx_size,
	float* grid_matrix_16f,
	float* view_matrix_16f,
	float* projection_matrix_16f,
	ushort* depth_buffer,
	ushort window_width,
	ushort window_height)
{
	const ulong total_voxels = vx_count * vx_count * vx_count;
	const ulong total_pixels = window_width * window_height;
	const ulong threadId = blockIdx.x * blockDim.x + threadIdx.x;
	const ulong z = threadId;

	const dim3 voxel_count = { vx_count, vx_count, vx_count };

	const float half_vol_size = vx_count * vx_size * 0.5f;

	const short m = 4;
	const short k = 4;

	// get translation vector
	const float4 ti = make_float4(
		view_matrix_16f[12], 
		view_matrix_16f[13],
		view_matrix_16f[14],
		view_matrix_16f[15]);


	for (short y = 0; y < voxel_count.y; ++y)
	{
		for (short x = 0; x < voxel_count.x; ++x)
		{
			const unsigned long voxel_index = x + voxel_count.x * (y + voxel_count.z * z);
			float vg[4] = { 0, 0, 0, 0 };
			float v[4] = { 0, 0, 0, 0 };


			// grid space
			float g[4] = {
				float(x * vx_size - half_vol_size),
				float(y * vx_size - half_vol_size),
				float(z * vx_size - half_vol_size),
				1.0f };


			// to world space
			for (short i = 0; i < m; i++)
				for (short j = 0; j < k; j++)
					vg[j] += grid_matrix_16f[i * m + j] * g[i];		// col major



			// to camera space
			for (short i = 0; i < m; i++)
				for (short j = 0; j < k; j++)
					v[j] += view_matrix_16f[i * m + j] * vg[i];		// col major
					//v[j] += view_matrix_inv_16f[i * m + j] * vg[i];	// col major
					//v[i] += view_matrix_inv_16f[i * m + j] * vg[j];	// row major


			// compute clip space vertex
			float clip[4] = { 0, 0, 0, 0 };
			for (short i = 0; i < m; i++)
				for (short j = 0; j < k; j++)
					clip[j] += projection_matrix_16f[i * m + j] * v[i];


			// compute ndc vertex
			const float3 ndc = {
				clip[0] / clip[3],
				clip[1] / clip[3],
				clip[2] / clip[3] };


			// compute window coordinates
			const float2 window_coord = {
				window_width / 2.0f * ndc.x + window_width / 2.0f,
				window_height / 2.0f * ndc.y + window_height / 2.0f };

			// cast to int 
			const int2 pixel = { (int)window_coord.x, (int)window_coord.y };

			// compute depth buffer pixel index in the array
			const int depth_pixel_index = pixel.y * window_width + pixel.x;

			// check if it is out of window size
			if (depth_pixel_index < 0 || depth_pixel_index > total_pixels - 1)
				continue;


			// get depth buffer value
			//const float Dp = fabs(depth_buffer[depth_pixel_index]) * 0.1f;
			const float Dp = depth_buffer[depth_pixel_index] * 0.1f;

			// compute distance from vertex to camera
			float distance_vertex_camera = sqrt(
				pow(ti.x - vg[0], 2) +
				pow(ti.y - vg[1], 2) +
				pow(ti.z - vg[2], 2) +
				pow(ti.w - vg[3], 2));


			//// compute signed distance function
			const float sdf = Dp - distance_vertex_camera;



			//const double half_voxel_size = voxel_size;// *0.5;
			if (fabs(sdf) > vx_size)
				continue;


			const float prev_tsdf = grid_voxels_params_2f[voxel_index * 2 + 0];
			const float prev_weight = grid_voxels_params_2f[voxel_index * 2 + 1];


			float tsdf = sdf;

			if (sdf > 0)
				tsdf = fmin(1.0f, sdf / MaxTruncation);
			else
				tsdf = fmax(-1.0f, sdf / MinTruncation);



			// Izadi method
			const float weight = fmin(MaxWeight, prev_weight + 1);
			const float tsdf_avg = (prev_tsdf * prev_weight + tsdf * weight) / (prev_weight + weight);
			//// Open Fusion method
			//const float weight = std::fmin(MaxWeight, prev_weight + 1);
			//const float tsdf_avg = (prev_tsdf * prev_weight + tsdf * 1) / (prev_weight + 1);

			// update grid with the new values
			grid_voxels_params_2f[voxel_index * 2 + 0] = tsdf_avg;
			grid_voxels_params_2f[voxel_index * 2 + 1] = weight;

		}
	}

}


extern "C"
{

	

	void knt_cuda_setup(
		ushort vx_count,
		ushort vx_size,
		const float* grid_matrix_16f,
		const float* projection_matrix_16f,
		ushort depth_width,
		ushort depth_height)
	{
		grid.voxel_count = make_ushort3(vx_count, vx_count, vx_count);
		grid.voxel_size = make_ushort3(vx_size, vx_size, vx_size);

		std::memcpy(&grid_matrix_host[0], grid_matrix_16f, sizeof(float) * 16);
		std::memcpy(&projection_matrix_host[0], projection_matrix_16f, sizeof(float) * 16);

		depth_buffer.width = depth_width;
		depth_buffer.height = depth_height;
	}


	void knt_cuda_allocate()
	{
		//
		// allocate memory in gpu for matrices
		//
		checkCudaErrors(cudaMalloc(&grid_matrix_dev_ptr, sizeof(float) * 16));
		checkCudaErrors(cudaMalloc(&projection_matrix_dev_ptr, sizeof(float) * 16));
		checkCudaErrors(cudaMalloc(&view_matrix_dev_ptr, sizeof(float) * 16));

		//
		// allocate memory in gpu for grid parameters
		//
		checkCudaErrors(cudaMalloc(&grid_params_dev_ptr, sizeof(float) * 2 * grid.total_voxels()));

		//
		// allocate memory in gpu for depth buffer
		//
		checkCudaErrors(
			cudaMallocPitch(
			&depth_buffer.dev_ptr,
			&depth_buffer.pitch,
			sizeof(ushort) * depth_buffer.width,
			depth_buffer.height));

		////
		//// allocate memory in gpu for vertices
		////
		//checkCudaErrors(
		//	cudaMallocPitch(
		//	&vertex_buffer_dev,
		//	&vertex_pitch,
		//	depth_width * sizeof(float4),
		//	depth_height));

		////
		//// allocate memory in gpu for normals
		////
		//checkCudaErrors(
		//	cudaMallocPitch(
		//	&normal_buffer_dev,
		//	&normal_pitch,
		//	depth_width * sizeof(float4),
		//	depth_height));

	}


	void knt_cuda_free()
	{
		checkCudaErrors(cudaFree(grid_matrix_dev_ptr));
		checkCudaErrors(cudaFree(projection_matrix_dev_ptr));
		checkCudaErrors(cudaFree(view_matrix_dev_ptr));

		grid_matrix_dev_ptr			= nullptr;
		projection_matrix_dev_ptr	= nullptr;
		view_matrix_dev_ptr			= nullptr;

		checkCudaErrors(cudaFree(grid_params_dev_ptr));
		grid_params_dev_ptr			= nullptr;

		checkCudaErrors(cudaFree(depth_buffer.dev_ptr));
		depth_buffer.dev_ptr		= nullptr;


		//checkCudaErrors(cudaFree(vertex_buffer_dev));
		//checkCudaErrors(cudaFree(normal_buffer_dev));
	}

	void knt_cuda_update_grid(const float* view_matrix_16f)
	{
		checkCudaErrors(
			cudaMemcpy(
			view_matrix_dev_ptr,
			view_matrix_16f,
			sizeof(float) * 16,
			cudaMemcpyHostToDevice
			));

		grid_update_kernel << < 1, grid.voxel_count.z >> >(
			grid_params_dev_ptr,
			grid.voxel_count.x,
			grid.voxel_size.x,
			grid_matrix_dev_ptr,
			projection_matrix_dev_ptr,
			view_matrix_dev_ptr,
			depth_buffer.dev_ptr,
			depth_buffer.width,
			depth_buffer.height
			);

		checkCudaErrors(cudaDeviceSynchronize());
	}

	void knt_cuda_copy_host_to_device()
	{
		checkCudaErrors(
			cudaMemcpy(
			grid_matrix_dev_ptr,
			&grid_matrix_host[0],
			sizeof(float) * 16,
			cudaMemcpyHostToDevice
			));

		checkCudaErrors(
			cudaMemcpy(
			projection_matrix_dev_ptr,
			&projection_matrix_host[0],
			sizeof(float) * 16,
			cudaMemcpyHostToDevice
			));

		checkCudaErrors(
			cudaMemcpy(
			view_matrix_dev_ptr,
			&matrix_identity[0],
			sizeof(float) * 16,
			cudaMemcpyHostToDevice
			));
	}

	void knt_cuda_copy_device_to_host()
	{
		
	}

	void knt_cuda_grid_params_copy_device_to_host(float* grid_params_2f)
	{
		checkCudaErrors(
			cudaMemcpy(
			grid_params_2f,
			grid_params_dev_ptr,
			sizeof(float) * 2 * grid.total_voxels(),
			cudaMemcpyDeviceToHost
			));
	}

}

#endif // #ifndef _KINECT_CUDA_KERNELS_CU_
