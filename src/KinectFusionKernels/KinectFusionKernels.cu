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
	ushort* host_ptr;

	buffer_2d() :dev_ptr(nullptr), host_ptr(nullptr){}
};
struct buffer_image_2d
{
	ushort width;
	ushort height;
	size_t pitch;
	uchar4* dev_ptr;
	uchar4* host_ptr;

	buffer_image_2d() :dev_ptr(nullptr), host_ptr(nullptr){}
};
struct buffer_2d_f4
{
	ushort width;
	ushort height;
	size_t pitch;
	float4* dev_ptr;
	float4* host_ptr;

	buffer_2d_f4() :dev_ptr(nullptr), host_ptr(nullptr){}
};

static const float matrix_identity[16] = {
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1 };


buffer_2d		depth_buffer;
ushort			depth_min_distance;
ushort			depth_max_distance;
buffer_2d_f4	vertex_buffer;
buffer_2d_f4	normal_buffer;
grid_3d			grid;
buffer_image_2d	image_buffer;


float* grid_params_dev_ptr					= nullptr;

float* grid_matrix_dev_ptr					= nullptr;
float* projection_matrix_dev_ptr			= nullptr;
float* projection_inverse_matrix_dev_ptr	= nullptr;
float* view_matrix_dev_ptr					= nullptr;
float* camera_to_world_matrix_dev_ptr		= nullptr;

float grid_matrix_host[16];
float projection_matrix_host[16];
float projection_inverse_matrix_host[16];

//
// Gpu typedefs
//
texture<ushort, 2> ushortTexture;
texture<float4, 2, cudaReadModeElementType> float4Texture;

#define PI 3.14159265359
__host__ __device__ float deg2rad(float deg) { return deg*PI / 180.0; }
__host__ __device__ float rad2deg(float rad) { return 180.0*rad / PI; }

__global__ void grid_init_kernel(
	float* grid_voxels_params_2f,
	ushort vx_count)
{
	const ulong threadId = blockIdx.x * blockDim.x + threadIdx.x;
	const ulong z = threadId;

	const dim3 voxel_count = { vx_count, vx_count, vx_count };

	for (short y = 0; y < voxel_count.y; ++y)
	{
		for (short x = 0; x < voxel_count.x; ++x)
		{
			const unsigned long voxel_index = x + voxel_count.x * (y + voxel_count.z * z);
			
			// initialize grid with the new values
			grid_voxels_params_2f[voxel_index * 2 + 0] = 1.0f;
			grid_voxels_params_2f[voxel_index * 2 + 1] = 1.0f;
		}
	}

}


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
			const ulong voxel_index = x + voxel_count.x * (y + voxel_count.z * z);
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


// cross product 
inline __host__ __device__ float4 cross(float4 a, float4 b)
{
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 1.0f);
}


__device__ void matrix_mul_mat_vec_kernel_device(
	const float *a, 
	const float *b, 
	float *c, 
	int width)
{
	int m = width;
	int k = width;
	//int n = 1;

	for (int i = 0; i < m; i++)
		for (int j = 0; j < k; j++)
			c[j] += a[i * m + j] * b[i];		// col major
	//c[i] += a[i * m + j] * b[j];		// row major
}


__device__ void window_coord_to_3d_kernel_device(
	float4* out_vertex,
	const int x,
	const int y,
	const float depth,
	const float* inverse_projection_mat4x4,
	const int window_width,
	const int window_height)
{
	float ndc[3];
	ndc[0] = (x - (window_width * 0.5f)) / (window_width * 0.5f);
	ndc[1] = (y - (window_height * 0.5f)) / (window_height * 0.5f);
	ndc[2] = -1.0f;

	float clip[4];
	clip[0] = ndc[0] * depth;
	clip[1] = ndc[1] * depth;
	clip[2] = ndc[2] * depth;
	clip[3] = 1.0f;

	float vertex_proj_inv[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	matrix_mul_mat_vec_kernel_device(inverse_projection_mat4x4, clip, vertex_proj_inv, 4);

	out_vertex->x = -vertex_proj_inv[0];
	out_vertex->y = -vertex_proj_inv[1];
	out_vertex->z = depth;
	out_vertex->w = 1.0f;

	//out_vertex->x = clip[0];
	//out_vertex->y = clip[1];
	//out_vertex->z = clip[2];
	//out_vertex->w = 1.0f;
}


__global__ void	back_projection_with_normal_estimate_kernel(
	float4 *out_vertices,
	int w, int h,
	ushort max_depth,
	float* inverse_projection_16f)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}

	float depth = ((float)tex2D(ushortTexture, x, y)) * 0.1f;
	float4 vertex;
	window_coord_to_3d_kernel_device(&vertex, x, y, depth, inverse_projection_16f, w, h);

	out_vertices[y * w + x] = vertex;
}


__global__ void	normal_estimate_kernel(float4 *out_normals, int w, int h)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}

	const float4 vertex_uv = tex2D(float4Texture, x, y);
	const float4 vertex_u1v = tex2D(float4Texture, x + 1, y);
	const float4 vertex_uv1 = tex2D(float4Texture, x, y + 1);

	if (vertex_uv.z < 0.01 || vertex_u1v.z < 0.01 || vertex_uv1.z < 0.01)
	{
		out_normals[y * w + x] = make_float4(0, 0, 1, 1);
		return;
	}

	const float4 n1 = vertex_u1v - vertex_uv;
	const float4 n2 = vertex_uv1 - vertex_uv;
	const float4 n = cross(n1, n2);

	out_normals[y * w + x] = normalize(n);
}



__global__ void	raycast_kernel(
	uchar4 *out_image,
	int w, int h,
	float fov_scale,
	float aspect_ratio,
	float* camera_to_world_16f)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}


	out_image[y * w + x] = make_uchar4(128, 0, 0, 255);
}


extern "C"
{

	void knt_cuda_setup(
		ushort vx_count,
		ushort vx_size,
		const float* grid_matrix_16f,
		const float* projection_matrix_16f,
		const float* projection_inverse_matrix_16f,
		ushort depth_width,
		ushort depth_height,
		ushort min_depth,
		ushort max_depth,
		float4& vertex_4f_host_ref,
		float4& normal_4f_host_ref,
		ushort output_image_width,
		ushort output_image_height,
		uchar4& output_image_4uc_ref)
	{
		grid.voxel_count = make_ushort3(vx_count, vx_count, vx_count);
		grid.voxel_size = make_ushort3(vx_size, vx_size, vx_size);

		std::memcpy(&grid_matrix_host[0], grid_matrix_16f, sizeof(float) * 16);
		std::memcpy(&projection_matrix_host[0], projection_matrix_16f, sizeof(float) * 16);
		std::memcpy(&projection_inverse_matrix_host[0], projection_inverse_matrix_16f, sizeof(float) * 16);

		depth_buffer.width = depth_width;
		depth_buffer.height = depth_height;
		depth_min_distance = min_depth;
		depth_max_distance = max_depth;

		vertex_buffer.width = normal_buffer.width = depth_width;
		vertex_buffer.height = normal_buffer.height = depth_height;
		vertex_buffer.host_ptr = &vertex_4f_host_ref;
		normal_buffer.host_ptr = &normal_4f_host_ref;

		image_buffer.width = output_image_width;
		image_buffer.height = output_image_height;
		image_buffer.host_ptr = &output_image_4uc_ref;
	}


	void knt_cuda_allocate()
	{
		//
		// allocate memory in gpu for matrices
		//
		checkCudaErrors(cudaMalloc(&grid_matrix_dev_ptr, sizeof(float) * 16));
		checkCudaErrors(cudaMalloc(&projection_matrix_dev_ptr, sizeof(float) * 16));
		checkCudaErrors(cudaMalloc(&projection_inverse_matrix_dev_ptr, sizeof(float) * 16));
		checkCudaErrors(cudaMalloc(&view_matrix_dev_ptr, sizeof(float) * 16));
		checkCudaErrors(cudaMalloc(&camera_to_world_matrix_dev_ptr, sizeof(float) * 16));
		

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

		//
		// allocate memory in gpu for vertices
		//
		checkCudaErrors(
			cudaMallocPitch(
			&vertex_buffer.dev_ptr,
			&vertex_buffer.pitch,
			sizeof(float4) * vertex_buffer.width,
			vertex_buffer.height));

		//
		// allocate memory in gpu for normals
		//
		checkCudaErrors(
			cudaMallocPitch(
			&normal_buffer.dev_ptr,
			&normal_buffer.pitch,
			sizeof(float4) * normal_buffer.width,
			normal_buffer.height));


		//
		// allocate memory in gpu for output image
		//
		checkCudaErrors(
			cudaMallocPitch(
			&image_buffer.dev_ptr,
			&image_buffer.pitch,
			sizeof(uchar4) * image_buffer.width,
			image_buffer.height));
	}


	void knt_cuda_free()
	{
		checkCudaErrors(cudaFree(grid_matrix_dev_ptr));
		checkCudaErrors(cudaFree(projection_matrix_dev_ptr));
		checkCudaErrors(cudaFree(view_matrix_dev_ptr));
		checkCudaErrors(cudaFree(camera_to_world_matrix_dev_ptr));

		grid_matrix_dev_ptr				= nullptr;
		projection_matrix_dev_ptr		= nullptr;
		view_matrix_dev_ptr				= nullptr;
		camera_to_world_matrix_dev_ptr	= nullptr;

		checkCudaErrors(cudaFree(grid_params_dev_ptr));
		grid_params_dev_ptr			= nullptr;

		checkCudaErrors(cudaFree(depth_buffer.dev_ptr));
		depth_buffer.dev_ptr		= nullptr;

		checkCudaErrors(cudaFree(vertex_buffer.dev_ptr));
		vertex_buffer.dev_ptr = nullptr;

		checkCudaErrors(cudaFree(normal_buffer.dev_ptr));
		normal_buffer.dev_ptr = nullptr;

		checkCudaErrors(cudaFree(image_buffer.dev_ptr));
		image_buffer.dev_ptr = nullptr;
	}

	void knt_cuda_init_grid()
	{
		grid_init_kernel << < 1, grid.voxel_count.z >> >(
			grid_params_dev_ptr,
			grid.voxel_count.x
			);

		checkCudaErrors(cudaDeviceSynchronize());
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



	void knt_cuda_normal_estimation()
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
		checkCudaErrors(cudaBindTexture2D(0, ushortTexture, depth_buffer.dev_ptr, desc, depth_buffer.width, depth_buffer.height, depth_buffer.pitch));

		const dim3 threads_per_block(32, 32);
		dim3 num_blocks;
		num_blocks.x = (depth_buffer.width + threads_per_block.x - 1) / threads_per_block.x;
		num_blocks.y = (depth_buffer.height + threads_per_block.y - 1) / threads_per_block.y;

		back_projection_with_normal_estimate_kernel << <  num_blocks, threads_per_block >> >(
			vertex_buffer.dev_ptr,
			vertex_buffer.width,
			vertex_buffer.height,
			depth_max_distance,
			projection_inverse_matrix_dev_ptr
			);

		cudaChannelFormatDesc desc_normal = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture2D(0, float4Texture, vertex_buffer.dev_ptr, desc_normal, vertex_buffer.width, vertex_buffer.height, normal_buffer.pitch));

		normal_estimate_kernel << <  num_blocks, threads_per_block >> >(
			normal_buffer.dev_ptr,
			normal_buffer.width,
			normal_buffer.height);
	}


	void knt_cuda_raycast(
		float fovy,
		float aspect_ratio,
		const float* camera_to_world_matrix_16f)
	{
		float fov_scale = tan(deg2rad(fovy * 0.5f));

		checkCudaErrors(
			cudaMemcpy(
			camera_to_world_matrix_dev_ptr,
			camera_to_world_matrix_16f,
			sizeof(float) * 16,
			cudaMemcpyHostToDevice
			));

		const dim3 threads_per_block(32, 32);
		dim3 num_blocks;
		num_blocks.x = (image_buffer.width + threads_per_block.x - 1) / threads_per_block.x;
		num_blocks.y = (image_buffer.height + threads_per_block.y - 1) / threads_per_block.y;

		//cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
		//checkCudaErrors(
		//	cudaBindTexture2D(
		//	0, 
		//	uchar4Texture, 
		//	image_buffer.dev_ptr, 
		//	channel_desc, 
		//	image_buffer.width, 
		//	image_buffer.height, 
		//	image_buffer.pitch));

		raycast_kernel << <  num_blocks, threads_per_block >> >(
			image_buffer.dev_ptr,
			image_buffer.width,
			image_buffer.height,
			fov_scale,
			aspect_ratio,
			camera_to_world_matrix_dev_ptr
			);
	}


	void knt_cuda_copy_depth_buffer_to_device(
		const ushort* depth_buffer_host_ptr)
	{
		checkCudaErrors(
			cudaMemcpy(
			depth_buffer.dev_ptr,
			depth_buffer_host_ptr,
			sizeof(ushort) * depth_buffer.width * depth_buffer.height,
			cudaMemcpyHostToDevice
			));
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
			projection_inverse_matrix_dev_ptr,
			&projection_inverse_matrix_host[0],
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

	void knt_cuda_copy_vertices_device_to_host(void* host_ptr)
	{
		cudaMemcpy2D(
			host_ptr,
			sizeof(float4) * depth_buffer.width,
			vertex_buffer.dev_ptr,
			vertex_buffer.pitch,
			sizeof(float4) * depth_buffer.width,
			depth_buffer.height,
			cudaMemcpyDeviceToHost);
	}

	void knt_cuda_copy_device_to_host()
	{
		if (vertex_buffer.host_ptr != nullptr)
			cudaMemcpy2D(
				vertex_buffer.host_ptr,
				sizeof(float4) * depth_buffer.width,
				vertex_buffer.dev_ptr,
				vertex_buffer.pitch,
				sizeof(float4) * depth_buffer.width,
				depth_buffer.height,
				cudaMemcpyDeviceToHost);

		if (normal_buffer.host_ptr != nullptr)
			cudaMemcpy2D(
				normal_buffer.host_ptr,
				sizeof(float4) * depth_buffer.width,
				normal_buffer.dev_ptr,
				normal_buffer.pitch,
				sizeof(float4) * depth_buffer.width,
				depth_buffer.height,
				cudaMemcpyDeviceToHost);
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
