#include "cuda_kernels.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>

extern "C"
{
	cublasHandle_t cublas_handle = nullptr;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary> Initialize CUBLAS creating an handle for its context. </summary>
	///
	/// <remarks> Diego Mazala, 15/02/2016. </remarks>
	///
	/// <returns> true if the context has been created, false otherwise. </returns>
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool cublas_init()
	{
		// Create a handle for CUBLAS
		return (cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
	}


	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary> Destroy CUBLAS context. </summary>
	///
	/// <remarks> Diego Mazala, 15/02/2016. </remarks>
	///
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool cublas_cleanup()
	{
		// Destroy the handle
		return (cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
	}


	void make_identity_4x4(float *mat)
	{
		for (int i = 0; i < 16; ++i)
			mat[i] = 0;


		mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
	}



	//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
	void print_matrix(const float *A, int nr_rows_A, int nr_cols_A)
	{
		for (int i = 0; i < nr_rows_A; ++i)
		{
			for (int j = 0; j < nr_cols_A; ++j)
			{
				std::cout << A[j * nr_rows_A + i] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}



	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>  Multiply the arrays A and B on GPU and save the result in C </summary>
	///
	/// <remarks> Diego Mazala, 15/02/2016. </remarks>
	///
	/// <param name="A"> Input left matrix. </param>
	/// <param name="B"> Input right matrix. </param>
	/// <param name="C"> Output result matrix. </param>
	///
	////////////////////////////////////////////////////////////////////////////////////////////////////
	bool cublas_matrix_mul(float *dev_C, const float *dev_A, const float *dev_B, const int m, const int k, const int n)
	{
		int lda = m, ldb = k, ldc = m;
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;

		if (!cublas_handle)
			cublas_init();

		// Do the actual multiplication
		return (cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, dev_A, lda, dev_B, ldb, beta, dev_C, ldc) == CUBLAS_STATUS_SUCCESS);
	}



	//usage example with eigen: matrix_mul(mat_C.data(), mat_A.data(), &vector_of_eigen_vector4[0][0], A.rows(), A.cols(), vector_of_eigen_vector4.size());
	void matrix_mulf(float* mat_c, const float* mat_a, const float* mat_b, int m, int k, int n)
	{
		// transfer to device 
		thrust::device_vector<float> d_a(&mat_a[0], &mat_a[0] + m * k);
		thrust::device_vector<float> d_b(&mat_b[0], &mat_b[0] + k * n);
		thrust::device_vector<float> d_c(&mat_c[0], &mat_c[0] + m * n);

		// Multiply A and B on GPU
		cublas_matrix_mul(thrust::raw_pointer_cast(&d_c[0]), thrust::raw_pointer_cast(&d_a[0]), thrust::raw_pointer_cast(&d_b[0]), m, k, n);

		thrust::copy(d_c.begin(), d_c.end(), &mat_c[0]);
	}

	__global__ void matrixMult(int *a, int *b, int *c, int width)
	{
		int k, sum = 0;
		int col = threadIdx.x + blockDim.x * blockIdx.x;
		int row = threadIdx.y + blockDim.y * blockIdx.y;
		if (col < width && row < width)
		{
			for (k = 0; k < width; k++)
				sum += a[row * width + k] * b[k * width + col];
			c[row * width + col] = sum;
		}
	}

	__global__ void compute_pixel_depth_kernel(
		float* in_out_depth_buffer_1f, 
		float* out_pixel_2f,
		const float* in_world_points_4f, 
		const float* in_clip_points_4f, 
		unsigned int point_count, 
		unsigned int window_width, 
		unsigned int window_height)
	{
		// unique block index inside a 3D block grid
		//const unsigned long long int blockId = blockIdx.x //1D
		//	+ blockIdx.y * gridDim.x //2D
		//	+ gridDim.x * gridDim.y * blockIdx.z; //3D
		//// global unique thread index, block dimension uses only x-coordinate
		//const unsigned long long int threadId = blockId * blockDim.x + threadIdx.x;
		
		const unsigned long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		const unsigned int pixel_count = window_width * window_height;

		if (threadId >= point_count)
			return;
		
		const float clip_x = in_clip_points_4f[threadId * 4 + 0];	
		const float clip_y = in_clip_points_4f[threadId * 4 + 1];
		const float clip_z = in_clip_points_4f[threadId * 4 + 2];
		const float clip_w = in_clip_points_4f[threadId * 4 + 3];
		const float ndc_x = clip_x / clip_w;
		const float ndc_y = clip_y / clip_w;
		const float ndc_z = clip_z / clip_w;

		if (ndc_x < -1 || ndc_x > 1 || ndc_y < -1 || ndc_y > 1 || ndc_z < -1 || ndc_z > 1)
			return;

		const float pixel_x = window_width / 2.0f * ndc_x + window_width / 2.0f;
		const float pixel_y = window_height / 2.0f * ndc_y + window_height / 2.0f;

		const int depth_index = (int)pixel_y * window_width + (int)pixel_x;

		if (depth_index > 0 && depth_index < pixel_count)
		{
			const float& curr_depth = fabs(in_out_depth_buffer_1f[depth_index]);
			const float& new_depth = fabs(in_world_points_4f[threadId * 4 + 2]);	// z coord
			__syncthreads();

			if (new_depth < curr_depth)
			{
				in_out_depth_buffer_1f[depth_index] = new_depth;
			}
		}
	}


	void compute_depth_buffer(	
			float* depth_buffer, 
			float* window_coords_2f,
			const float* world_points_4f, 
			unsigned int point_count, 
			const float* projection_mat4x4, 
			unsigned int window_width, 
			unsigned int window_height)
	{
		const unsigned int pixel_count = window_width * window_height;

		// transfer to device 
		thrust::device_vector<float> d_depth_buffer(&depth_buffer[0], &depth_buffer[0] + pixel_count);
		thrust::device_vector<float> d_projection_mat(&projection_mat4x4[0], &projection_mat4x4[0] + 16);
		
		thrust::device_vector<float> d_world_points(&world_points_4f[0], &world_points_4f[0] + point_count * 4);
		thrust::device_vector<float> d_clip_points(&world_points_4f[0], &world_points_4f[0] + point_count * 4);
		
		thrust::device_vector<float> d_window_coords_points(&window_coords_2f[0], &window_coords_2f[0] + point_count * 2);

		cublas_matrix_mul(thrust::raw_pointer_cast(&d_clip_points[0]), thrust::raw_pointer_cast(&d_projection_mat[0]), thrust::raw_pointer_cast(&d_world_points[0]), 4, 4, point_count);

		unsigned int threads_per_block = 1024;
		unsigned int num_blocks = 1 + point_count / threads_per_block;

		compute_pixel_depth_kernel <<< num_blocks, threads_per_block >>> (
			thrust::raw_pointer_cast(&d_depth_buffer[0]),
			thrust::raw_pointer_cast(&d_window_coords_points[0]),
			thrust::raw_pointer_cast(&d_world_points[0]),
			thrust::raw_pointer_cast(&d_clip_points[0]), 
			point_count,
			window_width, 
			window_height);

		thrust::copy(d_depth_buffer.begin(), d_depth_buffer.end(), &depth_buffer[0]);
		thrust::copy(d_window_coords_points.begin(), d_window_coords_points.end(), &window_coords_2f[0]);
	}



	__global__ void create_grid_kernel(
		unsigned int vol_size,
		unsigned int vx_size,
		float* grid_matrix,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f)
	{
		//const unsigned long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		uint3 dim;
		dim.x = vol_size / vx_size + 1;
		dim.y = vol_size / vx_size + 1;
		dim.z = vol_size / vx_size + 1;

		const int z = threadId;

		const int half_vol_size = vol_size / 2;

		for (int y = 0; y < dim.y; ++y)
		{
			for (int x = 0; x < dim.x; ++x)
			{
				const int voxel_index = x + dim.x * (y + dim.z * z);

				float4 v = {
					grid_matrix[12] + float(x * half_vol_size - half_vol_size),
					grid_matrix[13] + float(y * half_vol_size - half_vol_size),
					grid_matrix[14] - float(z * half_vol_size - half_vol_size),
					1.0f };


				grid_voxels_points_4f[voxel_index * 4 + 0] = v.x;
				grid_voxels_points_4f[voxel_index * 4 + 1] = v.y;
				grid_voxels_points_4f[voxel_index * 4 + 2] = v.z;
				grid_voxels_points_4f[voxel_index * 4 + 3] = v.w;

				grid_voxels_params_2f[voxel_index * 2 + 0] = 0.0f;
				grid_voxels_params_2f[voxel_index * 2 + 1] = 0.0f;
			}
		}

	}


	void create_grid(
		unsigned int vol_size,
		unsigned int vx_size,
		float* grid_matrix,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f)
	{
		const int total_voxels = (vol_size / vx_size + 1) *	(vol_size / vx_size + 1) *	(vol_size / vx_size + 1);

		thrust::device_vector<float> d_grid_matrix(&grid_matrix[0], &grid_matrix[0] + 16);
		thrust::device_vector<float> d_grid_voxels_points(&grid_voxels_points_4f[0], &grid_voxels_points_4f[0] + total_voxels * 4);
		thrust::device_vector<float> d_grid_voxels_params(&grid_voxels_params_2f[0], &grid_voxels_params_2f[0] + total_voxels * 2);

		std::cout << total_voxels << " Starting kernel: << 1, " << vol_size / vx_size + 1 << " >>" << std::endl;

		create_grid_kernel << < 1, vol_size / vx_size + 1 >> >
			(vol_size, vx_size,
			thrust::raw_pointer_cast(&d_grid_matrix[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_points[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_params[0]));

		thrust::copy(d_grid_voxels_points.begin(), d_grid_voxels_points.end(), &grid_voxels_points_4f[0]);
		thrust::copy(d_grid_voxels_params.begin(), d_grid_voxels_params.end(), &grid_voxels_params_2f[0]);
	}


	__global__ void update_grid_kernel(
		unsigned int vol_size, 
		unsigned int vx_size, 
		float* grid_matrix, 
		float* grid_voxels,
		float* depth_buffer)
	{
		//const unsigned long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		uint3 dim;
		dim.x = vol_size / vx_size + 1;
		dim.y = vol_size / vx_size + 1;
		dim.z = vol_size / vx_size + 1;

		const int z = threadId;

		const int half_vol_size = vol_size / 2;
		
		for (int y = 0; y < dim.y; ++y)
		{
			for (int x = 0; x < dim.x; ++x)
			{
				const int voxel_index = x + dim.x * (y + dim.z * z);

				float4 v = { 
					grid_matrix[12] + float(x * half_vol_size - half_vol_size),
					grid_matrix[13] + float(y * half_vol_size - half_vol_size),
					grid_matrix[14] - float(z * half_vol_size - half_vol_size),
					1.0f};



				grid_voxels[voxel_index * 4 + 0] = v.x;
				grid_voxels[voxel_index * 4 + 1] = v.y;
				grid_voxels[voxel_index * 4 + 2] = v.z;
				grid_voxels[voxel_index * 4 + 3] = v.w;
			}
		}

	}

	void update_grid(
		unsigned int vol_size, 
		unsigned int vx_size, 
		float* grid_matrix, 
		float* grid_voxels,
		float* depth_buffer,
		unsigned int window_width,
		unsigned int window_height)
	{
//		dim3 threads_per_block(n_threads, 1, 1);
		//dim3 num_blocks(n_blocks, 1, 1);
		const int total_voxels = (vol_size / vx_size + 1) *	(vol_size / vx_size + 1) *	(vol_size / vx_size + 1);

		thrust::device_vector<float> d_grid_matrix(&grid_matrix[0], &grid_matrix[0] + 16);
		thrust::device_vector<float> d_grid_voxels(&grid_voxels[0], &grid_voxels[0] + total_voxels * 4);
		thrust::device_vector<float> d_depth_buffer(&depth_buffer[0], &depth_buffer[0] + window_width * window_height);

		std::cout << total_voxels << " Starting kernel: << 1, " << vol_size / vx_size + 1 << " >>" << std::endl;

		update_grid_kernel << < 1, vol_size / vx_size + 1 >> > 
			(vol_size, vx_size, 
			thrust::raw_pointer_cast(&d_grid_matrix[0]),
			thrust::raw_pointer_cast(&d_grid_voxels[0]),
			thrust::raw_pointer_cast(&d_depth_buffer[0]));


		thrust::copy(d_grid_voxels.begin(), d_grid_voxels.end(), &grid_voxels[0]);
	}
};