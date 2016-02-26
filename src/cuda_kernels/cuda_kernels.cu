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
	#define MinTruncation 0.5f
	#define MaxTruncation 1.1f
	#define MaxWeight 10.0f

	cublasHandle_t cublas_handle = nullptr;

	unsigned short volume_size;
	unsigned short voxel_size;

	thrust::device_vector<float> d_grid_voxels_points_4f;
	thrust::device_vector<float> d_grid_voxels_params_2f;
	thrust::device_vector<float> d_grid_matrix_16f;
	thrust::device_vector<float> d_grid_matrix_inv_16f;
	thrust::device_vector<float> d_projection_matrix_16f;

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

	// Invoke kernel 
	// dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	// dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	// matrix_mul_kernel << <dimGrid, dimBlock >> >(d_A, d_B, d_C);
	__global__ void matrix_mul_kernel(int *a, int *b, int *c, int width)
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

				const int m = 4;
				const int k = 4;
				//const int n = 1;
				
				float vg[4] = {
					float(x * half_vol_size - half_vol_size),
					float(y * half_vol_size - half_vol_size),
					float(z * half_vol_size - half_vol_size),
					1.0f };


				float v[4] = {0,0,0,0};


				for (int i = 0; i < m; i++)
					for (int j = 0; j < k; j++)
						//v[i] += grid_matrix[i * m + j] * vg[j];	// row major
						v[j] += grid_matrix[i * m + j] * vg[i];		// col major


				grid_voxels_points_4f[voxel_index * 4 + 0] = v[0];
				grid_voxels_points_4f[voxel_index * 4 + 1] = v[1];
				grid_voxels_points_4f[voxel_index * 4 + 2] = v[2];
				grid_voxels_points_4f[voxel_index * 4 + 3] = v[3];


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

		d_grid_voxels_points_4f = thrust::device_vector<float>(&grid_voxels_points_4f[0], &grid_voxels_points_4f[0] + total_voxels * 4);
		d_grid_voxels_params_2f = thrust::device_vector<float>(&grid_voxels_params_2f[0], &grid_voxels_params_2f[0] + total_voxels * 2);

		d_grid_matrix_16f = thrust::device_vector<float>(&grid_matrix[0], &grid_matrix[0] + 16);

		std::cout << total_voxels << " Starting kernel: << 1, " << vol_size / vx_size + 1 << " >>" << std::endl;

		print_matrix(grid_matrix, 4, 4);

		create_grid_kernel <<< 1, vol_size / vx_size + 1 >>>
			(vol_size, vx_size,
			thrust::raw_pointer_cast(&d_grid_matrix_16f[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_points_4f[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_params_2f[0]));

		thrust::copy(d_grid_voxels_points_4f.begin(), d_grid_voxels_points_4f.end(), &grid_voxels_points_4f[0]);
		thrust::copy(d_grid_voxels_params_2f.begin(), d_grid_voxels_params_2f.end(), &grid_voxels_params_2f[0]);
	}


	__global__ void update_grid_kernel(
		unsigned int vol_size, 
		unsigned int vx_size, 
		float* grid_matrix, 
		float* grid_matrix_inv,
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
		float* grid_matrix_inv,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f,
		float* depth_buffer,
		const float* projection_mat16f,
		unsigned int window_width,
		unsigned int window_height)
	{
//		dim3 threads_per_block(n_threads, 1, 1);
		//dim3 num_blocks(n_blocks, 1, 1);
		const int total_voxels = (vol_size / vx_size + 1) *	(vol_size / vx_size + 1) *	(vol_size / vx_size + 1);

		thrust::device_vector<float> d_grid_matrix_inv_16f(&grid_matrix_inv[0], &grid_matrix_inv[0] + 16);
		thrust::device_vector<float> d_perspective_matrix(&projection_mat16f[0], &projection_mat16f[0] + 16);
		thrust::device_vector<float> d_depth_buffer(&depth_buffer[0], &depth_buffer[0] + window_width * window_height);

		std::cout << total_voxels << " Starting kernel: << 1, " << vol_size / vx_size + 1 << " >>" << std::endl;

		update_grid_kernel <<< 1, vol_size / vx_size + 1 >>> 
			(vol_size, vx_size, 
			thrust::raw_pointer_cast(&d_grid_matrix_16f[0]),
			thrust::raw_pointer_cast(&d_grid_matrix_inv_16f[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_params_2f[0]),
			thrust::raw_pointer_cast(&d_depth_buffer[0]));


		thrust::copy(d_grid_voxels_points_4f.begin(), d_grid_voxels_points_4f.end(), &grid_voxels_points_4f[0]);
	}





	__global__ void grid_init_kernel(
		unsigned short vol_size,
		unsigned short vx_size,
		float* grid_matrix_16f,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f)
	{
		
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		const int z = threadId;

		const int3 dim = { vol_size / vx_size + 1, vol_size / vx_size + 1, vol_size / vx_size + 1 };

		const short half_vol_size = vol_size / 2;

		const short m = 4;
		const short k = 4;
		//const short n = 1;

		for (short y = 0; y < dim.y; ++y)
		{
			for (short x = 0; x < dim.x; ++x)
			{
				const unsigned long voxel_index = x + dim.x * (y + dim.z * z);
				float vg[4] = { 0, 0, 0, 0 };

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

				
				grid_voxels_points_4f[voxel_index * 4 + 0] = vg[0];
				grid_voxels_points_4f[voxel_index * 4 + 1] = vg[1];
				grid_voxels_points_4f[voxel_index * 4 + 2] = vg[2];
				grid_voxels_points_4f[voxel_index * 4 + 3] = vg[3];

				//grid_voxels_params_2f[voxel_index * 2 + 0] = 0.0f;
				//grid_voxels_params_2f[voxel_index * 2 + 1] = 0.0f;
			}
		}
	}



	void grid_init(
		unsigned short vol_size,
		unsigned short vx_size,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f,
		const float* grid_matrix_16f,
		const float* grid_matrix_inv_16f,
		const float* projection_matrix_16f)
	{
		volume_size = vol_size;
		voxel_size = vx_size;

		const unsigned int total_voxels = static_cast<unsigned int>(pow((volume_size / voxel_size + 1), 3));
		

		d_grid_voxels_points_4f = thrust::device_vector<float>(&grid_voxels_points_4f[0], &grid_voxels_points_4f[0] + total_voxels * 4);
		d_grid_voxels_params_2f = thrust::device_vector<float>(&grid_voxels_params_2f[0], &grid_voxels_params_2f[0] + total_voxels * 2);

		d_grid_matrix_16f = thrust::device_vector<float>(&grid_matrix_16f[0], &grid_matrix_16f[0] + 16);
		d_grid_matrix_inv_16f = thrust::device_vector<float>(&grid_matrix_inv_16f[0], &grid_matrix_inv_16f[0] + 16);
		d_projection_matrix_16f = thrust::device_vector<float>(&projection_matrix_16f[0], &projection_matrix_16f[0] + 16);

		

		grid_init_kernel <<< 1, volume_size / voxel_size + 1 >>>
			(volume_size, voxel_size,
			thrust::raw_pointer_cast(&d_grid_matrix_16f[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_points_4f[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_params_2f[0]));
	}



	__global__ void grid_update_kernel(
		float* grid_voxels_params_2f,
		unsigned short vol_size,
		unsigned short vx_size,
		const float* grid_matrix_16f,
		const float* view_matrix_16f,
		const float* view_matrix_inv_16f,
		const float* projection_matrix_16f,
		const float* depth_buffer,
		unsigned short window_width,
		unsigned short window_height)
	{
		const unsigned int total_pixels = window_width * window_height;
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		const int z = threadId;

		const int3 dim = { vol_size / vx_size + 1, vol_size / vx_size + 1, vol_size / vx_size + 1 };

		const short half_vol_size = vol_size / 2;

		const short m = 4;
		const short k = 4;
		//const short n = 1;

		// get translation vector
		const float* ti = &view_matrix_16f[12];	

		//printf("%f %f %f %f \n", ti[0], ti[1], ti[2], ti[3]);

		for (short y = 0; y < dim.y; ++y)
		{
			for (short x = 0; x < dim.x; ++x)
			{
				const unsigned long voxel_index = x + dim.x * (y + dim.z * z);
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


				//printf("%f %f %f \n", vg[0], vg[1], vg[2]);

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
				const float Dp = fabs(depth_buffer[depth_pixel_index]);

				// compute distance from vertex to camera
				float distance_vertex_camera = sqrt(
					pow(ti[0] - vg[0], 2) +
					pow(ti[1] - vg[1], 2) +
					pow(ti[2] - vg[2], 2) + 
					pow(ti[3] - vg[3], 2));


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



	void grid_update(
		const float* view_matrix_16f,
		const float* view_matrix_inv_16f,
		const float* depth_buffer,
		unsigned short window_width,
		unsigned short window_height
		)
	{
		const unsigned int total_pixels = window_width * window_height;
		
		thrust::device_vector<float> d_view_matrix_16f(&view_matrix_16f[0], &view_matrix_16f[0] + 16);
		thrust::device_vector<float> d_view_matrix_inv_16f(&view_matrix_inv_16f[0], &view_matrix_inv_16f[0] + 16);
		thrust::device_vector<float> d_depth_buffer(&depth_buffer[0], &depth_buffer[0] + total_pixels);

		const unsigned int total_voxels = static_cast<unsigned int>(pow((volume_size / voxel_size + 1), 3));

		//for (int i = 0; i < total_voxels; ++i)
		//{
		//	std::cout << " --  "
		//		<< d_grid_voxels_points_4f[i * 4 + 0] << "  " << d_grid_voxels_points_4f[i * 4 + 1]
		//		<< d_grid_voxels_points_4f[i * 4 + 2] 
		//		<< " --  " << d_grid_voxels_params_2f[i * 2 + 0] << "      " << d_grid_voxels_params_2f[i * 2 + 1] << std::endl;
		//}

		//std::cout << std::endl;
		//std::cout << std::endl;
		//print_matrix(view_matrix_16f, 4, 4);
		//std::cout << std::endl;
		//print_matrix(view_matrix_inv_16f, 4, 4);

		grid_update_kernel <<< 1, volume_size / voxel_size + 1 >>>(
			thrust::raw_pointer_cast(&d_grid_voxels_params_2f[0]),
			volume_size, 
			voxel_size,
			thrust::raw_pointer_cast(&d_grid_matrix_16f[0]),
			thrust::raw_pointer_cast(&d_view_matrix_16f[0]),
			thrust::raw_pointer_cast(&d_view_matrix_inv_16f[0]),
			thrust::raw_pointer_cast(&d_projection_matrix_16f[0]),
			thrust::raw_pointer_cast(&d_depth_buffer[0]),
			window_width, 
			window_height
			);

		//for (int i = 0; i < total_voxels; ++i)
		//	std::cout << " ++  "
		//	<< d_grid_voxels_points_4f[i * 4 + 0] << "  " << d_grid_voxels_points_4f[i * 4 + 1]
		//	<< d_grid_voxels_points_4f[i * 4 + 2] 
		//	<< " ++  " << d_grid_voxels_params_2f[i * 2 + 0] << "      " << d_grid_voxels_params_2f[i * 2 + 1] << std::endl;
		//
		//std::cout << "Total Voxels: " << total_voxels << std::endl;
	}


	void grid_get_data(
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f
		)
	{
		thrust::copy(d_grid_voxels_points_4f.begin(), d_grid_voxels_points_4f.end(), &grid_voxels_points_4f[0]);
		thrust::copy(d_grid_voxels_params_2f.begin(), d_grid_voxels_params_2f.end(), &grid_voxels_params_2f[0]);
	}
};