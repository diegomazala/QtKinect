#include "cuda_kernels.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "helper_math.h"

#define PI 3.14159265359
__host__ __device__ float deg2rad(float deg) { return deg*PI / 180.0; }
__host__ __device__ float rad2deg(float rad) { return 180.0*rad / PI; }

texture<ushort, 2> ushortTexture;
texture<float4, 2, cudaReadModeElementType> float4Texture;


//typedef unsigned char VolumeType;
typedef float2 VolumeType;
texture<VolumeType, 3, cudaReadModeElementType> texVolumeType;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture
cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef struct
{
	float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix


__device__
float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}


__device__ uint saturate_rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}




inline __host__ __device__  int3 get_index_3d_from_array(
	int array_index,
	const dim3& voxel_count)
{
	//return make_int3(
	//	int(std::fmod(array_index, voxel_count.x)),
	//	int(std::fmod(array_index / voxel_count.y, voxel_count.y)),
	//	int(array_index / (voxel_count.x * voxel_count.y)));

	return make_int3(
		int(fmodf(array_index, (voxel_count.x + 1))),
		int(fmodf(array_index / (voxel_count.y + 1), (voxel_count.y + 1))),
		int(array_index / ((voxel_count.x + 1) * (voxel_count.y + 1))));
}

inline __host__ __device__ int get_index_from_3d_volume(int3 pt, dim3 voxel_count)
{
	//return pt.z * voxel_count.x * voxel_count.y + pt.y * voxel_count.y + pt.x;
	return pt.z * (voxel_count.x + 1) * (voxel_count.y + 1) + pt.y * (voxel_count.y + 1) + pt.x;
}

// 
// Face Index
// 0-Top, 1-Bottom, 2-Front, 3-Back, 4-Left, 5-Right
//
inline __device__ int get_index_from_box_face(int face, int last_index, dim3 voxel_count)
{
	switch (face)
	{
	case 0: return last_index + voxel_count.x;					// Top
	case 1: return last_index - voxel_count.x;					// Bottom
	case 2: return last_index - voxel_count.x * voxel_count.y;	// Front
	case 3: return last_index + voxel_count.x * voxel_count.y;	// Back
	case 4: return last_index - 1;								// Left
	case 5: return last_index + 1;								// Right
	default: return -1;
	}
}


inline __host__ __device__ float3 compute_normal(
	const float3& p1,
	const float3& p2,
	const float3& p3)
{
	float3 u = p2 - p1;
	float3 v = p3 - p1;

	return normalize(cross(v, u));
}


// http://www.graphics.cornell.edu/pubs/1997/MT97.html
__host__ __device__ bool triangle_intersection(
	const float3& p,
	const float3& d,
	const float3& v0,
	const float3& v1,
	const float3& v2,
	float3& hit)
{
	float a, f, u, v;
	const float3 e1 = v1 - v0;
	const float3 e2 = v2 - v0;

	const float3 h = cross(d, e2);
	a = dot(e1, h);

	if (a > -0.00001f && a < 0.00001f)
		return false;

	f = 1.0f / a;
	const float3 s = p - v0;
	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f)
		return false;

	const float3 q = cross(s, e1);
	v = f * dot(d, q);

	if (v < 0.0f || u + v > 1.0f)
		return false;

	float t = f * dot(e2, q);

	if (t > 0.00001f) // ray intersection
	{
		hit = p + (d * t);
		return true;
	}
	else
		return false;
}

__host__ __device__ bool quad_intersection(
	const float3& p,
	const float3& d,
	const float3& p1,
	const float3& p2,
	const float3& p3,
	const float3& p4,
	float3& hit)
{
	return (triangle_intersection(p, d, p1, p2, p3, hit)
		|| triangle_intersection(p, d, p3, p4, p1, hit));
}

__host__ __device__ bool quad_intersection(
	const float3& p,
	const float3& d,
	const float3& p1,
	const float3& p2,
	const float3& p3)
{

	// 
	// Computing normal of quad
	//
	float3 e21 = p2 - p1;					// compute edge 
	float3 e31 = p3 - p1;					// compute edge
	float3 n = normalize(cross(e21, e31));	// compute normal

	float ndotd = dot(n, d);

	//
	// check if dot == 0, 
	// i.e, plane is parallel to the ray
	//
	if (fabs(ndotd) < 1e-6f)					// Choose your tolerance
		return false;

	float t = -dot(n, p - p1) / ndotd;
	float3 M = p + d * t;

	// 
	// Projecting vector M - p1 onto e21 and e31
	//
	float3 Mp = M - p;
	float u = dot(Mp, e21);
	float v = dot(Mp, e31);

	//
	// If 0 <= u <= | p2 - p1 | ^ 2 and 0 <= v <= | p3 - p1 | ^ 2,
	// then the point of intersection M lies inside the square, 
	// else it's outside.
	//
	return (u >= 0.0f && u <= dot(e21, e21)
		&& v >= 0.0f && v <= dot(e31, e31));
}



__host__ __device__ int box_intersection(
	const float3 p,
	const float3 dir,
	const float3 boxCenter,
	float boxWidth,
	float boxHeigth,
	float boxDepth,
	float3& hit1,
	float3& hit2,
	float3& hit1Normal,
	float3& hit2Normal)
{
	float x2 = boxWidth * 0.5f;
	float y2 = boxHeigth * 0.5f;
	float z2 = boxDepth * 0.5f;

	float3 p1 = make_float3(-x2, y2, -z2);
	float3 p2 = make_float3(x2, y2, -z2);
	float3 p3 = make_float3(x2, y2, z2);
	float3 p4 = make_float3(-x2, y2, z2);
	float3 p5 = make_float3(-x2, -y2, -z2);
	float3 p6 = make_float3(x2, -y2, -z2);
	float3 p7 = make_float3(x2, -y2, z2);
	float3 p8 = make_float3(-x2, -y2, z2);

	p1 += boxCenter;
	p2 += boxCenter;
	p3 += boxCenter;
	p4 += boxCenter;
	p5 += boxCenter;
	p6 += boxCenter;
	p7 += boxCenter;
	p8 += boxCenter;


	float3 hit[2];
	float3 hitNormal[2];
	int hitCount = 0;

	// check top
	if (quad_intersection(p, dir, p1, p2, p3, p4, hit[hitCount]))
	{
		hitNormal[hitCount] = compute_normal(p1, p2, p3);
		hitCount++;
	}

	// check bottom
	if (quad_intersection(p, dir, p5, p8, p7, p6, hit[hitCount]))
	{
		hitNormal[hitCount] = compute_normal(p5, p8, p7);
		hitCount++;
	}

	// check front
	if (hitCount < 2 && quad_intersection(p, dir, p4, p3, p7, p8,
		hit[hitCount]))
	{
		hitNormal[hitCount] = compute_normal(p4, p3, p7);
		hitCount++;
	}

	// check back
	if (hitCount < 2 && quad_intersection(p, dir, p1, p5, p6, p2,
		hit[hitCount]))
	{
		hitNormal[hitCount] = compute_normal(p1, p5, p6);
		hitCount++;
	}

	// check left
	if (hitCount < 2 && quad_intersection(p, dir, p1, p4, p8, p5,
		hit[hitCount]))
	{
		hitNormal[hitCount] = compute_normal(p1, p4, p8);
		hitCount++;
	}

	// check right
	if (hitCount < 2 && quad_intersection(p, dir, p2, p6, p7, p3,
		hit[hitCount]))
	{
		hitNormal[hitCount] = compute_normal(p2, p6, p7);
		hitCount++;
	}

	if (hitCount > 0)
	{
		if (hitCount > 1)
		{
			if (length(p - hit[0]) < length(p - hit[1]))
			{
				hit1 = hit[0];
				hit2 = hit[1];
				hit1Normal = hitNormal[0];
				hit2Normal = hitNormal[1];
			}
			else
			{
				hit1 = hit[1];
				hit2 = hit[0];
				hit1Normal = hitNormal[1];
				hit2Normal = hitNormal[0];
			}
		}
		else
		{
			hit1 = hit[0];
			hit1Normal = hitNormal[0];
		}
	}

	return hitCount;
}

__host__ __device__ int box_intersection(
	const float3 p,
	const float3 dir,
	const float3 boxCenter,
	float boxWidth,
	float boxHeigth,
	float boxDepth,
	float3& hit1Normal,
	float3& hit2Normal,
	int& face)
{
	float x2 = boxWidth * 0.5f;
	float y2 = boxHeigth * 0.5f;
	float z2 = boxDepth * 0.5f;

	float3 p1 = make_float3(-x2, y2, -z2);
	float3 p2 = make_float3(x2, y2, -z2);
	float3 p3 = make_float3(x2, y2, z2);
	float3 p4 = make_float3(-x2, y2, z2);
	float3 p5 = make_float3(-x2, -y2, -z2);
	float3 p6 = make_float3(x2, -y2, -z2);
	float3 p7 = make_float3(x2, -y2, z2);
	float3 p8 = make_float3(-x2, -y2, z2);

	p1 += boxCenter;
	p2 += boxCenter;
	p3 += boxCenter;
	p4 += boxCenter;
	p5 += boxCenter;
	p6 += boxCenter;
	p7 += boxCenter;
	p8 += boxCenter;

	float3 hitNormal[2];
	int hitCount = 0;

	// check top
	if (quad_intersection(p, dir, p1, p2, p3))
	{
		hitNormal[hitCount] = compute_normal(p1, p2, p3);
		hitCount++;
		face = 0;
	}

	// check bottom
	if (quad_intersection(p, dir, p5, p8, p7))
	{
		hitNormal[hitCount] = compute_normal(p5, p8, p7);
		hitCount++;
		face = 1;
	}

	// check front
	if (hitCount < 2 && quad_intersection(p, dir, p4, p3, p7))
	{
		hitNormal[hitCount] = compute_normal(p4, p3, p7);
		hitCount++;
		face = 2;
	}

	// check back
	if (hitCount < 2 && quad_intersection(p, dir, p1, p5, p6))
	{
		hitNormal[hitCount] = compute_normal(p1, p5, p6);
		hitCount++;
		face = 3;
	}

	// check left
	if (hitCount < 2 && quad_intersection(p, dir, p1, p4, p8))
	{
		hitNormal[hitCount] = compute_normal(p1, p4, p8);
		hitCount++;
		face = 4;
	}

	// check right
	if (hitCount < 2 && quad_intersection(p, dir, p2, p6, p7))
	{
		hitNormal[hitCount] = compute_normal(p2, p6, p7);
		hitCount++;
		face = 5;
	}

	if (hitCount > 0)
	{
		if (hitCount > 1)
		{
			hit1Normal = hitNormal[0];
			hit2Normal = hitNormal[1];
		}
		else
		{
			hit1Normal = hitNormal[0];
		}
	}

	return hitCount;
}



extern "C"
{

	#define MinTruncation 0.5f
	#define MaxTruncation 1.1f
	#define MaxWeight 10.0f

	
	cublasHandle_t cublas_handle = nullptr;

	unsigned short volume_size;
	unsigned short voxel_count;
	unsigned short voxel_size;

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

	// cross product 
	inline __host__ __device__ float4 cross(float4 a, float4 b)
	{
		return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 1.0f);
	}


	



	__global__ void compute_pixel_depth_kernel(
		ushort* in_out_depth_buffer_1us, 
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
			const float& curr_depth = in_out_depth_buffer_1us[depth_index];
			const float& new_depth = fabs(in_world_points_4f[threadId * 4 + 2]);	// z coord
			__syncthreads();

			if (new_depth < curr_depth)
			{
				in_out_depth_buffer_1us[depth_index] = new_depth;
			}
		}
	}


	void compute_depth_buffer(	
			ushort* depth_buffer, 
			float* window_coords_2f,
			const float* world_points_4f, 
			unsigned int point_count, 
			const float* projection_mat4x4, 
			unsigned int window_width, 
			unsigned int window_height)
	{
		const unsigned int pixel_count = window_width * window_height;

		// transfer to device 
		thrust::device_vector<ushort> d_depth_buffer(&depth_buffer[0], &depth_buffer[0] + pixel_count);
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



	





	void grid_init(
		unsigned short vx_count,
		unsigned short vx_size,
		const float* grid_matrix_16f,
		const float* grid_matrix_inv_16f,
		const float* projection_matrix_16f)
	{
		volume_size = vx_count * vx_size;
		voxel_size = vx_size;
		voxel_count = vx_count;

		const unsigned int total_voxels = voxel_count * voxel_count * voxel_count;

		std::cout << "cuda        : " << voxel_count << " * " << voxel_size << " = " << volume_size << std::endl;
		std::cout << "total voxels: " << total_voxels << std::endl;

		
		d_grid_voxels_params_2f = thrust::device_vector<float>(total_voxels * 2, 1.0f);

		d_grid_matrix_16f = thrust::device_vector<float>(&grid_matrix_16f[0], &grid_matrix_16f[0] + 16);
		d_grid_matrix_inv_16f = thrust::device_vector<float>(&grid_matrix_inv_16f[0], &grid_matrix_inv_16f[0] + 16);
		d_projection_matrix_16f = thrust::device_vector<float>(&projection_matrix_16f[0], &projection_matrix_16f[0] + 16);
	}



	__global__ void grid_update_kernel(
		float* grid_voxels_params_2f,
		unsigned short vol_size,
		unsigned short vx_size,
		const float* grid_matrix_16f,
		const float* view_matrix_16f,
		const float* view_matrix_inv_16f,
		const float* projection_matrix_16f,
		const ushort* depth_buffer,
		unsigned short window_width,
		unsigned short window_height)
	{
		const unsigned int total_pixels = window_width * window_height;
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		const int z = threadId;

		const int3 voxel_count = { vol_size / vx_size, vol_size / vx_size, vol_size / vx_size };

		const short half_vol_size = vol_size / 2;

		const short m = 4;
		const short k = 4;
		//const short n = 1;

		// get translation vector
		const float* ti = &view_matrix_16f[12];	


		for (short y = 0; y < voxel_count.y; ++y)
		{
			for (short x = 0; x < voxel_count.x; ++x)
			{
				const unsigned long voxel_index = x + voxel_count.x * (y + voxel_count.z * z);
				float vg[4] = { 0, 0, 0, 0 };
				float v[4] = { 0, 0, 0, 0 };

				// grid space
#if 1
				float g[4] = {
					float(x * vx_size - half_vol_size),
					float(y * vx_size - half_vol_size),
					float(z * vx_size - half_vol_size),
					1.0f };
#else
				float g[4] = {
					float(x * vx_size),
					float(y * vx_size),
					float(z * vx_size),
					1.0f };
#endif

				

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
		const ushort* depth_buffer,
		unsigned short window_width,
		unsigned short window_height
		)
	{

		const unsigned int total_pixels = window_width * window_height;
		
		thrust::device_vector<float> d_view_matrix_16f(&view_matrix_16f[0], &view_matrix_16f[0] + 16);
		thrust::device_vector<float> d_view_matrix_inv_16f(&view_matrix_inv_16f[0], &view_matrix_inv_16f[0] + 16);
		thrust::device_vector<ushort> d_depth_buffer(&depth_buffer[0], &depth_buffer[0] + total_pixels);


		const unsigned int total_voxels = static_cast<unsigned int>(pow((volume_size / voxel_size), 3));

		grid_update_kernel << < 1, volume_size / voxel_size >> >(
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
			
	}


	void grid_get_data(
		float* grid_voxels_params_2f
		)
	{
		thrust::copy(d_grid_voxels_params_2f.begin(), d_grid_voxels_params_2f.end(), &grid_voxels_params_2f[0]);
	}


	__device__ float3 mul_vec_dir_matrix(const float* M_3x4, const float3& v)
	{
		return make_float3(
			dot(v, make_float3(M_3x4[0], M_3x4[4], M_3x4[8])),
			dot(v, make_float3(M_3x4[1], M_3x4[5], M_3x4[9])),
			dot(v, make_float3(M_3x4[2], M_3x4[6], M_3x4[10])));
	}

	__global__ void	raycast_image_grid_kernel(
		uchar3 *d_output_image,
		ushort image_width,
		ushort image_height,
		const dim3& voxel_count,
		const dim3& voxel_size,
		float fovy,
		const float* camera_to_world_mat4x4,
		const float* box_transf_mat4x4,
		float* grid_voxels_params_2f)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;

		if ((x >= image_width) || (y >= image_height))
			return;

		dim3 volume_size = dim3(voxel_count.x * voxel_size.x, voxel_count.y * voxel_size.y, voxel_count.z * voxel_size.z);


		float3 camera_pos = make_float3(camera_to_world_mat4x4[12], camera_to_world_mat4x4[13], camera_to_world_mat4x4[14]);

		float scale = tan(deg2rad(fovy * 0.5f));
		float aspect_ratio = (float)image_width / (float)image_height;

		float near_plane = 0.3f;
		float far_plane = 512.0f;

		// Convert from image space (in pixels) to screen space
		// Screen Space along X axis = [-aspect ratio, aspect ratio] 
		// Screen Space along Y axis = [-1, 1]
		float3 screen_coord = make_float3(
			(2 * (x + 0.5f) / (float)image_width - 1) * aspect_ratio * scale,
			(1 - 2 * (y + 0.5f) / (float)image_height) * scale,
			-1.0f);

		// transform vector by matrix (no translation)
		// multDirMatrix
		float3 dir = mul_vec_dir_matrix(camera_to_world_mat4x4, screen_coord);
		float3 direction = normalize(dir);

		// clear pixel
		d_output_image[y * image_width + x] = make_uchar3(8, 16, 32);



		//
		// Check if the ray of this pixel intersect the whole volume
		//

		float3 half_volume_size = make_float3(volume_size.x * 0.5f, volume_size.y * 0.5f, volume_size.z * 0.5f);
		float3 half_voxel_size = make_float3(voxel_size.x * 0.5f, voxel_size.y * 0.5f, voxel_size.z * 0.5f);
		float3 hit1;
		float3 hit2;
		float3 hit1_normal;
		float3 hit2_normal;
		int face = -1;

		int intersections_count = box_intersection(
			camera_pos,
			direction,
			half_volume_size,	// volume center
			volume_size.x,
			volume_size.y,
			volume_size.z,
			hit1,
			hit1,
			hit1_normal,
			hit2_normal
			);

		if (intersections_count < 1)
		{
			return;
		}

		// encontrar qual voxel inicial foi intersectado
		// a partir deste voxel, calcular as intersecções do raio até ele encontrar um zero-crossing

		int3 hit_int = make_int3(hit1.x, hit1.y, hit1.z);
		int voxel_index = get_index_from_3d_volume(hit_int, voxel_count);
		float3 last_voxel = make_float3(hit_int.x, hit_int.y, hit_int.z);
		float last_tsdf = grid_voxels_params_2f[voxel_index * 2];

		int total_voxels = voxel_count.x * voxel_count.y * voxel_count.z;

		bool zero_crossing = false;
		// 
		// Check intersection with each box inside of volume
		// 
		while (voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x * voxel_count.y) && !zero_crossing)
		{
			int face = -1;
			intersections_count = box_intersection(
				camera_pos,
				direction,
				last_voxel + half_voxel_size,
				voxel_size.x,
				voxel_size.y,
				voxel_size.z,
				hit1_normal,
				hit2_normal,
				face);
#if 0
			voxel_index = get_index_from_box_face(face, voxel_index, voxel_count);
			int3 last_voxel_index = make_int3(
				int(fmodf(voxel_index, (voxel_count.x + 1))),
				int(fmodf(voxel_index / (voxel_count.y + 1), (voxel_count.y + 1))),
				int(voxel_index / ((voxel_count.x + 1) * (voxel_count.y + 1))));
			
			float tsdf = grid_voxels_params_2f[voxel_index * 2];

			zero_crossing = (tsdf < 0 && last_tsdf < 0) || (tsdf > 0 && last_tsdf > 0);

			last_tsdf = tsdf;
#endif
		}

		if (zero_crossing)
			d_output_image[y * image_width + x] = make_uchar3(8, 128, 255);

	}

	void raycast_image_grid(
		void* image_rgb_output_uchar3,
		ushort image_width,
		ushort image_height,
		const ushort* voxel_count_xyz,
		const ushort* voxel_size_xyz,
		float fovy,
		const float* camera_to_world_mat4f,
		const float* box_transf_mat4f)
	{
		thrust::device_vector<uchar3> d_image_rgb = thrust::device_vector<uchar3>(image_width * image_height);
		thrust::device_vector<float> d_camera_to_world_mat4f = thrust::device_vector<float>(&camera_to_world_mat4f[0], &camera_to_world_mat4f[0] + 16);
		thrust::device_vector<float> d_box_transform_mat4f = thrust::device_vector<float>(&box_transf_mat4f[0], &box_transf_mat4f[0] + 16);

		dim3 voxel_count = dim3(voxel_count_xyz[0], voxel_count_xyz[1], voxel_count_xyz[2]);
		dim3 voxel_size = dim3(voxel_size_xyz[0], voxel_size_xyz[1], voxel_size_xyz[2]);

		const dim3 threads_per_block(32, 32);
		const dim3 num_blocks = dim3(iDivUp(image_width, threads_per_block.x), iDivUp(image_height, threads_per_block.y));

		// One kernel per pixel
		raycast_image_grid_kernel << <  num_blocks, threads_per_block >> >(
			thrust::raw_pointer_cast(&d_image_rgb[0]),
			image_width,
			image_height,
			voxel_count,
			voxel_size,
			fovy,
			thrust::raw_pointer_cast(&d_camera_to_world_mat4f[0]),
			thrust::raw_pointer_cast(&d_box_transform_mat4f[0]),
			thrust::raw_pointer_cast(&d_grid_voxels_params_2f[0])
			);

		thrust::copy(d_image_rgb.begin(), d_image_rgb.end(), (uchar3*)image_rgb_output_uchar3);
	}


	__device__ void matrix_mul_mat_vec_kernel_device(const float *a, const float *b, float *c, int width)
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

		float vertex_proj_inv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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

	__global__ void	d_back_projection_with_normal_estimate_kernel(
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

	

	__global__ void	d_normal_estimate_kernel(float4 *out_normals, int w, int h)
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
		
		const float4 n1 = vertex_u1v - vertex_uv;
		const float4 n2 = vertex_uv1 - vertex_uv;
		const float4 n = cross(n1, n2);

		out_normals[y * w + x] = normalize(n);
	}
	
	void back_projection_with_normal_estimation(
		float4* d_out_vertices_4f,
		float4* d_out_normals_4f,
		const ushort* d_depth_buffer,
		const ushort depth_width,
		const ushort depth_height,
		const ushort max_depth,
		const size_t in_pitch,
		const size_t out_pitch,
		const float* h_inverse_projection_mat4x4
		)
	{
		thrust::device_vector<float> d_inverse_projection_mat_16f(&h_inverse_projection_mat4x4[0], &h_inverse_projection_mat4x4[0] + 16);

		cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
		checkCudaErrors(cudaBindTexture2D(0, ushortTexture, d_depth_buffer, desc, depth_width, depth_height, in_pitch));


		const dim3 threads_per_block(32, 32);
		dim3 num_blocks;
		num_blocks.x = (depth_width + threads_per_block.x - 1) / threads_per_block.x;
		num_blocks.y = (depth_height + threads_per_block.y - 1) / threads_per_block.y;

		d_back_projection_with_normal_estimate_kernel << <  num_blocks, threads_per_block >> >(
			d_out_vertices_4f,
			depth_width,
			depth_height,
			max_depth,
			thrust::raw_pointer_cast(&d_inverse_projection_mat_16f[0])
			);

		checkCudaErrors(cudaDeviceSynchronize());

		cudaChannelFormatDesc desc_normal = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture2D(0, float4Texture, d_out_vertices_4f, desc_normal, depth_width, depth_height, out_pitch));

		d_normal_estimate_kernel << <  num_blocks, threads_per_block >> >(
			d_out_normals_4f,
			depth_width,
			depth_height);

		checkCudaErrors(cudaDeviceSynchronize());
	}

	__device__ float vec4f_magnitude(const float4& v)
	{
		return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
	}

	__device__ float4 vec4f_subtract(const float4& a, const float4& b)
	{
		return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}


	__global__ void	d_icp_matching_vertices_kernel(ushort2 *out_indices, float *out_distances, const float4 *in_vertices, ushort w, ushort h, ushort half_window)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= w || y >= h)
			return;

		if (x + half_window >= w || y + half_window >= h
			|| x - half_window < 0 || y - half_window < 0)
			return;


		const float4 in_vertex = in_vertices[y * w + x];
		float min_distance = FLT_MAX;
		ushort2 index = make_ushort2(0, 0);

		for (ushort xx = x - half_window; xx <= x + half_window; ++xx)
		{
			for (ushort yy = y - half_window; yy <= y + half_window; ++yy)
			{
				const float4 cur_vertex = tex2D(float4Texture, xx, yy);
				
				// only compute if the vertex has a valid distance, i.e, not zero
				if (cur_vertex.z > 0.1)
				{

					const float4 vec_diff = vec4f_subtract(cur_vertex, in_vertex);
					const float distance = vec4f_magnitude(vec_diff);

					if (distance < min_distance)
					{
						min_distance = distance;
						index.x = xx;
						index.y = yy;
					}
				}
			}
		}

		out_indices[y * w + x] = index;
		out_distances[y * w + x] = min_distance;
	}


	__global__ void	d_icp_matching_vertices_check_kernel(float4 *out_vertices, const ushort2 *in_indices, int w, int h)
	{
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		if (x >= w || y >= h)
			return;

		const ushort2 index = in_indices[y * w + x];
		out_vertices[y * w + x] = tex2D(float4Texture, index.x, index.y);
	}



	void icp_matching_vertices(
		ushort2* d_out_indices,
		float* d_out_distances,
		float4* d_in_vertices_t0_4f,
		float4* d_in_vertices_t1_4f,
		const ushort depth_width,
		const ushort depth_height,
		const size_t vertex_pitch,
		const size_t index_pitch,
		const ushort half_window_search_size
		)
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture2D(0, float4Texture, d_in_vertices_t0_4f, desc, depth_width, depth_height, vertex_pitch));

		const dim3 threads_per_block(32, 32);
		dim3 num_blocks;
		num_blocks.x = (depth_width + threads_per_block.x - 1) / threads_per_block.x;
		num_blocks.y = (depth_height + threads_per_block.y - 1) / threads_per_block.y;

		d_icp_matching_vertices_kernel << <  num_blocks, threads_per_block >> >(
			d_out_indices,
			d_out_distances,
			d_in_vertices_t1_4f,
			depth_width,
			depth_height,
			half_window_search_size);
	}

	struct Ray
	{
		float3 o;   // origin
		float3 d;   // direction
	};

	__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
	{
		// compute intersection of ray with all six bbox planes
		float3 invR = make_float3(1.0f) / r.d;
		float3 tbot = invR * (boxmin - r.o);
		float3 ttop = invR * (boxmax - r.o);

		// re-order intersections to find smallest and largest on each axis
		float3 tmin = fminf(ttop, tbot);
		float3 tmax = fmaxf(ttop, tbot);

		// find the largest tmin and the smallest tmax
		float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
		float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

		*tnear = largest_tmin;
		*tfar = smallest_tmax;

		return smallest_tmax > largest_tmin;
	}

	__global__ void raycast_kernel(
		uchar3* out_pixels,
		uint image_width,
		uint image_height,
		uint vol_size,
		uint vx_size,
		const float4* grid_vertices,
		float3 origin,
		float3 direction,
		float ray_near,
		float ray_far)
	{
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		if (x >= image_width || y >= image_height)
			return;
		
		float3 half_voxel = make_float3(vx_size * 0.5f, vx_size * 0.5f, vx_size * 0.5f);

		const uint total_voxels = static_cast<uint>((vol_size / vx_size + 1) * (vol_size / vx_size + 1) * (vol_size / vx_size + 1));

		// calculate eye ray in world space
		Ray eyeRay;
		eyeRay.o = origin;
		eyeRay.d = direction;

		uchar p = 128;
#if 0
		int prev_voxel = -1;
		float tnear, tfar;
		for (int i=0; i<total_voxels; ++i)
		{
			const float3 v = make_float3(grid_vertices[i].x, grid_vertices[i].y, grid_vertices[i].z);
			const float3 corner_min = v - half_voxel;
			const float3 corner_max = v + half_voxel;

			if (intersectBox(eyeRay, corner_min, corner_max, &tnear, &tfar))
			{
				if (p < 245)
					p += 10;
			}
		}
#endif
		out_pixels[y * image_width + x].x = p;
		out_pixels[y * image_width + x].y = 0;
		out_pixels[y * image_width + x].z = 0;

#if 0



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

				//float4 v = {
				//	grid_matrix[12] + float(x * half_vol_size - half_vol_size),
				//	grid_matrix[13] + float(y * half_vol_size - half_vol_size),
				//	grid_matrix[14] - float(z * half_vol_size - half_vol_size),
				//	1.0f };




				//grid_voxels[voxel_index * 4 + 0] = v.x;
				//grid_voxels[voxel_index * 4 + 1] = v.y;
				//grid_voxels[voxel_index * 4 + 2] = v.z;
				//grid_voxels[voxel_index * 4 + 3] = v.w;
			}
		}
#endif

	}


	void raycast_grid(
		float4* grid_vertices,
		float2* grid_params,
		ushort volume_size,
		ushort voxel_size,
		float3 origin,
		float3 direction,
		float ray_near,
		float ray_far,
		uchar3* pixel_bufer,
		const ushort width,
		const ushort height
		)
	{
		const uint total_voxels = static_cast<uint>(pow((volume_size / voxel_size + 1), 3));

		thrust::device_vector<float4> d_grid_vertices = thrust::device_vector<float4>(&grid_vertices[0], &grid_vertices[0] + total_voxels);
		thrust::device_vector<float2> d_grid_params = thrust::device_vector<float2>(&grid_params[0], &grid_params[0] + total_voxels);
		thrust::device_vector<uchar3> d_pixel_buffer = thrust::device_vector<uchar3>(width * height);
		
		//thrust::copy(d_grid_vertices.begin(), d_grid_vertices.end(), &grid_vertices[0]);
		//thrust::copy(d_grid_params.begin(), d_grid_params.end(), &grid_params[0]);

		const dim3 threads_per_block(32, 32);
		const dim3 num_blocks = dim3(iDivUp(width, threads_per_block.x), iDivUp(height, threads_per_block.y));
		//num_blocks.x = (width + threads_per_block.x - 1) / threads_per_block.x;
		//num_blocks.y = (height + threads_per_block.y - 1) / threads_per_block.y;

		raycast_kernel <<<  num_blocks, threads_per_block >>>(
			thrust::raw_pointer_cast(&d_pixel_buffer[0]),
			width,
			height,
			volume_size,
			voxel_size,
			thrust::raw_pointer_cast(&d_grid_vertices[0]),
			origin,
			direction,
			ray_near,
			ray_far
			);

		thrust::copy(d_pixel_buffer.begin(), d_pixel_buffer.end(), &pixel_bufer[0]);
	}




	__global__ void	d_render_volume_kernel(
		uint *d_output, 
		uint imageW, 
		uint imageH,
		float density, 
		float brightness,
		float transferOffset, 
		float transferScale)
	{
		const int maxSteps = 500;
		const float tstep = 0.01f;
		const float opacityThreshold = 0.95f;
		const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
		const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

		uint x = blockIdx.x*blockDim.x + threadIdx.x;
		uint y = blockIdx.y*blockDim.y + threadIdx.y;

		if ((x >= imageW) || (y >= imageH)) return;

		float u = (x / (float)imageW)*2.0f - 1.0f;
		float v = (y / (float)imageH)*2.0f - 1.0f;


		// calculate eye ray in world space
		Ray eyeRay;
		eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

		// find intersection with box
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

		if (!hit) return;

		if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

		// march along ray from front to back, accumulating color
		float4 sum = make_float4(0.0f);
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d*tnear;
		float3 step = eyeRay.d*tstep;

		float last_tsdf = tex3D(texVolumeType, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f).x;


		for (int i = 0; i<maxSteps; i++)
		{
			// read from 3D texture
			// remap position to [0, 1] coordinates
			float2 sample = tex3D(texVolumeType, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);

			//sample *= 64.0f;    // scale for 10-bit data

			float tsdf = sample.x;
			if (std::signbit(tsdf) != std::signbit(last_tsdf))
			{
				sum = make_float4(1);
				break;
			}
			else
			{
				last_tsdf = tsdf;
			}

			t += tstep;

			if (t > tfar) break;

			pos += step;
		}

		sum *= brightness;

		// write output color
		d_output[y*imageW + x] = saturate_rgbaFloatToInt(sum);

	}

	

	void render_volume(
		dim3 gridSize, 
		dim3 blockSize, 
		uint *d_output, 
		uint imageW, 
		uint imageH,
		float density, 
		float brightness, 
		float transferOffset, 
		float transferScale)
	{
		d_render_volume_kernel <<<gridSize, blockSize >>>
			(d_output, imageW, imageH, density, brightness, transferOffset, transferScale);
	}


	void initCuda_render_volume(void *h_volume, cudaExtent volumeSize)
	{
		// create 3D array
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
		checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

		// copy data to 3D array
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
		copyParams.dstArray = d_volumeArray;
		copyParams.extent = volumeSize;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));

		// set texture parameters
		texVolumeType.normalized = true;                      // access with normalized texture coordinates
		texVolumeType.filterMode = cudaFilterModeLinear;      // linear interpolation
		texVolumeType.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
		texVolumeType.addressMode[1] = cudaAddressModeClamp;

		// bind array to 3D texture
		checkCudaErrors(cudaBindTextureToArray(texVolumeType, d_volumeArray, channelDesc));

		// create transfer function texture
		float4 transferFunc[] =
		{
		{ 0.0, 0.0, 0.0, 0.0, },
		{ 1.0, 1.0, 1.0, 1.0, },
		};

		cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
		cudaArray *d_transferFuncArray;
		checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1));
		checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

		transferTex.filterMode = cudaFilterModeLinear;
		transferTex.normalized = true;    // access with normalized texture coordinates
		transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

		// Bind the array to the texture
		checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
	}

	void freeCudaBuffers_render_volume()
	{
		checkCudaErrors(cudaFreeArray(d_volumeArray));
		checkCudaErrors(cudaFreeArray(d_transferFuncArray));
	}

	void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
	{
		checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
	}


};	// extern "C"