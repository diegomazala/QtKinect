#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_


#if _MSC_VER // this is defined when compiling with Visual Studio
#define CUDA_KERNELS_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define CUDA_KERNELS_API // XCode does not need annotating exported functions, so define is empty
#endif

#ifdef _WIN32
#include <windows.h>
#endif


extern "C"
{
	bool cublas_init();
	bool cublas_cleanup();
	bool cublas_matrix_mul(float *dev_C, const float *dev_A, const float *dev_B, const int m, const int k, const int n);

	void make_identity_4x4(float *mat);
	void print_matrix(const float *A, int nr_rows_A, int nr_cols_A);

	void matrix_mulf(float* mat_c, const float* mat_a, const float* mat_b, int m, int k, int n);

	void compute_depth_buffer(
		float* depth_buffer, 
		float* window_coords_2f,
		const float* world_points_4f, 
		unsigned int point_count, 
		const float* projection_mat4x4, 
		unsigned int window_width, 
		unsigned int window_height);

	void create_grid(
		unsigned int vol_size,
		unsigned int vx_size,
		float* grid_matrix,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f);

	void update_grid(
		unsigned int vol_size, 
		unsigned int vx_size, 
		float* grid_matrix, 
		float* grid_voxels, 
		float* depth_buffer,
		unsigned int window_width,
		unsigned int window_height);
};



template<typename Type>
static void perspective_matrix(Type out[16], Type fovy, Type aspect_ratio, Type near_plane, Type far_plane);


////usage example with eigen: matrix_mul(mat_C.data(), mat_A.data(), &vector_of_eigen_vector4[0][0], A.rows(), A.cols(), vector_of_eigen_vector4.size());
//template<typename Type>
//void matrix_mul(Type* mat_c, const Type* mat_a, const Type* mat_b, int m, int k, int n);
//{
//	// transfer to device 
//	thrust::device_vector<Type> d_a(&mat_a[0], &mat_a[0] + m * k);
//	thrust::device_vector<Type> d_b(&mat_b[0], &mat_b[0] + k * n);
//	thrust::device_vector<Type> d_c(&mat_c[0], &mat_c[0] + m * n);
//
//	// Multiply A and B on GPU
//	//cublas_matrix_mul(thrust::raw_pointer_cast(&d_a[0]), thrust::raw_pointer_cast(&d_b[0]), thrust::raw_pointer_cast(&d_c[0]), m, k, n);
//
//	thrust::copy(d_c.begin(), d_c.end(), &mat_c[0]);
//}



#endif // _CUDA_KERNELS_H_