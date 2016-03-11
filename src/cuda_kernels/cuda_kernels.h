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


#define USE_RGBA 0

#include "helper_timer.h"
#include <cuda_runtime.h>

typedef unsigned char	uchar;
typedef unsigned int	uint;
typedef unsigned short	ushort;





extern "C"
{

	double bilateralFilterRGBA(
		unsigned int *d_dest, 
		int width, 
		int height,
		float e_d, 
		int radius, 
		int iterations,
		StopWatchInterface *timer);


	double bilateralFilterGray(
		uchar* dOutputImage,
		uchar* dInputImage,
		int width, 
		int height,
		size_t pitch,
		float e_d, 
		int radius, 
		int iterations,
		StopWatchInterface *timer);


	double bilateralFilter_ushort(
		ushort *dOutputImage,
		ushort *dInputImage,
		int width, 
		int height, 
		size_t pitch,
		ushort max_depth,
		float e_d, 
		int radius, 
		int iterations,
		StopWatchInterface *timer);

	
	void passthrough_texture_ushort(
		ushort* dOutputImage,
		ushort* dInputImage,
		int width,
		int height,
		size_t pitch);



	extern "C" void updateGaussian(float delta, int radius);




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
		float* grid_matrix_inv,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f,
		float* depth_buffer,
		const float* projection_mat16f,
		unsigned int window_width,
		unsigned int window_height);


	void grid_init(
		unsigned short vol_size,
		unsigned short vx_size,
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f,
		const float* grid_matrix_16f,
		const float* grid_matrix_inv_16f,
		const float* projection_matrix_16f
		);


	void grid_update(
		const float* view_matrix_16f,
		const float* view_matrix_inv_16f,
		const float* depth_buffer,
		unsigned short window_width,
		unsigned short window_height
		);


	void grid_get_data(
		float* grid_voxels_points_4f,
		float* grid_voxels_params_2f
		);
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