#include "cuda_kernels.h"
#include <cmath>
#include <cstring>
#include <thrust/device_vector.h>





template<typename Type>
static void perspective_matrix(Type out[16], Type fovy, Type aspect_ratio, Type near_plane, Type far_plane)
{
	std::memset(out, 0, 16 * sizeof(Type));

	const Type y_scale = (Type)(1.0 / tan((fovy / 2.0)*(M_PI / 180.0)));
	const Type x_scale = y_scale / aspect_ratio;
	const Type depth_length = far_plane - near_plane;

	out[0] = x_scale;
	out[5] = y_scale;
	out[10] = -((far_plane + near_plane) / depth_length);
	out[14] = -1.0;
	out[11] = -((2 * near_plane * far_plane) / depth_length);

}






template<typename Type>
void matrix_mul(Type* mat_c, const Type* mat_a, const Type* mat_b, int m, int k, int n)
{
	// transfer to device 
	thrust::device_vector<Type> d_a(&mat_a[0], &mat_a[0] + m * k);
	thrust::device_vector<Type> d_b(&mat_b[0], &mat_b[0] + k * n);
	thrust::device_vector<Type> d_c(&mat_c[0], &mat_c[0] + m * n);

	// Multiply A and B on GPU
	cublas_matrix_mul(thrust::raw_pointer_cast(&d_a[0]), thrust::raw_pointer_cast(&d_b[0]), thrust::raw_pointer_cast(&d_c[0]), m, k, n);

	thrust::copy(d_c.begin(), d_c.end(), &mat_c[0]);
}
