#ifndef _RAYCASTING_KERNEL_CU_
#define _RAYCASTING_KERNEL_CU_

#include "KinectFusionKernels.h"
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <helper_cuda.h>
#include <helper_math.h>


#define PI 3.14159265359

__host__ __device__ float deg2rad(float deg) { return deg*PI / 180.0;}
__host__ __device__ float rad2deg(float rad) { return 180.0*rad / PI; }





static int3 get_index_3d_from_array(
	int array_index,
	const int3& voxel_count)
{
	return make_int3(
		int(std::fmod(array_index, voxel_count.x)),
		int(std::fmod(array_index / voxel_count.y, voxel_count.y)),
		int(array_index / (voxel_count.x * voxel_count.y)));
}

inline __host__ __device__ int get_index_from_3d_volume(int3 pt, int3 voxel_count)
{
	return pt.z * voxel_count.x * voxel_count.y + pt.y * voxel_count.y + pt.x;
}

// 
// Face Index
// 0-Top, 1-Bottom, 2-Front, 3-Back, 4-Left, 5-Right
//
static int get_index_from_box_face(int face, int last_index, int3 voxel_count)
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
	void raycast_one(
		const float* origin_float3, 
		const float* direction_float3, 
		const int* voxel_count_int3, 
		const int* voxel_size_int3)
	{
		float3 origin = make_float3(origin_float3[0], origin_float3[1], origin_float3[2]);
		float3 direction = make_float3(direction_float3[0], direction_float3[1], direction_float3[2]);
		int3 voxel_count = make_int3(voxel_count_int3[0], voxel_count_int3[1], voxel_count_int3[2]);

		int3 voxel_size = make_int3(voxel_size_int3[0], voxel_size_int3[1], voxel_size_int3[2]);
		float3 half_voxel_size = make_float3(
			voxel_size.x * 0.5f,
			voxel_size.y * 0.5f,
			voxel_size.z * 0.5f);

		int3 volume_size = voxel_count * voxel_size;
		float3 half_volume_size = make_float3(
			volume_size.x * 0.5f,
			volume_size.y * 0.5f,
			volume_size.z * 0.5f);

		int total_voxels = voxel_count.x * voxel_count.y * voxel_count.z;

		float3 hit1;
		float3 hit2;
		float3 hit1_normal;
		float3 hit2_normal;

	//	std::cout << volume_size.x << ' ' << volume_size.y << ' ' << volume_size.z << std::endl;


		//
		// Check intersection with the whole volume
		//
		int intersections_count = box_intersection(
			origin,
			direction,
			half_volume_size,	//volume_center,
			volume_size.x,
			volume_size.y,
			volume_size.z,
			hit1,
			hit2,
			hit1_normal,
			hit2_normal);

		int3 hit_int = make_int3(hit1.x, hit1.y, hit1.z);
		int voxel_index = get_index_from_3d_volume(hit_int, voxel_count);
		float3 last_voxel = make_float3(hit_int.x, hit_int.y, hit_int.z);


		int loop_count = 0;
		std::cout << "First Intersected : " << voxel_index << std::endl;
	
		// 
		// Check intersection with each box inside of volume
		// 
		while (voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x * voxel_count.y))
		{

			int face = -1;
			intersections_count = box_intersection(
				origin,
				direction,
				last_voxel + half_voxel_size,
				voxel_size.x,
				voxel_size.y,
				voxel_size.z,
				hit1_normal,
				hit2_normal,
				face);


			voxel_index = get_index_from_box_face(face, voxel_index, voxel_count);
			int3 last_voxel_index = get_index_3d_from_array(voxel_index, voxel_count);
			loop_count++;

			std::cout << "Voxel Intersected : " << voxel_index << std::endl;
		}
	
	}

	__device__ float3 mul_vec_dir_matrix(const float* M_3x4, const float3& v)
	{
		return make_float3(
			dot(v, make_float3(M_3x4[0], M_3x4[4], M_3x4[8])),
			dot(v, make_float3(M_3x4[1], M_3x4[5], M_3x4[9])),
			dot(v, make_float3(M_3x4[2], M_3x4[6], M_3x4[10])));
	}

	__global__ void	raycast_box_kernel(
		uchar3 *d_output_image,
		uint image_width,
		uint image_height,
		uint box_size,
		float fovy,
		const float* camera_to_world_mat4x4,
		const float* box_transf_mat4x4)
	{
		float3 camera_pos = make_float3(camera_to_world_mat4x4[12], camera_to_world_mat4x4[13], camera_to_world_mat4x4[14]);

		float scale = tan(deg2rad(fovy * 0.5f));
		float aspect_ratio = (float)image_width / (float)image_height;

		float near_plane = 0.3f;
		float far_plane = 512.0f;

		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;

		if ((x >= image_width) || (y >= image_height)) 
			return;

		// Convert from image space (in pixels) to screen space
		// Screen Space alon X axis = [-aspect ratio, aspect ratio] 
		// Screen Space alon Y axis = [-1, 1]
		float3 screen_coord = make_float3(
			(2 * (x + 0.5f) / (float)image_width - 1) * aspect_ratio * scale,
			(1 - 2 * (y + 0.5f) / (float)image_height) * scale,
			-1.0f);

		// transform vector by matrix (no translation)
		// multDirMatrix
		float3 dir = mul_vec_dir_matrix(camera_to_world_mat4x4, screen_coord);
		

		//float3 direction = normalize(screen_coord - camera_pos);
		float3 direction = normalize(dir);	

#if 1


		float3 hit;
		float3 v1 = make_float3(0.0f, -1.0f, -2.0f);
		float3 v2 = make_float3(0.0f, 1.0f, -4.0f);
		float3 v3 = make_float3(-1.0f, -1.0f, -3.0f);
		float3 v4 = make_float3(0.0f, -1.0f, -2.0f);
		float3 v5 = make_float3(0.0f, 1.0f, -4.0f);
		float3 v6 = make_float3(1.0f, -1.0f, -3.0f);

		float3 diff_color = make_float3(1, 0, 0);
		float3 spec_color = make_float3(1, 1, 0);
		float spec_shininess = 1.0f;
		float3 E = make_float3(0, 0, -1);				// view direction
		float3 L = normalize(make_float3(0.2, -1, -1));	// light direction
		float3 N[2] = {
			compute_normal(v1, v2, v3),
			compute_normal(v4, v5, v6) };
		float3 R[2] = {
			normalize(-reflect(L, N[0])),
			normalize(-reflect(L, N[1])) };

		bool intersec[2] = {
			triangle_intersection(camera_pos, direction, v1, v2, v3, hit),
			triangle_intersection(camera_pos, direction, v4, v5, v6, hit) };

		// clear pixel
		d_output_image[y * image_width + x] = make_uchar3(8, 16, 32);

		for (int i = 0; i < 2; ++i)
		{
			if (intersec[i])
			{
				float3 diff = diff_color * saturate(dot(N[i], L));
				float3 spec = spec_color * pow(saturate(dot(R[i], E)), spec_shininess);

				float3 color = clamp(diff + spec, 0.f, 1.f);

				//d_output_image[y * image_width + x] = make_uchar3(255, 255, 255);
				//d_output_image[y * image_width + x] = make_uchar3(N[i].x * 255, N[i].y * 255, N[i].z * 255);
				d_output_image[y * image_width + x] = make_uchar3(color.x * 255, color.y * 255, color.z * 255);
			}
		}


		//d_output_image[y * image_width + x].x = (uchar)(((direction.x + 1) * 0.5) * 255);
		//d_output_image[y * image_width + x].y = (uchar)(((direction.y + 1) * 0.5) * 255);
		//d_output_image[y * image_width + x].z = (uchar)(((direction.z + 1) * 0.5) * 255);
#else

		//float3 box_center = make_float3(box_size * 0.5f);
		float3 box_center = make_float3(0.0f, 0.0f, 1.0f);
		float3 hit1;
		float3 hit2;
		float3 hit1Normal;
		float3 hit2Normal;
		int face = -1;

		int intersections = box_intersection(
			camera_pos,
			direction,
			box_center,
			box_size,
			box_size,
			box_size,
			hit1,
			hit1,
			hit1Normal,
			hit2Normal
			);

		if (intersections > 0)
		{
			d_output_image[y * image_width + x] = make_uchar3(0, 255, 255);
		}
		else
		{
			d_output_image[y * image_width + x] = make_uchar3(64, 32, 32);
		}
#endif

		//uchar xx = uchar((float)x / (float)image_width * 255);
		//uchar yy = uchar((float)y / (float)image_height * 255);

		//uchar xx = uchar(window_coord_norm.x * 255);
		//uchar yy = uchar(window_coord_norm.y * 255);

		//uchar xx = uchar(screen_coord.x * 255);
		//uchar yy = uchar(screen_coord.y * 255);

		//uchar xx = uchar(ndc.x * 255);
		//uchar yy = uchar(ndc.y * 255);

		// write output color
		//d_output_image[y * image_width + x] = make_uchar3(0, 255, 255);
		//d_output_image[y * image_width + x] = make_uchar3(xx, yy, 0);
	}

	void raycast_box(
		void* image_rgb_output_uchar3,
		uint width,
		uint height,
		uint box_size,
		float fovy,
		const float* camera_to_world_mat4f,
		const float* box_transf_mat4f)
	{
		std::cout << "fov " << fovy << std::endl;

		thrust::device_vector<uchar3> d_image_rgb = thrust::device_vector<uchar3>(width * height);
		thrust::device_vector<float> d_camera_to_world_mat4f = thrust::device_vector<float>(&camera_to_world_mat4f[0], &camera_to_world_mat4f[0] + 16);
		thrust::device_vector<float> d_box_transform_mat4f = thrust::device_vector<float>(&box_transf_mat4f[0], &box_transf_mat4f[0] + 16);

		const dim3 threads_per_block(32, 32);
		const dim3 num_blocks = dim3(iDivUp(width, threads_per_block.x), iDivUp(height, threads_per_block.y));

		raycast_box_kernel << <  num_blocks, threads_per_block >> >(
			thrust::raw_pointer_cast(&d_image_rgb[0]),
			width,
			height,
			box_size,
			fovy,
			thrust::raw_pointer_cast(&d_camera_to_world_mat4f[0]),
			thrust::raw_pointer_cast(&d_box_transform_mat4f[0])
			);

		thrust::copy(d_image_rgb.begin(), d_image_rgb.end(), (uchar3*)image_rgb_output_uchar3);
	}


};	// extern "C"


#endif // _RAYCASTING_KERNEL_CU_
