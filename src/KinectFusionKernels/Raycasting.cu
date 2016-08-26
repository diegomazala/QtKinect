#ifndef _RAYCASTING_KERNEL_CU_
#define _RAYCASTING_KERNEL_CU_

#include "KinectFusionKernels.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


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
	float3 e21 = p2 - p1;		// compute edge 
	float3 e31 = p3 - p1;		// compute edge
	float3 n = cross(e21, e31);	// compute normal

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

	
	float3 hit[2];
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
			if (length(p - hit[0]) < length(p - hit[1]))
			{
				hit1Normal = hitNormal[0];
				hit2Normal = hitNormal[1];
			}
			else
			{
				hit1Normal = hitNormal[1];
				hit2Normal = hitNormal[0];
			}
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
	//void raycast(float origin[3], float direction[3], int voxel_count[3], int voxel_size[3])
	void raycast(float* origin_float3, float* direction_float3, int* voxel_count_int3, int* voxel_size_int3)
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

			//std::cout << "Voxel Intersected : " << voxel_index << std::endl;
		}
	
	}

};	// extern "C"


#endif // _RAYCASTING_KERNEL_CU_
