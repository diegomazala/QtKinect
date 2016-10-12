#ifndef _KINECT_CUDA_KERNELS_CU_
#define _KINECT_CUDA_KERNELS_CU_

#include "KinectFusionKernels.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <thrust/device_vector.h>
#include <iostream>

#define MinTruncation 0.5f
#define MaxTruncation 1.1f
#define MaxWeight 10.0f

struct Ray
{
	float3 origin;   // origin
	float3 direction;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.direction;
	float3 tbot = invR * (boxmin - r.origin);
	float3 ttop = invR * (boxmax - r.origin);

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

struct grid_3d
{
	ushort3 voxel_count;
	ushort3 voxel_size;
	float* params_dev_ptr;
	float* params_host_ptr;

	grid_3d() : params_dev_ptr(nullptr), params_host_ptr(nullptr)
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
buffer_2d_f4	debug_buffer;
grid_3d			grid;
buffer_image_2d	image_buffer;


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
texture<ushort, 2, cudaReadModeElementType> depthTexture;
texture<float4, 2, cudaReadModeElementType> vertexTexture;
texture<float4, 2, cudaReadModeElementType> normalTexture;
texture<uchar4, 2, cudaReadModeNormalizedFloat> outputTexture;

#define PI 3.14159265359
__host__ __device__ float deg2rad(float deg) { return deg*PI / 180.0; }
__host__ __device__ float rad2deg(float rad) { return 180.0*rad / PI; }


typedef struct
{
	float4 m[3];
} float3x4;
__constant__ float3x4 camera_to_world_dev_matrix;  // inverse view matrix

__constant__ float4 lightData;  // xyz = direction, w = power

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}
__device__ uint rgbFloatToInt(float3 rgb)
{
	rgb.x = __saturatef(fabs(rgb.x));   // clamp to [0.0, 1.0]
	rgb.y = __saturatef(fabs(rgb.y));
	rgb.z = __saturatef(fabs(rgb.z));
	return (uint(255.0f) << 24) | (uint(rgb.z * 255.0f) << 16) | (uint(rgb.y * 255.0f) << 8) | uint(rgb.x * 255.0f);
}
	


static int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


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
	const float* grid_matrix_16f,
	const float* view_matrix_16f,
	const float* projection_matrix_16f,
	const ushort* depth_buffer,
	ushort window_width,
	ushort window_height)
{
	const ulong total_pixels = window_width * window_height;
	const ulong threadId = blockIdx.x * blockDim.x + threadIdx.x;
	const ulong z = threadId;

	if (z >= vx_count)
		return;

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

			if (pixel.x < 0 || pixel.y < 0 ||
				pixel.x > window_width || pixel.y > window_height)
				continue;

			// compute depth buffer pixel index in the array
			const int depth_pixel_index = pixel.y * window_width + pixel.x;

			// check if it is out of window size
			if (depth_pixel_index < 0 || depth_pixel_index > total_pixels - 1)
				continue;


			// get depth buffer value
			const float Dp = depth_buffer[depth_pixel_index] * 0.1f;


			// compute distance from vertex to camera
			float distance_vertex_camera = sqrt(
				pow(ti.x - vg[0], 2) +
				pow(ti.y - vg[1], 2) +
				pow(ti.z - vg[2], 2) +
				pow(ti.w - vg[3], 2));


			// compute signed distance function
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
	out_vertex->z = -depth;
	out_vertex->w = 1.0f;
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

	float depth = ((float)tex2D(depthTexture, x, y)) * 0.1f;
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

	const float4 vertex_uv = tex2D(vertexTexture, x, y);
	const float4 vertex_u1v = tex2D(vertexTexture, x + 1, y);
	const float4 vertex_uv1 = tex2D(vertexTexture, x, y + 1);

	const float4 n1 = vertex_u1v - vertex_uv;
	const float4 n2 = vertex_uv1 - vertex_uv;
	const float4 n = cross(n1, n2);

	out_normals[y * w + x] = normalize(n);
}



enum BoxFace
{
	Top = 0,
	Bottom,
	Front,
	Rear,
	Left,
	Right,
	Undefined
};


__device__ BoxFace box_face_from_normal(float3 normal)
{
	if (normal.z < -0.5f)
		return BoxFace::Front;

	if (normal.z > 0.5f)
		return BoxFace::Rear;

	if (normal.y < -0.5f)
		return BoxFace::Bottom;

	if (normal.y > 0.5f)
		return BoxFace::Top;

	if (normal.x < -0.5f)
		return BoxFace::Left;

	if (normal.x > 0.5f)
		return BoxFace::Right;

	return BoxFace::Undefined;
}


__device__ BoxFace box_face_in_face_out(BoxFace face_out)
{
	switch (face_out)
	{
		case BoxFace::Top: return BoxFace::Bottom;
		case BoxFace::Bottom: return BoxFace::Top;
		case BoxFace::Front: return BoxFace::Rear;
		case BoxFace::Rear: return BoxFace::Front;
		case BoxFace::Left: return BoxFace::Right;
		case BoxFace::Right: return BoxFace::Left;
		default:
		case BoxFace::Undefined: return BoxFace::Undefined;
	}
}


inline __device__ bool has_same_sign_tsdf(
	const float* voxels_params_2f, 
	ulong voxel_params_count,
	int prev_voxel_index, 
	int next_voxel_index)
{
	if (prev_voxel_index < 0 || prev_voxel_index > voxel_params_count * 2 - 1 ||
		next_voxel_index < 0 || next_voxel_index > voxel_params_count * 2 - 1)
		return false;

	return (voxels_params_2f[prev_voxel_index * 2] > 0 && voxels_params_2f[next_voxel_index * 2] > 0) ||
		(voxels_params_2f[prev_voxel_index * 2] < 0 && voxels_params_2f[next_voxel_index * 2] < 0);
}



inline __host__ __device__  int get_index_from_3d(const float3 hit, const ushort3 voxel_count, const ushort3 voxel_size)
{
	const int max_x = voxel_count.x * voxel_size.x;
	const int max_y = voxel_count.y * voxel_size.y;
	const int max_z = voxel_count.z * voxel_size.z;

	int x = (hit.x < max_x) ? (int)hit.x : max_x - 1;
	int y = (hit.y < max_y) ? (int)hit.y : max_y - 1;
	int z = (hit.z < max_z) ? (int)hit.z : max_z - 1;

	return z / voxel_size.z * voxel_count.x * voxel_count.y + y / voxel_size.y * voxel_count.y + x / voxel_size.x;
}


inline __device__ ushort3 index_3d_from_array(
	int array_index,
	ushort3 voxel_count,
	ushort3 voxel_size)
{
	return make_ushort3(
		int(fmod((float)array_index, (float)voxel_count.x)) * voxel_size.x,
		int(fmod((float)array_index / (float)voxel_count.y, (float)voxel_count.y)) * voxel_size.y,
		int(array_index / (voxel_count.x * voxel_count.y)) * voxel_size.z);
}

inline __device__ float3 compute_normal(
	const float3& p1,
	const float3& p2,
	const float3& p3)
{
	float3 u = p2 - p1;
	float3 v = p3 - p1;

	return normalize(cross(v, u));
}


// http://www.graphics.cornell.edu/pubs/1997/MT97.html
__device__ bool triangle_intersection(
	const float3& p,
	const float3& d,
	const float3& v0,
	const float3& v1,
	const float3& v2,
	float3& hit,
	float3& hit_normal)
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
		hit_normal = compute_normal(v0, v1, v2);
		return true;
	}
	else
		return false;
}

__device__ bool quad_intersection(
	const float3& p,
	const float3& d,
	const float3& p1,
	const float3& p2,
	const float3& p3,
	const float3& p4,
	float3& hit,
	float3& hit_normal)
{
	return (triangle_intersection(p, d, p1, p2, p3, hit, hit_normal)
		|| triangle_intersection(p, d, p3, p4, p1, hit, hit_normal));
}


__device__ int box_intersection(
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
	if (quad_intersection(p, dir, p1, p2, p3, p4, hit[hitCount], hitNormal[hitCount]))
	{
		hitCount++;
	}

	// check bottom
	if (quad_intersection(p, dir, p5, p8, p7, p6, hit[hitCount], hitNormal[hitCount]))
	{
		hitCount++;
	}

	// check front
	if (hitCount < 2 && quad_intersection(p, dir, p4, p3, p7, p8,
		hit[hitCount], hitNormal[hitCount]))
	{
		hitCount++;
	}

	// check back
	if (hitCount < 2 && quad_intersection(p, dir, p1, p5, p6, p2,
		hit[hitCount], hitNormal[hitCount]))
	{
		hitCount++;
	}

	// check left
	if (hitCount < 2 && quad_intersection(p, dir, p1, p4, p8, p5,
		hit[hitCount], hitNormal[hitCount]))
	{
		hitCount++;
	}

	// check right
	if (hitCount < 2 && quad_intersection(p, dir, p2, p6, p7, p3,
		hit[hitCount], hitNormal[hitCount]))
	{
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


struct FaceData
{
	BoxFace face;
	float3 hit;
	float3 normal;
	int voxel_index;
	float dist;
	__device__ FaceData(){}
	__device__ FaceData(BoxFace f, float3 ht, float3 n, int vx, float ds) : face(f), hit(ht), normal(n), voxel_index(vx), dist(ds){}
};


__device__ int face_intersections(
	float3 ray_origin,
	float3 ray_direction,
	ushort3 voxel_count,
	ushort3 voxel_size,
	int voxel_index,
	BoxFace face_in,
	float3 hit_in,
	BoxFace& face_out,
	float3& hit_out,
	float3& hit_normal,
	int& next_voxel_index)
{

	float3 hit;
	
	ushort3 ind_3d = index_3d_from_array(voxel_index, voxel_count, voxel_size);
	float3 voxel_pos = make_float3(ind_3d.x, ind_3d.y, ind_3d.z);
	//voxel_pos += (voxel_size * 0.5f);	// only if using the center of face

	BoxFace face = BoxFace::Undefined;
	FaceData face_list[6];
	int face_list_size = 0;

	face = BoxFace::Top;
	{
		float3 v1 = voxel_pos + make_float3(0, voxel_size.y, 0);
		float3 v2 = v1 + make_float3(voxel_size.x, 0, 0);
		float3 v3 = v1 + make_float3(voxel_size.x, 0, voxel_size.z);
		float3 v4 = v1 + make_float3(0, 0, voxel_size.z);
		if (quad_intersection(ray_origin, ray_direction, v1, v2, v3, v4, hit, hit_normal) && face != face_in)
		{
			face_out = face;
			hit_out = hit;
			float half_voxel_y = voxel_size.y * 0.5f;
			if (hit.y > voxel_count.y * voxel_size.y - half_voxel_y)
				next_voxel_index = -1;
			else
				next_voxel_index = voxel_index + voxel_count.y;

			float dist = length(hit_out - hit_in);
			face_list[face_list_size] = FaceData(face, hit, hit_normal, next_voxel_index, dist);
			face_list_size++;
			//return true;
		}
	}

	face = BoxFace::Bottom;
	{
		float3 v1 = voxel_pos;
		float3 v2 = v1 + make_float3(voxel_size.x, 0, 0);
		float3 v3 = v1 + make_float3(voxel_size.x, 0, voxel_size.z);
		float3 v4 = v1 + make_float3(0, 0, voxel_size.z);
		if (quad_intersection(ray_origin, ray_direction, v1, v2, v3, v4, hit, hit_normal) && face != face_in)
		{
			face_out = face;
			hit_out = hit;
			float half_voxel_y = voxel_size.y * 0.5f;
			if (hit.y < half_voxel_y)
				next_voxel_index = -1;
			else
				next_voxel_index = voxel_index - voxel_count.y;

			float dist = length(hit_out - hit_in);
			face_list[face_list_size] = FaceData(face, hit, hit_normal, next_voxel_index, dist);
			face_list_size++;
			//return true;
		}
	}

	face = BoxFace::Front;
	{
		float3 v1 = voxel_pos;
		float3 v2 = v1 + make_float3(voxel_size.x, 0, 0);
		float3 v3 = v1 + make_float3(voxel_size.x, voxel_size.y, 0);
		float3 v4 = v1 + make_float3(0, voxel_size.y, 0);
		if (quad_intersection(ray_origin, ray_direction, v1, v2, v3, v4, hit, hit_normal) && face != face_in)
		{
			face_out = face;
			hit_out = hit;
			float half_voxel_z = voxel_size.z * 0.5f;
			if (hit.z < half_voxel_z)
				next_voxel_index = -1;
			else
				next_voxel_index = voxel_index - voxel_count.x * voxel_count.y;

			float dist = length(hit_out - hit_in);
			face_list[face_list_size] = FaceData(face, hit, hit_normal, next_voxel_index, dist);
			face_list_size++;
			//return true;
		}
	}

	face = BoxFace::Rear;
	{
		float3 v1 = voxel_pos + make_float3(0, 0, voxel_size.z);
		float3 v2 = v1 + make_float3(voxel_size.x, 0, 0);
		float3 v3 = v1 + make_float3(voxel_size.x, voxel_size.y, 0);
		float3 v4 = v1 + make_float3(0, voxel_size.y, 0);
		if (quad_intersection(ray_origin, ray_direction, v1, v2, v3, v4, hit, hit_normal) && face != face_in)
		{
			face_out = face;
			hit_out = hit;
			float half_voxel_z = voxel_size.z * 0.5f;
			if (hit.z > voxel_count.z * voxel_size.z - half_voxel_z)
				next_voxel_index = -1;
			else
				next_voxel_index = voxel_index + voxel_count.x * voxel_count.y;

			float dist = length(hit_out - hit_in);
			face_list[face_list_size] = FaceData(face, hit, hit_normal, next_voxel_index, dist);
			face_list_size++;
			//return true;
		}
	}

	face = BoxFace::Left;
	{
		float3 v1 = voxel_pos;
		float3 v2 = v1 + make_float3(0, 0, voxel_size.z);
		float3 v3 = v1 + make_float3(0, voxel_size.y, voxel_size.z);
		float3 v4 = v1 + make_float3(0, voxel_size.y, 0);
		if (quad_intersection(ray_origin, ray_direction, v1, v2, v3, v4, hit, hit_normal) && face != face_in)
		{
			face_out = face;
			hit_out = hit;
			float half_voxel_x = voxel_size.x * 0.5f;
			if (hit.x < half_voxel_x)
				next_voxel_index = -1;
			else
				next_voxel_index = voxel_index - 1;

			float dist = length(hit_out - hit_in);
			face_list[face_list_size] = FaceData(face, hit, hit_normal, next_voxel_index, dist);
			face_list_size++;
			//return true;
		}
	}

	face = BoxFace::Right;
	{
		float3 v1 = voxel_pos + make_float3(voxel_size.x, 0, 0);
		float3 v2 = v1 + make_float3(0, 0, voxel_size.z);
		float3 v3 = v1 + make_float3(0, voxel_size.y, voxel_size.z);
		float3 v4 = v1 + make_float3(0, voxel_size.y, 0);
		if (quad_intersection(ray_origin, ray_direction, v1, v2, v3, v4, hit, hit_normal) && face != face_in)
		{
			face_out = face;
			hit_out = hit;
			float half_voxel_x = voxel_size.x * 0.5f;
			if (hit.x > voxel_count.x * voxel_size.x - half_voxel_x)
				next_voxel_index = -1;
			else
				next_voxel_index = voxel_index + 1;

			float dist = length(hit_out - hit_in);
			face_list[face_list_size] = FaceData(face, hit, hit_normal, next_voxel_index, dist);
			face_list_size++;
			//return true;
		}
	}

	if (face_list_size > 1)
	{
		float max_dist = -1;
		for (int i = 0; i < face_list_size;++i)
		{
			FaceData& d = face_list[i];
			if (d.dist > max_dist)
			{
				face_out = d.face;
				hit_out = d.hit;
				hit_normal = d.normal;
				next_voxel_index = d.voxel_index;
				max_dist = d.dist;
			}
		}
	}

	return face_list_size;
}


__device__ float3 mul_vec_dir_matrix(const float* M_3x4, const float3& v)
{
	return make_float3(
		dot(v, make_float3(M_3x4[0], M_3x4[4], M_3x4[8])),
		dot(v, make_float3(M_3x4[1], M_3x4[5], M_3x4[9])),
		dot(v, make_float3(M_3x4[2], M_3x4[6], M_3x4[10])));
}

__device__ float4 mul_vec_dir_matrix(const float* M_3x4, const float4& v)
{
	return make_float4(
		dot(v, make_float4(M_3x4[0], M_3x4[4], M_3x4[8], M_3x4[12])),
		dot(v, make_float4(M_3x4[1], M_3x4[5], M_3x4[9], M_3x4[13])),
		dot(v, make_float4(M_3x4[2], M_3x4[6], M_3x4[10], M_3x4[14])),
		1.0f);
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__device__ BoxFace raycast_face_volume(
	float3 ray_origin,
	float3 ray_direction,
	ushort3 voxel_count,
	ushort3 voxel_size,
	long& voxel_index,
	float3& hit)
{
	ushort3 volume_size = make_ushort3(
		voxel_count.x * voxel_size.x,
		voxel_count.y * voxel_size.y,
		voxel_count.z * voxel_size.z);

	float3 half_volume_size = make_float3(volume_size.x * 0.5f, volume_size.y * 0.5f, volume_size.z * 0.5f);
	float3 half_voxel_size = make_float3(voxel_size.x * 0.5f, voxel_size.y * 0.5f, voxel_size.z * 0.5f);


	float3 hit1;
	float3 hit2;
	float3 hit1_normal;
	float3 hit2_normal;

	//
	// Check intersection with the whole volume
	//
	int intersections_count = box_intersection(
		ray_origin,
		ray_direction,
		half_volume_size,	//volume_center,
		//Eigen::Matrix<Type, 3, 1>::Zero(),
		volume_size.x,
		volume_size.y,
		volume_size.z,
		hit1,
		hit2,
		hit1_normal,
		hit2_normal);

	if (intersections_count > 0)
	{
		voxel_index = get_index_from_3d(hit1, voxel_count, voxel_size);
		return box_face_from_normal(hit1_normal);
	}
	else
	{
		voxel_index = -1;
		return BoxFace::Undefined;
	}
}



__device__ int raycast_face_in_out(
	float3 ray_origin,
	float3 ray_direction,
	ushort3 voxel_count,
	ushort3 voxel_size,
	BoxFace& face_in,
	BoxFace& face_out,
	float3& hit_in,
	float3& hit_out,
	float3& hit_in_normal,
	float3& hit_out_normal)
{

	ushort3 volume_size = make_ushort3(
		voxel_count.x * voxel_size.x,
		voxel_count.y * voxel_size.y,
		voxel_count.z * voxel_size.z);

	float3 half_volume_size = make_float3(
		volume_size.x * 0.5f,
		volume_size.y * 0.5f,
		volume_size.z * 0.5f);

	float3 half_voxel_size = make_float3(
		voxel_count.x * 0.5f,
		voxel_count.y * 0.5f,
		voxel_count.z * 0.5f);

	float3 hit1;
	float3 hit2;
	float3 hit1_normal;
	float3 hit2_normal;

	//
	// Check intersection with the whole volume
	//
	int intersections_count = box_intersection(
		ray_origin,
		ray_direction,
		half_volume_size,	//volume_center,
		//Eigen::Matrix<Type, 3, 1>::Zero(),
		volume_size.x,
		volume_size.y,
		volume_size.z,
		hit1,
		hit2,
		hit1_normal,
		hit2_normal);

	if (intersections_count == 2)
	{
		face_in = box_face_from_normal(hit1_normal);
		face_out = box_face_from_normal(hit2_normal);

		hit_in = hit1;
		hit_out = hit2;
		
		hit_in_normal = hit1_normal;
		hit_out_normal = hit2_normal;
	}
	else if (intersections_count == 1)
	{
		face_in = face_out = box_face_from_normal(hit1_normal);
		hit_in = hit_out = hit1;

		hit_in_normal = hit1_normal;
	}
	else
	{
		face_in = face_out = BoxFace::Undefined;
	}

	return intersections_count;
}


__device__ int raycast_tsdf_volume(
	float3 ray_origin,
	float3 ray_direction,
	ushort3 voxel_count,
	ushort3 voxel_size,
	float* grid_voxels_params_2f,
	long voxels_zero_crossing_indices[2],
	float3& hit_normal)
{
	voxels_zero_crossing_indices[0] = voxels_zero_crossing_indices[1] = -1;

	ulong total_voxels = voxel_count.x * voxel_count.y * voxel_count.z;

	long voxel_index = -1;
	int next_voxel_index = -1;
	int intersections_count = 0;
	float3 hit_in;
	float3 hit_out;
	float3 hit_in_normal;
	float3 hit_out_normal;
	BoxFace face_in = BoxFace::Undefined;
	BoxFace face_out = BoxFace::Undefined;

	face_in = raycast_face_volume(ray_origin, ray_direction, voxel_count, voxel_size, voxel_index, hit_in);

	// the ray does not hits the volume
	if (face_in == BoxFace::Undefined || voxel_index < 0)
		return -1;

	intersections_count = raycast_face_in_out(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out, hit_in_normal, hit_out_normal);
	
	bool is_inside = intersections_count > 0;

	while (is_inside)
	{
		if (face_intersections(
			ray_origin, ray_direction, 
			voxel_count, voxel_size, voxel_index,
			face_in, hit_in, face_out, 
			hit_out, hit_in_normal,
			next_voxel_index))
		{
			if (next_voxel_index < 0)
			{
				is_inside = false;
			}
			else
			{
				intersections_count++;

				face_in = box_face_in_face_out(face_out);
				hit_in = hit_out;

				if (!has_same_sign_tsdf(grid_voxels_params_2f, total_voxels, voxel_index, next_voxel_index))
				{
					voxels_zero_crossing_indices[0] = voxel_index;
					voxels_zero_crossing_indices[1] = next_voxel_index;
					voxel_index = next_voxel_index;
					hit_normal = hit_in_normal;
					intersections_count = 2;
					return intersections_count;
				}
				else
				{
					voxel_index = next_voxel_index;
				}
			}
		}
		else
		{
			is_inside = false;
		}
	}

	return intersections_count;	// return hit count
}



__global__ void	raycast_kernel_gl(
	uint* out_image,
	ushort image_width,
	ushort image_height,
	ushort3 voxel_count,
	ushort3 voxel_size,
	float* grid_voxels_params_2f,
	float fov_scale,
	float aspect_ratio,
	float* camera_to_world_mat4x4)
{
	ulong x = blockIdx.x*blockDim.x + threadIdx.x;
	ulong y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= image_width || y >= image_height)
	{
		return;
	}

	// Convert from image space (in pixels) to screen space
	float u = (x / (float)image_width) * 2.0f - 1.0f;
	float v = (y / (float)image_height) * 2.0f - 1.0f;
	Ray eye_ray;
	eye_ray.origin = make_float3(mul(camera_to_world_dev_matrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	float3 screen_coord = normalize(make_float3(u, -v, -2.0f));
	eye_ray.direction = mul(camera_to_world_dev_matrix, screen_coord);

	long voxels_zero_crossing[2] = { -1, -1 };

#if 0
	//
	// Check intersection with the whole volume
	//
	ushort3 volume_size = make_ushort3(
		voxel_count.x * voxel_size.x,
		voxel_count.y * voxel_size.y,
		voxel_count.z * voxel_size.z);
	float3 half_volume_size = make_float3(volume_size.x * 0.5f, volume_size.y * 0.5f, volume_size.z * 0.5f);
	float3 hit1, hit2, hit1_normal, hit2_normal;
	int hit_count = box_intersection(
		eye_ray.origin,
		eye_ray.direction,
		//make_float3(0, 0, 0),	
		half_volume_size,	//volume_center,
		volume_size.x,
		volume_size.y,
		volume_size.z,
		hit1,
		hit2,
		hit1_normal,
		hit2_normal);
#else
	float3 hit_normal;
	int hit_count = raycast_tsdf_volume(
		eye_ray.origin,
		eye_ray.direction,
		voxel_count,
		voxel_size,
		grid_voxels_params_2f,
		voxels_zero_crossing,
		hit_normal);
#endif

	const float4 normal = tex2D(normalTexture, x, y);

	if (hit_count > 0)
	{
		if (voxels_zero_crossing[0] > -1 && voxels_zero_crossing[1] > -1)
			out_image[y * image_width + x] = rgbaFloatToInt(make_float4(fabs(normal.x), fabs(normal.y), fabs(normal.z), 1.f));
		else if (voxels_zero_crossing[0] > -1 || voxels_zero_crossing[1] > -1)
			out_image[y * image_width + x] = rgbaFloatToInt(make_float4(fabs(normal.x), fabs(normal.y), fabs(normal.z), 1.f));
		else
			out_image[y * image_width + x] = rgbaFloatToInt(make_float4(0.0f, 0.5f, 0.5f, 1.f));
	}
	else
	{
		out_image[y * image_width + x] = rgbaFloatToInt(make_float4(0.5f, 0.f, 0.f, 1.f));
	}

}


__global__ void	raycast_kernel(
	uchar4* out_image,
	ushort image_width,
	ushort image_height,
	ushort3 voxel_count,
	ushort3 voxel_size,
	float* grid_voxels_params_2f,
	float fov_scale,
	float aspect_ratio,
	float* camera_to_world_mat4x4)
{
	ulong x = blockIdx.x*blockDim.x + threadIdx.x;
	ulong y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= image_width || y >= image_height)
	{
		return;
	}

	// Convert from image space (in pixels) to screen space
	float u = (x / (float)image_width) * 2.0f - 1.0f;
	float v = (y / (float)image_height) * 2.0f - 1.0f;
	//Ray eye_ray;
	//eye_ray.origin = make_float3(mul(camera_to_world_dev_matrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	//float3 screen_coord = normalize(make_float3(u, -v, -2.0f));
	//eye_ray.direction = mul(camera_to_world_dev_matrix, screen_coord);


	// Convert from image space (in pixels) to screen space
	// Screen Space along X axis = [-aspect ratio, aspect ratio] 
	// Screen Space along Y axis = [-1, 1]
	float x2_norm = (2.f * float(x) + 0.5f) / (float)image_width;
	float y2_norm = (2.f * float(y) + 0.5f) / (float)image_height;
	float3 screen_coord = make_float3(
		(x2_norm - 1.f) * aspect_ratio * fov_scale,
		(1.f - y2_norm) * fov_scale,
		1.0f);

	//screen_coord = normalize(make_float3(u, -v, -2.0f));
	screen_coord = normalize(screen_coord);

	// ray origin
	float3 camera_pos = make_float3(mul_vec_dir_matrix(camera_to_world_mat4x4, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		//make_float3(camera_to_world_mat4x4[12], camera_to_world_mat4x4[13], camera_to_world_mat4x4[14]);

	// transform vector by matrix (no translation)
	// multDirMatrix
	float3 dir = mul_vec_dir_matrix(camera_to_world_mat4x4, screen_coord);
	// ray direction
	float3 direction = normalize(dir);
	
	long voxels_zero_crossing[2] = { -1, -1 };


#if 0
	//
	// Check intersection with the whole volume
	//
	ushort3 volume_size = make_ushort3(
		voxel_count.x * voxel_size.x,
		voxel_count.y * voxel_size.y,
		voxel_count.z * voxel_size.z);
	float3 half_volume_size = make_float3(volume_size.x * 0.5f, volume_size.y * 0.5f, volume_size.z * 0.5f);
	float3 hit1, hit2, hit1_normal, hit2_normal;
	int hit_count = box_intersection(
		camera_pos,
		direction,
		make_float3(0, 0, 0),	
		//half_volume_size,	//volume_center,
		volume_size.x,
		volume_size.y,
		volume_size.z,
		hit1,
		hit2,
		hit1_normal,
		hit2_normal);

	float3 N = hit2_normal;
#else

	float3 hit_normal;
	int hit_count = raycast_tsdf_volume(
		camera_pos,
		direction,
		voxel_count,
		voxel_size,
		grid_voxels_params_2f,
		voxels_zero_crossing,
		hit_normal);

	const float4 normal = tex2D(normalTexture, x, y);
	float3 N = hit_normal;
#endif
	

	
	//float3 diff_color = make_float3(1.f, 1.f, 1.f);
	float3 diff_color = make_float3(fabs(N.x), fabs(N.y), fabs(N.z));
	//float3 diff_color = make_float3(fabs(normal.x), fabs(normal.y), fabs(normal.z));
	float3 spec_color = make_float3(1.f, 1.f, 1.f);
	float spec_shininess = 1.0f;
	float3 E = direction;								// view direction
	//float3 L = normalize(make_float3(0.0f, -1.f, -1.f));	// light direction
	float3 L = normalize(make_float3(lightData.x, lightData.y, lightData.z));	// light direction
	float3 R = normalize(-reflect(L, N));
	float3 diff = diff_color * saturate(dot(N, L));
	float3 spec = spec_color * pow(saturate(dot(R, E)), spec_shininess);
	float3 color = clamp(diff + spec, 0.f, 1.f);

	if (hit_count > 0)
	{
		if (voxels_zero_crossing[0] > -1 && voxels_zero_crossing[1] > -1)
		{
			//out_image[y * image_width + x] = make_uchar4(0, 128, 128, 255);
			//out_image[y * image_width + x].x = uchar(fabs(N.x) * 255);
			//out_image[y * image_width + x].y = uchar(fabs(N.y) * 255);
			//out_image[y * image_width + x].z = uchar(fabs(N.z) * 255);
			
			out_image[y * image_width + x].x = uchar(color.x * 255);
			out_image[y * image_width + x].y = uchar(color.y * 255);
			out_image[y * image_width + x].z = uchar(color.z * 255);

			out_image[y * image_width + x].w = 255;
		}
		else
		{
			out_image[y * image_width + x] = make_uchar4(64, 0, 0, 255);
		}
	}
	else
	{
		out_image[y * image_width + x] = make_uchar4(32, 16, 0, 255);
	}

}



__global__ void	raycast_box_kernel(
	uint* out_image,
	ushort image_width,
	ushort image_height,
	float fov_scale,
	float aspect_ratio,
	ushort3 box_size)
{
	ulong x = blockIdx.x * blockDim.x + threadIdx.x;
	ulong y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= image_width || y >= image_height)
		return;

	// Convert from image space (in pixels) to screen space
	float u = (x / (float)image_width) * 2.0f - 1.0f;
	float v = (y / (float)image_height) * 2.0f - 1.0f;
	Ray eye_ray;
	eye_ray.origin = make_float3(mul(camera_to_world_dev_matrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	float3 screen_coord = normalize(make_float3(u, -v, -2.0f));
	eye_ray.direction = mul(camera_to_world_dev_matrix, screen_coord);

#if 1
	float3 half_box_size = make_float3(
		box_size.x * 0.5f,
		box_size.y * 0.5f,
		box_size.z * 0.5f);
	
	float3 hit1, hit2, hit1_normal, hit2_normal;
	int hit_count = box_intersection(
		eye_ray.origin,
		eye_ray.direction,
		//half_box_size,	//volume_center,
		make_float3(0, 0, 0),
		box_size.x,
		box_size.y,
		box_size.z,
		hit1,
		hit2,
		hit1_normal,
		hit2_normal);

	if (hit_count > 0)
		out_image[y * image_width + x] = rgbaFloatToInt(make_float4(0.f, 0.5f, 0.5f, 1.f));
	else
		out_image[y * image_width + x] = rgbaFloatToInt(make_float4(0.f, 0.0f, 0.0f, 1.f));
#else
	// find intersection with box
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);
	float tnear, tfar;
	int hit_count = intersectBox(eye_ray, boxMin, boxMax, &tnear, &tfar);

	if (hit_count)
		out_image[y * image_width + x] = rgbaFloatToInt(make_float4(0.f, 0.5f, 0.5f, 1.f));
	else
		out_image[y * image_width + x] = rgbaFloatToInt(make_float4(0.f, 0.0f, 0.0f, 1.f));
#endif
}


extern "C"
{
	void knt_set_light(float4 light_data)
	{
		checkCudaErrors(cudaMemcpyToSymbol(lightData, &light_data, sizeof(float4)));
	}

	void knt_cuda_setup(
		ushort vx_count,
		ushort vx_size,
		const float* grid_matrix_16f,
		const float* projection_matrix_16f,
		const float* projection_inverse_matrix_16f,
		float& grid_params_2f_host_ref,
		ushort depth_width,
		ushort depth_height,
		ushort min_depth,
		ushort max_depth,
		float4& vertex_4f_host_ref,
		float4& normal_4f_host_ref,
		ushort output_image_width,
		ushort output_image_height)
	{
		knt_set_light(make_float4(0.f, -1.f, -1.f, 1.f));

		grid.voxel_count = make_ushort3(vx_count, vx_count, vx_count);
		grid.voxel_size = make_ushort3(vx_size, vx_size, vx_size);
		grid.params_host_ptr = &grid_params_2f_host_ref;

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

		debug_buffer.width = output_image_width;
		debug_buffer.height = output_image_height;
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
		checkCudaErrors(cudaMalloc(&grid.params_dev_ptr, sizeof(float) * 2 * grid.total_voxels()));

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

		checkCudaErrors(
			cudaMallocPitch(
			&debug_buffer.dev_ptr,
			&debug_buffer.pitch,
			sizeof(float4) * debug_buffer.width,
			debug_buffer.height));

		checkCudaErrors(cudaDeviceSynchronize());
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

		checkCudaErrors(cudaFree(grid.params_dev_ptr));
		grid.params_dev_ptr			= nullptr;

		checkCudaErrors(cudaFree(depth_buffer.dev_ptr));
		depth_buffer.dev_ptr		= nullptr;

		checkCudaErrors(cudaFree(vertex_buffer.dev_ptr));
		vertex_buffer.dev_ptr = nullptr;

		checkCudaErrors(cudaFree(normal_buffer.dev_ptr));
		normal_buffer.dev_ptr = nullptr;

		checkCudaErrors(cudaFree(image_buffer.dev_ptr));
		image_buffer.dev_ptr = nullptr;

		checkCudaErrors(cudaFree(debug_buffer.dev_ptr));
		debug_buffer.dev_ptr = nullptr;

		
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaDeviceReset());
	}

	void knt_cuda_init_grid()
	{
		grid_init_kernel << < 1, grid.voxel_count.z >> >(
			grid.params_dev_ptr,
			grid.voxel_count.x
			);

		checkCudaErrors(cudaDeviceSynchronize());
	}

	void knt_cuda_grid_sample_test(float* grid_params_2f_host, size_t count)
	{
		checkCudaErrors(
			cudaMemcpy(
			grid.params_dev_ptr,
			grid_params_2f_host,
			sizeof(float2) * count,
			cudaMemcpyHostToDevice
			));

		checkCudaErrors(cudaDeviceSynchronize());
	}

	void knt_cuda_update_grid(const float* view_matrix_16f)
	{
		// it's identity. So, we are using the deault constructor 
		//checkCudaErrors(
		//	cudaMemcpy(
		//	view_matrix_dev_ptr,
		//	view_matrix_16f, //&matrix_identity[0],
		//	sizeof(float) * 16,
		//	cudaMemcpyHostToDevice
		//	));

		//cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
		//checkCudaErrors(cudaBindTexture2D(0, depthTexture, depth_buffer.dev_ptr, desc, depth_buffer.width, depth_buffer.height, depth_buffer.pitch));


		grid_update_kernel << < 1, grid.voxel_count.z >> >(
			grid.params_dev_ptr,
			grid.voxel_count.x,
			grid.voxel_size.x,
			grid_matrix_dev_ptr,
			projection_matrix_dev_ptr,
			view_matrix_dev_ptr,
			depth_buffer.dev_ptr,
			depth_buffer.width,
			depth_buffer.height
			);

		//checkCudaErrors(cudaUnbindTexture(depthTexture));
		checkCudaErrors(cudaDeviceSynchronize());
	}



	void knt_cuda_normal_estimation()
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
		checkCudaErrors(cudaBindTexture2D(0, depthTexture, depth_buffer.dev_ptr, desc, depth_buffer.width, depth_buffer.height, depth_buffer.pitch));

		cudaChannelFormatDesc desc_vertex = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture2D(0, vertexTexture, vertex_buffer.dev_ptr, desc_vertex, vertex_buffer.width, vertex_buffer.height, vertex_buffer.pitch));

		cudaChannelFormatDesc desc_normal = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture2D(0, normalTexture, normal_buffer.dev_ptr, desc_normal, normal_buffer.width, normal_buffer.height, normal_buffer.pitch));

		const dim3 threads_per_block(16, 16);
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

		checkCudaErrors(cudaDeviceSynchronize());

		normal_estimate_kernel << <  num_blocks, threads_per_block >> >(
			normal_buffer.dev_ptr,
			normal_buffer.width,
			normal_buffer.height);

		
		//checkCudaErrors(cudaUnbindTexture(depthTexture));
		//checkCudaErrors(cudaUnbindTexture(vertexTexture));
		//checkCudaErrors(cudaUnbindTexture(normalTexture));

		checkCudaErrors(cudaDeviceSynchronize());
	}



	void raycast_box(
		uint* image_data_ref,
		ushort width,
		ushort height,
		float fov_scale,
		float aspect_ratio,
		float* camera_to_world_mat3x4,
		ushort3 box_size)
	{
		checkCudaErrors(cudaMemcpyToSymbol(camera_to_world_dev_matrix, camera_to_world_mat3x4, sizeof(float4) * 3));

		const dim3 threads_per_block(16, 16);
		dim3 num_blocks = dim3(iDivUp(width, threads_per_block.x), iDivUp(height, threads_per_block.y));;

		raycast_box_kernel << < num_blocks, threads_per_block >> >(
			image_data_ref,
			width,
			height,
			fov_scale,
			aspect_ratio,
			box_size
			);

		checkCudaErrors(cudaDeviceSynchronize());
	}

	void knt_cuda_raycast_gl(
		uint* image_data_ref,
		ushort width,
		ushort height,
		float fovy,
		float aspect_ratio,
		const float* camera_to_world_mat3x4)
	{
		checkCudaErrors(cudaMemcpyToSymbol(camera_to_world_dev_matrix, camera_to_world_mat3x4, sizeof(float4) * 3));

		float fov_scale = -tan(deg2rad(fovy * 0.5f));

		const dim3 threads_per_block(16, 16);
		dim3 num_blocks = dim3(iDivUp(width, threads_per_block.x), iDivUp(height, threads_per_block.y));;

		raycast_kernel_gl << <  num_blocks, threads_per_block >> >(
			image_data_ref,
			image_buffer.width,
			image_buffer.height,
			grid.voxel_count,
			grid.voxel_size,
			grid.params_dev_ptr,
			fov_scale,
			aspect_ratio,
			camera_to_world_matrix_dev_ptr
			);

		checkCudaErrors(cudaDeviceSynchronize());
	}


	void knt_cuda_raycast(
		float fovy,
		float aspect_ratio,
		const float* camera_to_world_matrix_16f)
	{
		float fov_scale = tan(deg2rad(fovy * 0.5f));

		checkCudaErrors(cudaMemcpyToSymbol(camera_to_world_dev_matrix, camera_to_world_matrix_16f, sizeof(float4) * 3));

		checkCudaErrors(
			cudaMemcpy(
			camera_to_world_matrix_dev_ptr,
			camera_to_world_matrix_16f,
			sizeof(float) * 16,
			cudaMemcpyHostToDevice
			));

		const dim3 threads_per_block(16, 16);
		dim3 num_blocks = dim3(iDivUp(image_buffer.width, threads_per_block.x), iDivUp(image_buffer.height, threads_per_block.y));;

		raycast_kernel << <  num_blocks, threads_per_block >> >(
			image_buffer.dev_ptr,
			image_buffer.width,
			image_buffer.height,
			grid.voxel_count,
			grid.voxel_size,
			grid.params_dev_ptr,
			fov_scale,
			aspect_ratio,
			camera_to_world_matrix_dev_ptr
			);

		checkCudaErrors(cudaDeviceSynchronize());
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
				sizeof(float4) * vertex_buffer.width,
				vertex_buffer.dev_ptr,
				vertex_buffer.pitch,
				sizeof(float4) * vertex_buffer.width,
				vertex_buffer.height,
				cudaMemcpyDeviceToHost);

		if (normal_buffer.host_ptr != nullptr)
			cudaMemcpy2D(
				normal_buffer.host_ptr,
				sizeof(float4) * normal_buffer.width,
				normal_buffer.dev_ptr,
				normal_buffer.pitch,
				sizeof(float4) * normal_buffer.width,
				normal_buffer.height,
				cudaMemcpyDeviceToHost);
	}

	void knt_cuda_grid_params_copy_device_to_host()
	{
		checkCudaErrors(
			cudaMemcpy(
			grid.params_host_ptr,
			grid.params_dev_ptr,
			sizeof(float) * 2 * grid.total_voxels(),
			cudaMemcpyDeviceToHost
			));
	}

	void knt_cuda_copy_image_device_to_host(uchar4& image_host_ptr)
	{
		cudaMemcpy2D(
			&image_host_ptr,
			sizeof(uchar4) * image_buffer.width,
			image_buffer.dev_ptr,
			image_buffer.pitch,
			sizeof(uchar4) * image_buffer.width,
			image_buffer.height,
			cudaMemcpyDeviceToHost);

		//if (debug_buffer.host_ptr != nullptr)
		//	cudaMemcpy2D(
		//	debug_buffer.host_ptr,
		//	sizeof(float4) * debug_buffer.width,
		//	debug_buffer.dev_ptr,
		//	debug_buffer.pitch,
		//	sizeof(float4) * debug_buffer.width,
		//	debug_buffer.height,
		//	cudaMemcpyDeviceToHost);
	}
	
	
}

#endif // #ifndef _KINECT_CUDA_KERNELS_CU_
