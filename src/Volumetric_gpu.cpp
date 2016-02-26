/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/* Example of integrating CUDA functions into an existing
* application / framework.
* CPP code representing the existing application / framework.
* Compiled with default CPP compiler.
*/

// includes, system
#include <iostream>
#include <stdlib.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_kernels/cuda_kernels.h"
#include "Volumetric_helper.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
//extern "C" bool runTest(const int argc, const char **argv,
//	char *data, int2 *data_int2, unsigned int len);



int test_projection_gpu()
{
	const Eigen::Vector4f p3d(-0.5f, -0.5f, -0.88f, 1.0f);
	const Eigen::Vector2f pixel(285.71f, 5.71f);

	const float window_width = 1280.0f;
	const float window_height = 720.0f;
	const float near_plane = 0.1f;
	const float far_plane = 100.0f;
	const float fovy = 60.0f;
	const float aspect_ratio = window_width / window_height;
	const float y_scale = (float)(1.0 / tan((fovy / 2.0)*(M_PI / 180.0)));
	const float x_scale = y_scale / aspect_ratio;

	const Eigen::Matrix4f Proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

	Eigen::Vector2f window_coord = vertex_to_window_coord(p3d, Proj, window_width, window_height);

	std::cout << Proj << std::endl << std::endl;

	std::cout << (Proj * p3d).transpose() << std::endl << std::endl;

	std::cout << window_coord.transpose() << std::endl << std::endl;

	if (pixel.isApprox(window_coord, 0.01f))
	{
		std::cout << "Test CPU OK" << std::endl;
	}
	else
	{
		std::cout << "Test CPU FAIL" << std::endl;
	}

	return 0;
}




void test_grid(int argc, char **argv)
{
	const std::string filepath = argv[1];
	const int vol_size = atoi(argv[2]);
	const int vx_size = atoi(argv[3]);
	const int cloud_count = atoi(argv[4]);
	const int rot_interval = atoi(argv[5]);



	//
	// Creating volume
	//
	Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	//
	const int total_voxels = (volume_size.x() / voxel_size.x() + 1) *
		(volume_size.y() / voxel_size.y() + 1) *
		(volume_size.z() / voxel_size.z() + 1);

	std::cout << "Total Voxels: ("
		<< (volume_size.x() / voxel_size.x() + 1) << ", "
		<< (volume_size.y() / voxel_size.y() + 1) << ", "
		<< (volume_size.z() / voxel_size.z() + 1) << ") =  "
		<< total_voxels	<< std::endl << std::endl;
	//
	Eigen::Affine3f grid_affine = Eigen::Affine3f::Identity();
	grid_affine.translate(Eigen::Vector3f(0, 0, -256));
	grid_affine.scale(Eigen::Vector3f(1, 1, -1));	// z is negative inside of screen
	Grid grid(volume_size, voxel_size, grid_affine.matrix().cast<double>());

	//std::cout << "Grid Transformation: " << std::endl << grid_affine.matrix() << std::endl;

	//std::cout << std::endl << "update " << std::endl << std::endl;
	//const std::size_t slice_size = (grid.voxel_count.x() + 1) * (grid.voxel_count.y() + 1);
	//for (auto it_volume = grid.data.begin(); it_volume != grid.data.end(); it_volume += slice_size)
	//{
	//	auto z_slice_begin = it_volume;
	//	auto z_slice_end = it_volume + slice_size;

	//	for (auto it = z_slice_begin; it != z_slice_end; ++it)
	//	{
	//		std::cout << it->point.transpose() << std::endl;
	//	}
	//}

	//std::cout << std::endl << std::endl;



	int threadId = 0;
	const int half_vol_size = vol_size / 2;

	int voxel_step = vol_size / vx_size;

	uint3 dim;
	dim.x = vol_size / vx_size + 1;
	dim.y = vol_size / vx_size + 1;
	dim.z = vol_size / vx_size + 1;

	const int z_slice_size = (vol_size / vx_size + 1) * (vol_size / vx_size + 1);	// x_size * y_size
	const int z_slice_index = threadId * z_slice_size * 4;


	std::vector<Eigen::Vector4f> grid_voxels_points(total_voxels);
	std::vector<Eigen::Vector2f> grid_voxels_params(total_voxels);

	Eigen::Matrix4f to_origin = Eigen::Matrix4f::Identity();
	to_origin.col(3) << -(volume_size.x() / 2.0), -(volume_size.y() / 2.0), -(volume_size.z() / 2.0), 1.0;	// set translate

	const int m = 4;
	const int k = 4;
	for (int z = 0; z <= voxel_step; ++z)
	{
		for (int y = 0; y <= voxel_step; ++y)
		{
			//printf("\nz,y=%d,%d\n", z,y);
			for (int x = 0; x <= voxel_step; ++x)
			{
				float vg[4] = {
					float(x * half_vol_size - half_vol_size),
					float(y * half_vol_size - half_vol_size),
					float(threadId * half_vol_size - half_vol_size),
					1.0f };

				float v[4] = { 0, 0, 0, 0 };

				for (int i = 0; i < m; i++)
					for (int j = 0; j < k; j++)
						v[j] += grid_affine.matrix()(i * m + j) * vg[i];

				std::cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << std::endl;

				Eigen::Vector4f vpt(vg[0], vg[1], vg[2], vg[3]);
				//std::cout << (grid_affine.matrix() * vpt).transpose() << std::endl;

				int index = x + dim.x * (y + dim.z * z);

//				printf("-->%d %d : \t %f,\t %f,\t %f\n",
//					index, ((int)threadId), 
//					v.x, v.y, v.z);

				Eigen::Vector4f pt(x * half_vol_size, y * half_vol_size, z * half_vol_size, 1.0f);
//				std::cout << (grid_affine.matrix() * to_origin * pt).transpose() << std::endl << std::endl;

				grid_voxels_points.at(index) = (grid_affine.matrix() * to_origin * pt);
			}
		}
		threadId++;
	}
	
	std::cout << std::endl << std::endl;


	std::vector<float> depth_buffer(window_width * window_height);
	Eigen::Matrix4f grid_matrix = grid_affine.matrix().cast<float>();
	//update_grid(vol_size, vx_size, grid_matrix.data(), &grid_voxels[0][0], depth_buffer.data(), window_width, window_height);
	//create_grid(vol_size, vx_size, grid_matrix.data(), &grid_voxels_points[0][0], &grid_voxels_params[0][0]);

	std::cout << "------- // --------" << std::endl;
	for (const auto v : grid_voxels_points)
	{
		std::cout << v.transpose() << std::endl;
	}
	std::cout << "------- // --------" << std::endl;
}


void test_mult_mat()
{
	const int m = 4;
	const int k = 4;
	const int n = 1;

	Eigen::Affine3f translate = Eigen::Affine3f::Identity();
	translate.translate(Eigen::Vector3f(0, 0, -256));
	translate.scale(Eigen::Vector3f(1, 1, -1));	// z is negative inside of screen

	

	Eigen::Matrix<float, m, k> identity = translate.matrix(); //Eigen::Matrix<float, m, k>::Identity();
	Eigen::Matrix<float, k, n> vec = {1, 1, 1, 1};

	Eigen::Matrix4f id = Eigen::Matrix<float, m, k>::Identity();
	std::cout << "row major: " << identity.IsRowMajor << std::endl;
	std::cout << "row major: " << id.IsRowMajor << std::endl;

	Eigen::Matrix<float, m, n> res = Eigen::Matrix<float, m, n>::Zero();

	std::cout << identity.rows() << ", " << identity.cols() << std::endl;
	std::cout << vec.rows() << ", " << vec.cols() << std::endl;
	std::cout << res.rows() << ", " << res.cols() << std::endl;

	std::cout << identity << std::endl << std::endl;
	std::cout << vec << std::endl << std::endl;
	std::cout << res << std::endl << std::endl;

	res = identity * vec;
	std::cout << "res size: "<< res.rows() << ", " << res.cols() << std::endl;
	std::cout << "res \n"<< res << std::endl << std::endl;
	std::cout << "res \n" << (identity.transpose() * vec) << std::endl << std::endl;
	res = Eigen::Matrix<float, m, n>::Zero();



	float *a = identity.data(); // identity.data();
	float *b = &vec[0]; // identity.data();
	float *c = &res[0];

	//float a[m * k] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	//float b[k * n] = { 1, 1, 1, 1 };
	//float c[m * n] = {0,0,0,0};

	
	// colum major
	res = Eigen::Matrix<float, m, n>::Zero();
	c = &res[0];
	for (int i = 0; i < m; i++)
		for (int j = 0; j < k; j++)
			c[j] += a[i * m + j] * b[j];


	std::cout << "\n-------------------------\n";
	print_matrix(c, m, n);
	std::cout << "-------------------------\n";


	// row major
	res = Eigen::Matrix<float, m, n>::Zero();
	for (int i = 0; i < m; i++) 
		for (int j = 0; j < k; j++) 
			c[i] += a[i * m + j] * b[j];


	std::cout << "\n-------------------------\n";
	print_matrix(c, m, n);
	std::cout << "-------------------------\n";
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	if (argc < 6)
	{
		std::cerr << "Missing parameters. Abort."
			<< std::endl
			<< "Usage:  ./Volumetricd.exe ../../data/monkey.obj 256 8 2 90"
			<< std::endl;
		return EXIT_FAILURE;
	}


	Timer timer;
	const std::string filepath = argv[1];
	const int vol_size = atoi(argv[2]);
	const int vx_size = atoi(argv[3]);
	const int cloud_count = atoi(argv[4]);
	const int rot_interval = atoi(argv[5]);


	//
	// Importing .obj
	//
	timer.start();
	std::vector<Eigen::Vector4f> points3DOrig;
	import_obj(filepath, points3DOrig);
	timer.print_interval("Importing monkey    : ");
	std::cout << filepath << " point count  : " << points3DOrig.size() << std::endl;

	const std::size_t point_count = points3DOrig.size();
	const std::size_t pixel_count = static_cast<const std::size_t>(window_width * window_height);


	Eigen::Matrix4f K = perspective_matrix<float>(fov_y, aspect_ratio, near_plane, far_plane);
	std::pair<Eigen::Matrix4f, Eigen::Matrix4f>	T(Eigen::Matrix4f::Identity(), Eigen::Matrix4f::Identity());

	// 
	// Translating and rotating monkey point cloud 
	std::pair<std::vector<Eigen::Vector4f>, std::vector<Eigen::Vector4f>> cloud;
	cloud.first.resize(points3DOrig.size());
	cloud.second.resize(points3DOrig.size());
	//
	Eigen::Affine3f rotate = Eigen::Affine3f::Identity();
	//rotate.rotate(Eigen::AngleAxisf(DegToRad(90), Eigen::Vector3f::UnitY()));
	Eigen::Affine3f translate = Eigen::Affine3f::Identity();
	translate.translate(Eigen::Vector3f(0, 0, -256));


	Eigen::Matrix4f identity_mat4f = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f trans_rot = translate.matrix() * rotate.matrix();

	// 
	// Compute first cloud
	//
	timer.start();
	//matrix_mulf(&cloud.first[0][0], translate.matrix().data(), &points3DOrig[0][0], translate.matrix().rows(), translate.matrix().cols(), points3DOrig.size());
	matrix_mulf(&cloud.first[0][0], trans_rot.data(), &points3DOrig[0][0], trans_rot.rows(), trans_rot.cols(), points3DOrig.size());
	//matrix_mulf(&cloud.second[0][0], trans_rot.data(), &points3DOrig[0][0], trans_rot.rows(), trans_rot.cols(), points3DOrig.size());
	timer.print_interval("GPU compute first cloud : ");

	//export_obj("../../data/out_gpu_1.obj", cloud.first);
	//export_obj("../../data/out_gpu_2.obj", cloud.second);

	std::pair<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>> window_coords;
	window_coords.first.resize(point_count);
	window_coords.second.resize(point_count);
	
	std::pair<std::vector<float>, std::vector<float>> depth_buffer;
	depth_buffer.first.resize(pixel_count, far_plane);
	depth_buffer.second.resize(pixel_count, far_plane);

	timer.start();
	compute_depth_buffer(
		&depth_buffer.first.data()[0],
		&window_coords.first[0][0],
		&cloud.first[0][0], 
		cloud.first.size(), 
		K.data(), 
		window_width, 
		window_height);
	timer.print_interval("GPU compute depth 1     : ");

	timer.start();
	compute_depth_buffer(
		&depth_buffer.second.data()[0],
		&window_coords.first[0][0],
		&cloud.second[0][0],
		cloud.second.size(),
		K.data(),
		window_width,
		window_height);
	timer.print_interval("GPU compute depth 2     : ");

	//export_depth_buffer("../../data/gpu_depth_buffer_1.obj", depth_buffer.first);
	//export_depth_buffer("../../data/gpu_depth_buffer_2.obj", depth_buffer.second);
	

	//
	// Creating volume
	//
	Eigen::Vector3f voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3f volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3f voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	//
	const int total_voxels = 
		(volume_size.x() / voxel_size.x() + 1) *
		(volume_size.y() / voxel_size.y() + 1) *
		(volume_size.z() / voxel_size.z() + 1);


	Eigen::Affine3f grid_affine = Eigen::Affine3f::Identity();
	grid_affine.translate(Eigen::Vector3f(0, 0, -256));
	grid_affine.scale(Eigen::Vector3f(1, 1, -1));	// z is negative inside of screen

	std::vector<Eigen::Vector4f> grid_voxels_points(total_voxels);
	std::vector<Eigen::Vector2f> grid_voxels_params(total_voxels, Eigen::Vector2f(0.0f, 1.0f));

	//
	// Creating grid in GPU
	//
	timer.start();
	Eigen::Matrix4f identity_mat = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f grid_matrix = grid_affine.matrix();
	Eigen::Matrix4f grid_matrix_inv = grid_matrix.inverse();

	//create_grid(vol_size, vx_size, grid_matrix.data(), &grid_voxels_points[0][0], &grid_voxels_params[0][0]);
	grid_init(vol_size, vx_size, &grid_voxels_points[0][0], &grid_voxels_params[0][0], grid_matrix.data(), grid_matrix_inv.data(), K.data());
	timer.print_interval("GPU create grid         : ");

	//
	// Update volume
	//
	timer.start();
	Eigen::Matrix4f view_matrix = T.first;
	Eigen::Matrix4f view_matrix_inv = view_matrix.inverse();
	grid_update(view_matrix.data(), view_matrix_inv.data(), &depth_buffer.first.data()[0], window_width, window_height);
	timer.print_interval("GPU update              : ");

	////
	//// Get data from gpu
	////
	//timer.start();
	//grid_get_data(&grid_voxels_points[0][0], &grid_voxels_params[0][0]);
	//timer.print_interval("GPU get data            : ");
	//
	//
	//timer.start();
	//export_volume("../../data/grid_volume_gpu.obj", grid_voxels_points, grid_voxels_params);
	//timer.print_interval("Exporting volume        : ");

	grid_get_data(&grid_voxels_points[0][0], &grid_voxels_params[0][0]);

	//std::cout << "------- // --------" << std::endl;
	//for (int i = 0; i < grid_voxels_points.size(); ++i)
	//{
	//	const Eigen::Vector4f& point = grid_voxels_points[i];
	//	const Eigen::Vector2f& param = grid_voxels_params[i];

	//	std::cout << std::fixed << point.transpose() << "\t\t" << param.transpose() << std::endl;
	//}
	//std::cout << "------- // --------" << std::endl;


	//
	// Compute next clouds
	Eigen::Matrix4f cloud_mat = Eigen::Matrix4f::Identity();
	Timer iter_timer;
	for (int i = 1; i < cloud_count; ++i)
	{
		std::cout << std::endl << i << " : " << i * rot_interval << std::endl;
		iter_timer.start();

		//
		// Rotation matrix
		//
		rotate = Eigen::Affine3f::Identity();
		rotate.rotate(Eigen::AngleAxisf(DegToRad(i * rot_interval), Eigen::Vector3f::UnitY()));


		// 
		// Compute next cloud
		//
		timer.start();
		cloud.second.clear();
		cloud.second.resize(points3DOrig.size());
		matrix_mulf(&cloud.second[0][0], trans_rot.data(), &points3DOrig[0][0], trans_rot.rows(), trans_rot.cols(), points3DOrig.size());
		timer.print_interval("GPU compute next cloud  : ");

		//export_obj("../../data/cloud_cpu_2.obj", cloud.second);


		// 
		// Compute depth buffer
		//
		timer.start();
		compute_depth_buffer(
			&depth_buffer.second.data()[0],
			&window_coords.second[0][0],
			&cloud.second[0][0],
			cloud.second.size(),
			K.data(),
			window_width,
			window_height);
		timer.print_interval("GPU compute depth 2     : ");
		
		//export_depth_buffer("../../data/gpu_depth_buffer_2.obj", depth_buffer.second);

		timer.start();
		Eigen::Matrix4f icp_mat;
		ComputeRigidTransform(cloud.first, cloud.second, icp_mat);
		timer.print_interval("Compute rigid transform : ");

		// accumulate matrix
		cloud_mat = cloud_mat * icp_mat;

		////
		//// Update Volume in CPU
		////
		//timer.start();
		//grid_get_data(&grid_voxels_points[0][0], &grid_voxels_params[0][0]);
		//update_volume(grid_voxels_points, grid_voxels_params, depth_buffer.second, K, cloud_mat.inverse(), vol_size, vx_size);
		//timer.print_interval("Update volume           : ");

		//
		// Update volume in GPU
		//
		timer.start();
		view_matrix = cloud_mat;
		view_matrix_inv = view_matrix.inverse();
		//grid_update(view_matrix.data(), view_matrix_inv.data(), &depth_buffer.second.data()[0], window_width, window_height);
		grid_update(view_matrix_inv.data(), view_matrix.data(), &depth_buffer.second.data()[0], window_width, window_height);
		timer.print_interval("Update volume           : ");

		// copy second point cloud to first
		cloud.first = cloud.second;
		//depth_buffer.first = depth_buffer.second;

		iter_timer.print_interval("Iteration time          : ");
	}


	//
	// Get data from gpu
	//
	timer.start();
	grid_get_data(&grid_voxels_points[0][0], &grid_voxels_params[0][0]);
	timer.print_interval("GPU get data            : ");


	timer.start();
	export_volume("../../data/grid_volume_gpu.obj", grid_voxels_points, grid_voxels_params);
	timer.print_interval("Exporting volume        : ");



	//std::cout << "------- // --------" << std::endl;
	//for (int i = 0; i < grid_voxels_points.size(); ++i)
	//{
	//	const Eigen::Vector4f& point = grid_voxels_points[i];
	//	const Eigen::Vector2f& param = grid_voxels_params[i];
	//
	//	std::cout << point.transpose() << "\t\t" << param.transpose() << std::endl;
	//}
	//std::cout << "------- // --------" << std::endl;

}

