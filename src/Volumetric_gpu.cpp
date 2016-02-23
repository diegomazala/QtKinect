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
	Eigen::Affine3d grid_affine = Eigen::Affine3d::Identity();
	grid_affine.translate(Eigen::Vector3d(0, 0, -256));
	grid_affine.scale(Eigen::Vector3d(1, 1, -1));	// z is negative inside of screen
	Grid grid(volume_size, voxel_size, grid_affine.matrix());

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


	//for (int z = 0; z <= voxel_step; ++z)
	//{
	//	for (int y = 0; y <= voxel_step; ++y)
	//	{
	//		printf("\nz,y=%d,%d\n", z,y);
	//		for (int x = 0; x <= voxel_step; ++x)
	//		{
	//			float3 v;
	//			v.x = float(x * half_vol_size - half_vol_size);
	//			v.y = float(y * half_vol_size - half_vol_size);
	//			v.z = -256.0f - float(threadId * half_vol_size - half_vol_size);

	//			//int index = z * dim.z + y * dim.y + x;
	//			int index = x + dim.x * (y + dim.z * z);

	//			printf("-->%d %d : \t %f,\t %f,\t %f\n",
	//				index, ((int)threadId), 
	//				v.x, v.y, v.z);

	//			

	//			grid_voxels.at(index) = Eigen::Vector4f(v.x, v.y, v.z, 1.0f);
	//		}
	//	}
	//	threadId++;
	//}
	//
	//std::cout << std::endl << std::endl;

	std::vector<float> depth_buffer(window_width * window_height);
	Eigen::Matrix4f grid_matrix = grid_affine.matrix().cast<float>();
	//update_grid(vol_size, vx_size, grid_matrix.data(), &grid_voxels[0][0], depth_buffer.data(), window_width, window_height);
	create_grid(vol_size, vx_size, grid_matrix.data(), &grid_voxels_points[0][0], &grid_voxels_params[0][0]);

	std::cout << "------- // --------" << std::endl;
	for (const auto v : grid_voxels_points)
	{
		std::cout << v.transpose() << std::endl;
	}
	std::cout << "------- // --------" << std::endl;
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

	test_grid(argc, argv);
	return 0;


	//
	// Importing .obj
	//
	timer.start();
	std::vector<Eigen::Vector4f> points3DOrig, out_points;
	import_obj(filepath, points3DOrig);
	timer.print_interval("Importing monkey    : ");
	std::cout << filepath << " point count  : " << points3DOrig.size() << std::endl;

	const std::size_t point_count = points3DOrig.size();
	const std::size_t pixel_count = static_cast<const std::size_t>(window_width * window_height);


	Eigen::Matrix4f K = perspective_matrix<float>(fov_y, aspect_ratio, near_plane, far_plane);

	// 
	// Translating and rotating monkey point cloud 
	std::pair<std::vector<Eigen::Vector4f>, std::vector<Eigen::Vector4f>> cloud;
	cloud.first.resize(points3DOrig.size());
	cloud.second.resize(points3DOrig.size());
	//
	Eigen::Affine3f rotate = Eigen::Affine3f::Identity();
	rotate.rotate(Eigen::AngleAxisf(DegToRad(90), Eigen::Vector3f::UnitY()));
	Eigen::Affine3f translate = Eigen::Affine3f::Identity();
	translate.translate(Eigen::Vector3f(0, 0, -256));


	// 
	// Compute first cloud
	//
	//for (Eigen::Vector4f p3d : points3DOrig)
	//{
	//	Eigen::Vector4f trans = translate.matrix() * p3d;
	//	trans /= trans.w();
	//	cloud.first.push_back(trans);

	//	Eigen::Vector4f rot = translate.matrix() * rotate.matrix() * p3d;
	//	rot /= rot.w();
	//	cloud.second.push_back(rot);
	//}

	Eigen::Matrix4f identity_mat4f = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f trans_rot = translate.matrix() * rotate.matrix();

	timer.start();
	matrix_mulf(&cloud.first[0][0], translate.matrix().data(), &points3DOrig[0][0], translate.matrix().rows(), translate.matrix().cols(), points3DOrig.size());
	matrix_mulf(&cloud.second[0][0], trans_rot.data(), &points3DOrig[0][0], trans_rot.rows(), trans_rot.cols(), points3DOrig.size());
	timer.print_interval("GPU transform points    : ");

	//export_obj("../../data/out_gpu_1.obj", cloud.first);
	//export_obj("../../data/out_gpu_2.obj", cloud.second);

	std::pair<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>> window_coords;
	window_coords.first.resize(point_count);
	window_coords.second.resize(point_count);
	
	std::pair<std::vector<float>, std::vector<float>> depth_buffer;
	depth_buffer.first.resize(pixel_count, far_plane);
	depth_buffer.second.resize(pixel_count, far_plane);


	out_points.resize(points3DOrig.size());

	compute_depth_buffer(
		&depth_buffer.first.data()[0],
		&window_coords.first[0][0],
		&cloud.first[0][0], 
		cloud.first.size(), 
		K.data(), 
		window_width, 
		window_height);

	compute_depth_buffer(
		&depth_buffer.second.data()[0],
		&window_coords.first[0][0],
		&cloud.second[0][0],
		cloud.second.size(),
		K.data(),
		window_width,
		window_height);

	//std::ofstream file;
	//file.open("../../data/gpu_out_window_coords.txt");
	//for (int i = 0; i < window_coords.first.size(); ++i)
	//{
	//	file << std::fixed << window_coords.first.at(i).cast<int>().transpose() << std::endl;
	//}
	//file.close();

	//export_obj("../../data/gpu_out_points.obj", out_points);

	//export_depth_buffer("../../data/gpu_depth_buffer_1.obj", depth_buffer.first);
	//export_depth_buffer("../../data/gpu_depth_buffer_2.obj", depth_buffer.second);

	//create_depth_buffer(depth_buffer.first, cloud.first, K, Eigen::Matrix4f::Identity(), far_plane);
	//export_depth_buffer("../../data/cpu_depth_buffer.obj", depth_buffer.first);

	//file.open("../../data/cpu_out_index.txt");
	//for (int i = 0; i < depth_buffer.first.size(); ++i)
	//{
	//	file << std::fixed << depth_buffer.first.at(i) << std::endl;
	//}
	//file.close();




#if 0
	std::pair<std::vector<double>, std::vector<double>> depth_buffer;

	//
	// Projection and Modelview Matrices
	//
	Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);
	std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());


	//
	// Creating volume
	//
	Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	//
	Eigen::Affine3d grid_affine = Eigen::Affine3d::Identity();
	grid_affine.translate(Eigen::Vector3d(0, 0, -256));
	grid_affine.scale(Eigen::Vector3d(1, 1, -1));	// z is negative inside of screen
	Grid grid(volume_size, voxel_size, grid_affine.matrix());


	
	// 
	// Translating and rotating monkey point cloud 
	std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> cloud;
	//
	Eigen::Affine3d rotate = Eigen::Affine3d::Identity();
	Eigen::Affine3d translate = Eigen::Affine3d::Identity();
	translate.translate(Eigen::Vector3d(0, 0, -256));

	// 
	// Compute first cloud
	//
	for (Eigen::Vector3d p3d : points3DOrig)
	{
		Eigen::Vector4d rot = translate.matrix() * rotate.matrix() * p3d.homogeneous();
		rot /= rot.w();
		cloud.first.push_back(rot.head<3>());
	}
	//
	// Update grid with first cloud
	//
	create_depth_buffer(depth_buffer.first, cloud.first, K, Eigen::Matrix4d::Identity(), far_plane);
	update_volume(grid, depth_buffer.first, K, T.first.inverse());

	//
	// Compute next clouds
	Eigen::Matrix4d cloud_mat = Eigen::Matrix4d::Identity();
	Timer iter_timer;
	for (int i = 1; i < cloud_count; ++i)
	{
		std::cout << std::endl << i << " : " << i * rot_interval << std::endl;
		iter_timer.start();

		// Rotation matrix
		rotate = Eigen::Affine3d::Identity();
		rotate.rotate(Eigen::AngleAxisd(DegToRad(i * rot_interval), Eigen::Vector3d::UnitY()));

		cloud.second.clear();
		for (Eigen::Vector3d p3d : points3DOrig)
		{
			Eigen::Vector4d rot = translate.matrix() * rotate.matrix() * p3d.homogeneous();
			rot /= rot.w();
			cloud.second.push_back(rot.head<3>());
		}

		timer.start();
		create_depth_buffer(depth_buffer.second, cloud.second, K, Eigen::Matrix4d::Identity(), far_plane);
		timer.print_interval("Compute depth buffer: ");

		timer.start();
		Eigen::Matrix4d icp_mat;
		ComputeRigidTransform(cloud.first, cloud.second, icp_mat);
		timer.print_interval("Compute rigid transf: ");

		//std::cout << std::fixed << std::endl << "icp_mat " << std::endl << icp_mat << std::endl;

		// accumulate matrix
		cloud_mat = cloud_mat * icp_mat;

		//std::cout << std::fixed << std::endl << "cloud_mat " << std::endl << cloud_mat << std::endl;

		timer.start();
		update_volume(grid, depth_buffer.second, K, cloud_mat.inverse());
		timer.print_interval("Update volume       : ");


		// copy second point cloud to first
		cloud.first = cloud.second;
		depth_buffer.first = depth_buffer.second;


		iter_timer.print_interval("Iteration time      : ");
	}


	timer.start();
	export_volume("../../data/grid_volume.obj", grid.data);
	timer.print_interval("Exporting volume    : ");


#endif


}
