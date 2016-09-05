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
#include <cassert>

#include <QApplication>
#include "QImageWidget.h"

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_kernels/cuda_kernels.h"
#include "Volumetric_helper.h"
#include "Projection.h"
#include "KinectFrame.h"
#include "KinectSpecs.h"
#include "Raycasting.h"


//
// globals
// 
Timer timer;
std::string filepath = "../../data/monkey.obj";
int vol_size = 256;
int vx_size = 2;
int cloud_count = 1;
int rot_interval = 30;
std::vector<Eigen::Vector4f> points3DOrig;
std::size_t point_count = 0;
std::size_t pixel_count = 0;
Eigen::Matrix4f K;

static void export_knt_frame(const std::string& filename, const KinectFrame& knt)
{
	std::ofstream file;
	file.open(filename);
	for (ushort y = 0; y < knt.depth_height(); ++y)
	{
		for (ushort x = 0; x < knt.depth_width(); ++x)
		{
			file << std::fixed << "v " << x << ' ' << y << ' ' << knt.depth_at(x, y) << std::endl;
		}
	}
	file.close();
}

static void export_params(const std::string& filename, const std::vector<Eigen::Vector2f>& params) //, const Eigen::Matrix4f& transformation = Eigen::Matrix4f::Identity())
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < params.size(); ++i)
	{
		file << std::fixed << params[i].x() << ' ' << params[i].y() << std::endl;
	}
	file.close();
}

static void multDirMatrix(const Eigen::Vector3f &src, const Eigen::Matrix4f &mat, Eigen::Vector3f &dst)
{
	float a, b, c;

	a = src[0] * mat(0, 0) + src[1] * mat(1, 0) + src[2] * mat(2, 0);
	b = src[0] * mat(0, 1) + src[1] * mat(1, 1) + src[2] * mat(2, 1);
	c = src[0] * mat(0, 2) + src[1] * mat(1, 2) + src[2] * mat(2, 2);

	dst.x() = a;
	dst.y() = b;
	dst.z() = c;
}


template <typename Type>
static Eigen::Matrix<Type, 3, 1> compute_normal(
	const Eigen::Matrix<Type, 3, 1>& p1,
	const Eigen::Matrix<Type, 3, 1>& p2,
	const Eigen::Matrix<Type, 3, 1>& p3)
{
	Eigen::Matrix<Type, 3, 1> u = p2 - p1;
	Eigen::Matrix<Type, 3, 1> v = p3 - p1;

	return v.cross(u).normalized();
}

template <typename Type>
static Eigen::Matrix<Type, 3, 1> reflect(const Eigen::Matrix<Type, 3, 1>& i, const Eigen::Matrix<Type, 3, 1>& n)
{
	return i - 2.0 * n * n.dot(i);
}





void run_for_obj()
{
	import_obj(filepath, points3DOrig);
	timer.print_interval("Importing obj file  : ");
	std::cout << filepath << " point count  : " << points3DOrig.size() << std::endl;

	point_count = points3DOrig.size();
	pixel_count = static_cast<const std::size_t>(window_width * window_height);

	K = perspective_matrix<float>(fov_y, aspect_ratio, near_plane, far_plane);
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
	matrix_mulf(&cloud.first[0][0], trans_rot.data(), &points3DOrig[0][0], trans_rot.rows(), trans_rot.cols(), (int)points3DOrig.size());
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
		(uint)cloud.first.size(),
		K.data(),
		window_width,
		window_height);
	timer.print_interval("GPU compute depth 1     : ");

	timer.start();
	compute_depth_buffer(
		&depth_buffer.second.data()[0],
		&window_coords.first[0][0],
		&cloud.second[0][0],
		(uint)cloud.second.size(),
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

	//std::vector<Eigen::Vector4f> grid_voxels_points(total_voxels);
	std::vector<Eigen::Vector2f> grid_voxels_params(total_voxels, Eigen::Vector2f(0.0f, 1.0f));

	//
	// Creating grid in GPU
	//
	timer.start();
	Eigen::Matrix4f identity_mat = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f grid_matrix = grid_affine.matrix();
	Eigen::Matrix4f grid_matrix_inv = grid_matrix.inverse();

	grid_init(vol_size, vx_size, &grid_voxels_params[0][0], grid_matrix.data(), grid_matrix_inv.data(), K.data());
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

	grid_get_data(&grid_voxels_params[0][0]);

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
		matrix_mulf(&cloud.second[0][0], trans_rot.data(), &points3DOrig[0][0], trans_rot.rows(), trans_rot.cols(), (int)points3DOrig.size());
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
			(uint)cloud.second.size(),
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
	grid_get_data(&grid_voxels_params[0][0]);
	timer.print_interval("GPU get data            : ");


	timer.start();
	//export_volume("../../data/grid_volume_gpu.obj", grid_voxels_points, grid_voxels_params);
	timer.print_interval("Exporting volume        : ");
}




int run_for_knt(int argc, char **argv)
{
	timer.start();
	float knt_near_plane = 0.1f;
	float knt_far_plane = 10240.0f;

	KinectFrame knt(filepath);
	std::cout << "KinectFrame loaded: " << knt.depth.size() << std::endl;
	timer.print_interval("Importing knt frame : ");

	//export_knt_frame("../../data/knt_frame_depth.obj", knt);
	//return;

	//std::pair<Eigen::Matrix4f, Eigen::Matrix4f>	T(Eigen::Matrix4f::Identity(), Eigen::Matrix4f::Identity());
	
	K = perspective_matrix<float>(KINECT_V2_FOVY, KINECT_V2_DEPTH_ASPECT_RATIO, knt_near_plane, knt_far_plane);
	Eigen::Matrix4f proj_inv = K.inverse();
	
	
	pixel_count = static_cast<const std::size_t>(knt.depth_width() * knt.depth_height());


	std::pair<std::vector<float>, std::vector<float>> depth_buffer;
	depth_buffer.first.resize(pixel_count, far_plane);
	depth_buffer.second.resize(pixel_count, far_plane);


	//
	// converting depth buffer to float
	// //
	for (int i = 0; i < pixel_count; ++i)
	{
		depth_buffer.first[i] = depth_buffer.second[i] = (float)knt.depth[i];
	}


	//
	// Creating volume
	//
	Eigen::Vector3f voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3f volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3f voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	Eigen::Vector3f half_volume_size = volume_size * 0.5f;

	//
	//const int total_voxels =
	//	(volume_size.x() / voxel_size.x() + 1) *
	//	(volume_size.y() / voxel_size.y() + 1) *
	//	(volume_size.z() / voxel_size.z() + 1);

	const int total_voxels =
		(volume_size.x() / voxel_size.x()) *
		(volume_size.y() / voxel_size.y()) *
		(volume_size.z() / voxel_size.z());

	const float half_vol_size = vol_size * 0.5f;

	Eigen::Affine3f grid_affine = Eigen::Affine3f::Identity();
	grid_affine.translate(Eigen::Vector3f(0, 0, half_vol_size));
	grid_affine.scale(Eigen::Vector3f(1, 1, 1));	// z is negative inside of screen

	//std::vector<Eigen::Vector4f> grid_voxels_points(total_voxels);
	std::vector<Eigen::Vector2f> grid_voxels_params(total_voxels, Eigen::Vector2f(1.0f, 1.0f));

	//
	// Creating grid in GPU
	//
	timer.start();
	Eigen::Matrix4f identity_mat = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f grid_matrix = grid_affine.matrix();
	Eigen::Matrix4f grid_matrix_inv = grid_matrix.inverse();

	grid_init(vol_size, vx_size, &grid_voxels_params[0][0], grid_matrix.data(), grid_matrix_inv.data(), K.data());
	timer.print_interval("GPU create grid         : ");


	//
	// Update volume
	//
	//timer.start();
	Eigen::Matrix4f view_matrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f view_matrix_inv = view_matrix.inverse();
	grid_update(view_matrix.data(), view_matrix_inv.data(), &depth_buffer.first.data()[0], knt.depth_width(), knt.depth_height());
	timer.print_interval("GPU update              : ");



	//
	// Get data from gpu
	//
	timer.start();
	grid_get_data(&grid_voxels_params[0][0]);
	timer.print_interval("GPU get data            : ");
	

	Eigen::Affine3f grid_affine_2 = Eigen::Affine3f::Identity();
	grid_affine_2.translate(Eigen::Vector3f(-half_vol_size, -half_vol_size, 0));



	timer.start();
	//export_volume("../../data/grid_volume_gpu_knt.obj", grid_voxels_points, grid_voxels_params);
	//export_volume("../../data/grid_volume_gpu_knt.obj", voxel_count.cast<int>(), voxel_size.cast<int>(), grid_voxels_params, grid_affine_2.matrix());
	//export_params("../../data/grid_volume_gpu_params.txt", grid_voxels_params);
	timer.print_interval("Exporting volume        : ");



	std::cout << std::fixed
		<< "Voxel Count  : " << voxel_count.transpose() << std::endl
		<< "Voxel Size   : " << voxel_size.transpose() << std::endl
		<< "Volume Size  : " << volume_size.transpose() << std::endl
		<< "Total Voxels : " << total_voxels << std::endl
		<< "Total Params : " << grid_voxels_params.size() << std::endl;
	
	//return 0;


	//
	// setup camera parameters
	//
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	float cam_z = -512; // (-voxel_count.z() - 1) * vx_size;
	camera_to_world.translate(Eigen::Vector3f(half_volume_size.x(), half_volume_size.y(), cam_z));


	//camera_to_world.translate(Eigen::Vector3f(256, 1024, -512));
	//camera_to_world.rotate(Eigen::AngleAxisf((float)DegToRad(-45.0f), Eigen::Vector3f::UnitX()));

	Eigen::Vector3f camera_pos = camera_to_world.matrix().col(3).head<3>();
	float scale = (float)tan(DegToRad(KINECT_V2_FOVY * 0.5f));
	float aspect_ratio = KINECT_V2_DEPTH_ASPECT_RATIO;

	// 
	// setup image parameters
	//
	unsigned short image_width = KINECT_V2_DEPTH_WIDTH;
	unsigned short image_height = image_width / aspect_ratio;
	unsigned char* image_data = new unsigned char[image_width * image_height * 3]{0}; // rgb
	QImage image(image_data, image_width, image_height, QImage::Format_RGB888);
	image.fill(Qt::GlobalColor::black);

	//
	// for each pixel, trace a ray
	//
	timer.start();
	for (int y = 0; y < image_height; ++y)
	{
		for (int x = 0; x < image_width; ++x)
		{
			// Convert from image space (in pixels) to screen space
			// Screen Space along X axis = [-aspect ratio, aspect ratio] 
			// Screen Space along Y axis = [-1, 1]
			Eigen::Vector3f screen_coord(
				(2 * (x + 0.5f) / (float)image_width - 1) * aspect_ratio * scale,
				(1 - 2 * (y + 0.5f) / (float)image_height) * scale,
				1.0f);

			Eigen::Vector3f direction;
			multDirMatrix(screen_coord, camera_to_world.matrix(), direction);
			direction.normalize();

			std::vector<int> voxels_zero_crossing;
			if (raycast_tsdf_volume<float>(
				camera_pos,
				direction,
				voxel_count.cast<int>(),
				voxel_size.cast<int>(),
				grid_voxels_params,
				voxels_zero_crossing) > 0)
			{
				if (voxels_zero_crossing.size() == 2)
				{
					image.setPixel(QPoint(x, y), qRgb(128, 128, 0));
				}
				else
				{
					image.setPixel(QPoint(x, y), qRgb(128, 0, 0));
				}
			}
		}
	}
	timer.print_interval("Raycasting to image     : ");

	QApplication app(argc, argv);
	QImageWidget widget;
	widget.resize(KINECT_V2_DEPTH_WIDTH, KINECT_V2_DEPTH_HEIGHT);
	widget.setImage(image);
	widget.show();

	return app.exec();


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
			<< std::endl
			<< "The app will continue with default parameters."
			<< std::endl;
		
		filepath = "../../data/monkey.obj";
		vol_size = 256;
		vx_size = 8;
		cloud_count = 2;
		rot_interval = 90;
	}
	else
	{
		filepath = argv[1];
		vol_size = atoi(argv[2]);
		vx_size = atoi(argv[3]);
		cloud_count = atoi(argv[4]);
		rot_interval = atoi(argv[5]);
	}

	if (vol_size / vx_size >= 512)
	{
		std::cerr << "Error: This resolution is not supported due to max number of threads per block. (wip)" << std::endl;
		return EXIT_FAILURE;
	}

	//
	// Importing file
	//
	timer.start();
	

	if (filepath.find(".knt") != std::string::npos)
	{
		return run_for_knt(argc, argv);
	}
	else if (filepath.find(".obj") != std::string::npos)
	{
		run_for_obj();
	}
	else
	{
		std::cerr << "Error: File format not supported. Use .obj or .knt" << std::endl;
	}


	


	



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

