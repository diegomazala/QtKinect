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
	grid_get_data(&grid_voxels_points[0][0], &grid_voxels_params[0][0]);
	timer.print_interval("GPU get data            : ");


	timer.start();
	export_volume("../../data/grid_volume_gpu.obj", grid_voxels_points, grid_voxels_params);
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
	
	K = perspective_matrix<float>(KINECT_V1_FOVY, KINECT_V1_ASPECT_RATIO, knt_near_plane, knt_far_plane);
	Eigen::Matrix4f proj_inv = K.inverse();
	
	
	//std::cout << "Perspective: " << std::endl << K << std::endl;


	//K = perspective_matrix<float>(fov_y, aspect_ratio, near_plane, far_plane);
	//std::cout << "Perspective: " << std::endl << K << std::endl;

#if 0
	std::vector<Eigen::Vector3f> vertices;
	std::cout << "depth : " << knt.depth_width() << ", " << knt.depth_height() << std::endl;

	for (ushort y = 0; y < knt.depth_height(); ++y)
	{
		for (ushort x = 0; x < knt.depth_width(); ++x)
		{
			const float depth = (float)knt.depth_at(x, y) * 0.1f;
			const Eigen::Vector2f pixel(x, y);
			const Eigen::Vector3f v = window_coord_to_3d(pixel, depth, proj_inv, (float)knt.depth_width(), (float)knt.depth_height());
			//points3DOrig.push_back((v * 0.025f).homogeneous());
			points3DOrig.push_back(v.homogeneous());
		}
	}

	//export_obj("../../data/knt_frame.obj", points3DOrig);
	//return;


	timer.print_interval("Importing knt frame : ");
	std::cout << filepath << " point count  : " << points3DOrig.size() << std::endl;

	point_count = points3DOrig.size();
#endif


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
	//timer.start();
	Eigen::Matrix4f view_matrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f view_matrix_inv = view_matrix.inverse();
	grid_update(view_matrix.data(), view_matrix_inv.data(), &depth_buffer.first.data()[0], knt.depth_width(), knt.depth_height());
	timer.print_interval("GPU update              : ");



	//
	// Get data from gpu
	//
	timer.start();
	grid_get_data(&grid_voxels_points[0][0], &grid_voxels_params[0][0]);
	timer.print_interval("GPU get data            : ");
	

	Eigen::Affine3f grid_affine_2 = Eigen::Affine3f::Identity();
	grid_affine_2.translate(Eigen::Vector3f(-half_vol_size, -half_vol_size, 0));
#if 1
	
	timer.start();
	//export_volume("../../data/grid_volume_gpu_knt.obj", grid_voxels_points, grid_voxels_params);
	//export_volume("../../data/grid_volume_gpu_knt_2.obj", voxel_count.cast<int>(), voxel_size.cast<int>(), grid_voxels_params, grid_affine_2.matrix());
	//export_params("../../data/grid_volume_gpu_params.txt", grid_voxels_params);
	timer.print_interval("Exporting volume        : ");

	//return 0;
	
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	//camera_to_world.translate(Eigen::Vector3f(0, 0, -10000));
	//camera_to_world.translate(Eigen::Vector3f(0, 0, 511));
	Eigen::Vector3f camera_pos = camera_to_world.matrix().col(3).head<3>();
	float scale = tan(DegToRad(KINECT_V1_FOVY * 0.5f));
	float aspect_ratio = KINECT_V1_ASPECT_RATIO;
	ushort image_width = KINECT_V1_COLOR_WIDTH / 4;
	ushort image_height = KINECT_V1_COLOR_HEIGHT / 4;
	uchar* image_data = new uchar[image_width * image_height * 3]{0}; // rgb
	QImage image(image_data, image_width, image_height, QImage::Format_RGB888);
	image.fill(Qt::GlobalColor::black);

	Eigen::Vector3f hit;
	Eigen::Vector3f v1(0.0f, -1.0f, -2.0f);
	Eigen::Vector3f v2(0.0f, 1.0f, -4.0f);
	Eigen::Vector3f v3(-1.0f, -1.0f, -3.0f);
	Eigen::Vector3f v4(0.0f, -1.0f, -2.0f);
	Eigen::Vector3f v5(0.0f, 1.0f, -4.0f);
	Eigen::Vector3f v6(1.0f, -1.0f, -3.0f);

	Eigen::Vector3f diff_color(1, 0, 0);
	Eigen::Vector3f spec_color(1, 1, 0);
	float spec_shininess = 1.0f;
	Eigen::Vector3f E(0, 0, -1);				// view direction
	Eigen::Vector3f L = Eigen::Vector3f(0.2f, -1, -1).normalized();	// light direction
	Eigen::Vector3f N[2] = {
		compute_normal(v1, v2, v3),
		compute_normal(v4, v5, v6) };
	Eigen::Vector3f R[2] = {
		-reflect(L, N[0]).normalized(),
		-reflect(L, N[1]).normalized() };

	timer.start();
	for (int y = 0; y < image_height; ++y)
	{
		for (int x = 0; x < image_width; ++x)
		{
			//
			// Convert from image space (in pixels) to screen space
			// Screen Space alon X axis = [-aspect ratio, aspect ratio] 
			// Screen Space alon Y axis = [-1, 1]
			//
			Eigen::Vector3f screen_coord(
				(2 * (x + 0.5f) / (float)image_width - 1) * aspect_ratio * scale,
				(1 - 2 * (y + 0.5f) / (float)image_height) * scale,
				1.0f);

			//
			// compute direction of the ray
			//
			Eigen::Vector3f direction;
			multDirMatrix(screen_coord, camera_to_world.matrix(), direction);
			direction.normalize();

			//
			// compute intersection for a ray through the volume
			//
			std::vector<int> voxels_zero_crossing;
			int voxels_cross = raycast_tsdf_volume(
				camera_pos,
				direction,
				voxel_count.cast<int>(),
				voxel_size.cast<int>(),
				grid_voxels_params,
				voxels_zero_crossing);

			if (voxels_cross == 2)
			{ 
				//Eigen::Vector3f diff = diff_color * std::fmax(N[i].dot(L), 0.0f);
				//Eigen::Vector3f spec = spec_color * pow(std::fmax(R[i].dot(E), 0.0f), spec_shininess);
				//Eigen::Vector3f color = eigen_clamp(diff + spec, 0.f, 1.f) * 255;
				Eigen::Vector3f color(255, 255, 0);
				image.setPixel(QPoint(x, y), qRgb(color.x(), color.y(), color.z()));
			}
		}
	}
	timer.print_interval("Raycasting volume       : ");

	
	QApplication app(argc, argv);
	QImageWidget widget;
	widget.setImage(image);
	widget.show();

	return app.exec();


#else
	Eigen::Affine3f box_transform = Eigen::Affine3f::Identity();
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	float fovy = KINECT_V1_FOVY;
	int width = KINECT_V1_COLOR_WIDTH;
	int height = KINECT_V1_COLOR_HEIGHT;
	uchar* image_data = new uchar[width * height * 3]; // rgb

	ushort voxel_count_us[3] = { voxel_count.x(), voxel_count.y(), voxel_count.z() };
	ushort voxel_size_us[3] = { voxel_size.x(), voxel_size.y(), voxel_size.z() };

	raycast_image_grid(
		image_data, 
		width, 
		height, 
		voxel_count_us,
		voxel_size_us,
		fovy, 
		camera_to_world.matrix().data(), 
		box_transform.matrix().data());

	QImage image(image_data, width, height, QImage::Format_RGB888);

	QApplication app(argc, argv);
	QImageWidget widget;
	widget.setImage(image);
	widget.show();

	return app.exec();
#endif

	//std::cout << "------- // --------" << std::endl;
	//for (int i = 0; i < grid_voxels_points.size(); ++i)
	//{
	//	const Eigen::Vector4f& point = grid_voxels_points[i];
	//	const Eigen::Vector2f& param = grid_voxels_params[i];

	//	std::cout << std::fixed << point.transpose() << "\t\t" << param.transpose() << std::endl;
	//}
	//std::cout << "------- // --------" << std::endl;
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

