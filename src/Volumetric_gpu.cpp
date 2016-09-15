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
#include "Timer.h"
#include "Eigen/Dense"
#include "Projection.h"
#include "KinectFrame.h"
#include "KinectSpecs.h"
#include "Volumetric_helper.h"

#include "KinectFusionKernels/KinectFusionKernels.h"

//
// globals
// 
Timer timer;
std::string filepath = "../../data/monkey.obj";
int vx_count = 256;
int vx_size = 2;


static void export_obj_with_colors(const std::string& filename, const std::vector<float4>& vertices, const std::vector<float4>& normals)
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < vertices.size(); ++i)
	{
		const auto& v = vertices[i];
		const auto& c = normals[i];
		file << std::fixed << "v " 
			<< v.x << ' ' << v.y << ' ' << v.z 
			<< '\t' 
			<< int(c.x * 255) << ' ' << int(c.y * 255) << ' ' << int(c.z * 255)
			<< std::endl;
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

int volumetric_knt_cuda(int argc, char **argv)
{
	Timer timer;
	//vx_count = 3;
	//vx_size = 1;
	int vol_size = vx_count * vx_size;
	float half_vol_size = vol_size * 0.5f;

	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3i volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
	int total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();


	std::cout << std::fixed
		<< "Voxel Count  : " << voxel_count.transpose() << std::endl
		<< "Voxel Size   : " << voxel_size.transpose() << std::endl
		<< "Volume Size  : " << volume_size.transpose() << std::endl
		<< "Total Voxels : " << total_voxels << std::endl
		<< std::endl;

	timer.start();
	KinectFrame knt(filepath);
	std::cout << "KinectFrame loaded: " << knt.depth.size() << std::endl;
	timer.print_interval("Importing knt frame : ");

	Eigen::Affine3f grid_affine = Eigen::Affine3f::Identity();
	grid_affine.translate(Eigen::Vector3f(0, 0, half_vol_size));
	grid_affine.scale(Eigen::Vector3f(1, 1, 1));	// z is negative inside of screen
	Eigen::Matrix4f grid_matrix = grid_affine.matrix();

	float knt_near_plane = 0.1f;
	float knt_far_plane = 10240.0f;
	Eigen::Matrix4f projection = perspective_matrix<float>(KINECT_V2_FOVY, KINECT_V2_DEPTH_ASPECT_RATIO, knt_near_plane, knt_far_plane);
	Eigen::Matrix4f projection_inverse = projection.inverse();
	Eigen::Matrix4f view_matrix = Eigen::Matrix4f::Identity();

	std::vector<float4> vertices(KINECT_V2_DEPTH_WIDTH * KINECT_V2_DEPTH_HEIGHT);
	std::vector<float4> normals(KINECT_V2_DEPTH_WIDTH * KINECT_V2_DEPTH_HEIGHT);

	knt_cuda_setup(
		vx_count, vx_size, 
		grid_matrix.data(), 
		projection.data(), 
		projection_inverse.data(),
		KINECT_V2_DEPTH_WIDTH, 
		KINECT_V2_DEPTH_HEIGHT,
		KINECT_V2_DEPTH_MIN,
		KINECT_V2_DEPTH_MAX,
		vertices.data()[0],
		normals.data()[0]);

	std::cout << "Cuda allocating ...      " << std::endl;
	knt_cuda_allocate();
	knt_cuda_init_grid();

	std::cout << "Cuda host to device ...  " << std::endl;
	knt_cuda_copy_host_to_device();

	std::cout << "Cuda update grid ...     " << std::endl;
	knt_cuda_copy_depth_buffer_to_device(knt.depth.data());
	knt_cuda_normal_estimation();
	knt_cuda_update_grid(view_matrix.data());

	std::vector<Eigen::Vector2f> grid_voxels_params(total_voxels);
	knt_cuda_grid_params_copy_device_to_host(&grid_voxels_params[0][0]);
	
	std::cout << "Cuda get data from dev..." << std::endl;
	knt_cuda_copy_device_to_host();



	std::cout << "Grid exporting to file..." << std::endl;

	Eigen::Affine3f grid_affine_2 = Eigen::Affine3f::Identity();
	grid_affine_2.translate(Eigen::Vector3f(-half_vol_size, -half_vol_size, 0));

	timer.start();
	export_volume(
		"../../data/grid_volume_gpu_knt.obj", 
		voxel_count,
		voxel_size,
		grid_voxels_params, 
		grid_affine_2.matrix());

	//export_params("../../data/grid_volume_gpu_params_knt_cuda.txt", grid_voxels_params);

	//export_obj<float>("../../data/knt_frame.obj", &vertices.data()[0].x, vertices.size(), 4);
	export_obj_with_colors("../../data/knt_frame.obj", vertices, normals);

	timer.print_interval("Exporting volume        : ");

	std::cout << "Cuda cleanup ...         " << std::endl;
	knt_cuda_free();

	return 0;

	//
	// setup camera parameters
	//
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	float cam_z = -128; // -512; // (-voxel_count.z() - 1) * vx_size;
	camera_to_world.translate(Eigen::Vector3f(half_vol_size, half_vol_size, cam_z));


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
	if (argc < 4)
	{
		std::cerr << "Missing parameters. Abort."
			<< std::endl
			<< "Usage:  ./Volumetricd.exe ../../data/room.knt 256 2 2 90"
			<< std::endl
			<< "The app will continue with default parameters."
			<< std::endl;
		
		filepath = "../../data/room.knt";
		vx_count = 256;
		vx_size = 2;
	}
	else
	{
		filepath = argv[1];
		vx_count = atoi(argv[2]);
		vx_size = atoi(argv[3]);
	}

	return volumetric_knt_cuda(argc, argv);
}

