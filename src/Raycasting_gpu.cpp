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
#include "QRaycastImageWidget.h"
#include "GLRaycastTextureWidget.h"
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
std::string filepath = "../../data/room.knt";
int vx_count = 256;
int vx_size = 2;


static void export_debug_buffer(const std::string& filename, const float4* image_data, int width, int height)
{
	std::ofstream file;
	file.open(filename);
	int i = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			file << std::fixed << x << ' ' << y << ' ' << image_data[i].x << ' ' << image_data[i].y << ' ' << image_data[i].z << std::endl;
			++i;
		}
	}
	file.close();
}


static void export_image_buffer(const std::string& filename, const uchar4* image_data, int width, int height)
{
	std::ofstream file;
	file.open(filename);
	int i = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			file << std::fixed << x << ' ' << y << ' ' << (int)image_data[i].x << ' ' << (int)image_data[i].y << ' ' << (int)image_data[i].z << std::endl;
			++i;
		}
	}
	file.close();
}

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
			<< int((c.x * 0.5f + 0.5f) * 255) << ' ' << int((c.y * 0.5f + 0.5f) * 255) << ' ' << int((c.z * 0.5f + 0.5f) * 255)
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

	std::vector<float4> vertices(knt.depth.size(), make_float4(0, 0, 0, 1));
	std::vector<float4> normals(knt.depth.size(), make_float4(0, 0, 1, 1));
	std::vector<Eigen::Vector2f> grid_voxels_params(total_voxels);

	// 
	// setup image parameters
	//
	unsigned short image_width = KINECT_V2_DEPTH_WIDTH;
	unsigned short image_height = image_width / aspect_ratio;
	QImage img(image_width, image_height, QImage::Format::Format_RGBA8888);
	img.fill(Qt::GlobalColor::gray);
	uchar4* image_data = (uchar4*)img.bits();
	//float4* debug_buffer = new float4[image_width * image_height];
	//memset(debug_buffer, 0, image_width * image_height * sizeof(float4));

	knt_cuda_setup(
		vx_count, vx_size,
		grid_matrix.data(),
		projection.data(),
		projection_inverse.data(),
		*grid_voxels_params.data()->data(),
		KINECT_V2_DEPTH_WIDTH,
		KINECT_V2_DEPTH_HEIGHT,
		KINECT_V2_DEPTH_MIN,
		KINECT_V2_DEPTH_MAX,
		vertices.data()[0],
		normals.data()[0],
		image_width,
		image_height
		);

	timer.start();
	knt_cuda_allocate();
	knt_cuda_init_grid();
	timer.print_interval("Allocating gpu      : ");

	timer.start();
	knt_cuda_copy_host_to_device();
	knt_cuda_copy_depth_buffer_to_device(knt.depth.data());
	timer.print_interval("Copy host to device : ");

	timer.start();
	knt_cuda_normal_estimation();
	timer.print_interval("Normal estimation   : ");

	timer.start();
	knt_cuda_update_grid(view_matrix.data());
	timer.print_interval("Update grid         : ");

	timer.start();
	knt_cuda_grid_params_copy_device_to_host();
	knt_cuda_copy_device_to_host();
	timer.print_interval("Copy device to host : ");




	//
	// setup camera parameters
	//
	timer.start();
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	float cam_z = -half_vol_size;
	camera_to_world.scale(Eigen::Vector3f(1, 1, -1));
	camera_to_world.translate(Eigen::Vector3f(half_vol_size, half_vol_size, cam_z));

	
	Eigen::Matrix4f camera_to_world_matrix = camera_to_world.matrix();
		
	knt_cuda_raycast(KINECT_V2_FOVY, KINECT_V2_DEPTH_ASPECT_RATIO, camera_to_world_matrix.data());
	timer.print_interval("Raycast             : ");

	timer.start();
	knt_cuda_copy_image_device_to_host(*(uchar4*)img.bits());
	timer.print_interval("Copy Img to host    : ");
	
	timer.start();
	knt_cuda_free();
	timer.print_interval("Cleanup gpu         : ");

#if 0
	//memset(image_data, 0, image_width * image_height * sizeof(uchar4));
	//memset(debug_buffer, 0, image_width * image_height * sizeof(float4));

	Eigen::Vector3f camera_pos = camera_to_world_matrix.col(3).head<3>();
	float fov_scale = (float)tan(DegToRad(KINECT_V2_FOVY * 0.5f));
	float aspect_ratio = KINECT_V2_DEPTH_ASPECT_RATIO;


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
			float x_norm = (2.f * float(x) + 0.5f) / (float)image_width;
			float y_norm = (2.f * float(y) + 0.5f) / (float)image_height;
			Eigen::Vector3f screen_coord(
				(x_norm - 1.f) * aspect_ratio * fov_scale,
				(1.f - y_norm) * fov_scale,
				1.0f);

			Eigen::Vector3f direction;
			multDirMatrix(screen_coord, camera_to_world_matrix, direction);
			direction.normalize();

			long voxels_zero_crossing[2] = { -1, -1 };

			int hit_count = raycast_tsdf_volume<float>(
				camera_pos,
				direction,
				voxel_count.cast<int>(),
				voxel_size.cast<int>(),
				grid_voxels_params,
				voxels_zero_crossing);

			if (hit_count > 0)
			{
				if (hit_count == 2)
				{
					float4 n = normals[y * image_width + x];

					//image_data[y * image_width + x].x = 0;
					//image_data[y * image_width + x].y = 128;
					//image_data[y * image_width + x].z = 128;
					//image_data[y * image_width + x].w = 255;
					
					image_data[y * image_width + x].x = uchar((n.x * 0.5f + 0.5f) * 255);
					image_data[y * image_width + x].y = uchar((n.y * 0.5f + 0.5f) * 255);
					image_data[y * image_width + x].z = uchar((n.z * 0.5f + 0.5f) * 255);
					image_data[y * image_width + x].w = 255;
				}
				else
				{
					image_data[y * image_width + x].x = 128;
					image_data[y * image_width + x].y = 128;
					image_data[y * image_width + x].z = 0;
					image_data[y * image_width + x].w = 255;
				}
			}
			else
			{
				image_data[y * image_width + x].x = 128;
				image_data[y * image_width + x].y = 0;
				image_data[y * image_width + x].z = 0;
				image_data[y * image_width + x].w = 255;
			}
		}
	}
	timer.print_interval("Raycasting to image     : ");
	//export_debug_buffer("../../data/cpu_image_data_screen_coord_f4.txt", debug_buffer, image_width, image_height);
	//export_image_buffer("../../data/cpu_image_data_screen_coord_uc.txt", image_data, image_width, image_height);
#else

	//export_debug_buffer("../../data/gpu_image_data_screen_coord_f4.txt", debug_buffer, image_width, image_height);
	//export_image_buffer("../../data/gpu_image_data_screen_coord_uc.txt", image_data, image_width, image_height);
#endif

	



	QImage image(&image_data[0].x, image_width, image_height, QImage::Format_RGBA8888);
	//image.fill(Qt::GlobalColor::black);
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
			<< "Usage:  ./Raycasting_gpud.exe ../../data/room.knt 256 2 2 90"
			<< std::endl
			<< "The app will continue with default parameters."
			<< std::endl;

		filepath = "../../data/room.knt";
		vx_count = 32;
		vx_size = 32;
	}
	else
	{
		filepath = argv[1];
		vx_count = atoi(argv[2]);
		vx_size = atoi(argv[3]);
	}

	//return volumetric_knt_cuda(argc, argv);

	QApplication app(argc, argv);

#if 0
	GLRaycastTextureWidget glwidget;
	glwidget.setWindowTitle("GL Widget");
	glwidget.resize(KINECT_V2_DEPTH_WIDTH, KINECT_V2_DEPTH_HEIGHT);
	glwidget.setup(filepath, vx_count, vx_size);
	glwidget.show();
#else
	QRaycastImageWidget widget;
	widget.setWindowTitle("Raycast Widget");
	widget.resize(KINECT_V2_DEPTH_WIDTH, KINECT_V2_DEPTH_HEIGHT);
	widget.setup(filepath, vx_count, vx_size);
	widget.show();
#endif
	

	return app.exec();
}

