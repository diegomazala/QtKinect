

#include <QtWidgets>
#include "RaycastImageWidget.h"
#include "Timer.h"
#include "Eigen/Dense"
#include "Projection.h"
#include "KinectFrame.h"
#include "KinectSpecs.h"
#include "Volumetric_helper.h"
#include "KinectFusionKernels/KinectFusionKernels.h"
#include <iostream>

static Timer timer;

RaycastImageWidget::RaycastImageWidget(QWidget* parent) : 
	QImageWidget(parent),
	raycastImage(KINECT_V2_DEPTH_WIDTH, KINECT_V2_DEPTH_HEIGHT, QImage::Format::Format_RGBA8888)
{
}

RaycastImageWidget::~RaycastImageWidget()
{
	cleanup();
}


void RaycastImageWidget::setup(
	const std::string& filepath, 
	ushort vx_count, 
	ushort vx_size)
{
	int vol_size = vx_count * vx_size;
	float half_vol_size = vol_size * 0.5f;

	this->voxel_count = vx_count;
	this->voxel_size = vx_size;
	this->cam_z_coord = -half_vol_size;
	this->fov = KINECT_V2_FOVY;


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
	unsigned short image_width = KINECT_V2_DEPTH_WIDTH;
	unsigned short image_height = image_width / aspect_ratio;

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

}

void RaycastImageWidget::cleanup()
{
	timer.start();
	knt_cuda_free();
	timer.print_interval("Cleanup gpu         : ");
}


void RaycastImageWidget::computeRaycast()
{
	int vol_size = voxel_count * voxel_size;
	float half_vol_size = vol_size * 0.5f;

	std::cout << "compute raycast " << std::endl;

	//
	// setup camera parameters
	//
	timer.start();
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	float cam_z = cam_z_coord;
	camera_to_world.scale(Eigen::Vector3f(1, 1, -1));
	camera_to_world.translate(Eigen::Vector3f(half_vol_size, half_vol_size, cam_z));

	Eigen::Matrix4f camera_to_world_matrix = camera_to_world.matrix();

	knt_cuda_raycast(fov, KINECT_V2_DEPTH_ASPECT_RATIO, camera_to_world_matrix.data());
	timer.print_interval("Raycast             : ");

	timer.start();
	knt_cuda_copy_image_device_to_host(*(uchar4*)raycastImage.bits());
	timer.print_interval("Copy img to host    : ");

	setImage(raycastImage);
}


void RaycastImageWidget::keyReleaseEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Q || e->key() == Qt::Key_Escape)
		this->close();

	if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_S)
	{
	}

	if (e->key() == Qt::Key_W)
	{
		cam_z_coord+=10;
		std::cout << cam_z_coord << std::endl;
		//raycastImage.fill(Qt::GlobalColor::gray);
		computeRaycast();
	}


	if (e->key() == Qt::Key_S)
	{
		cam_z_coord-=10;
		std::cout << cam_z_coord << std::endl;
		//raycastImage.fill(Qt::GlobalColor::gray);
		computeRaycast();
	}
}


void RaycastImageWidget::wheelEvent(QWheelEvent* event)
{
	fov += event->delta() * 0.01f;
	std::cout << fov << std::endl;
	event->accept();

	computeRaycast();
}


