
#include "QRaycastImageWidget.h"
#include <QMouseEvent>
#include <QPainter>
#include <QFileDialog>
#include <iostream>

#include <math.h>
#include "Timer.h"
#include "Eigen/Dense"
#include "Projection.h"
#include "KinectFrame.h"
#include "KinectSpecs.h"
#include "Volumetric_helper.h"
#include "KinectFusionKernels/KinectFusionKernels.h"

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

static Timer ttimer;
static ushort g_width = KINECT_V2_DEPTH_WIDTH;
static ushort g_height = KINECT_V2_DEPTH_HEIGHT;



QRaycastImageWidget::QRaycastImageWidget(QWidget *parent) :
	QImageWidget(parent),
	altPressed(false),
	angularSpeed(0),
	lightDirection(0, 0, -1),
	lightSpecPower(1),
	fovy(60.0f),
	nearPlane(0.1f),
	farPlane(1024.f)
{
	timer.start(12, this);

	weelSpeed = 0.01f;
	cameraPosition.setZ(-32);
}

QRaycastImageWidget::~QRaycastImageWidget()
{
	cleanup();

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}

void QRaycastImageWidget::setImage(const QImage& image)
{
	raycastImage = image;
	QImageWidget::setImage(raycastImage);
}




void QRaycastImageWidget::computeRaycast()
{
	float half_vol_size = vol_size * 0.5f;
	unsigned short image_width = g_width;
	unsigned short image_height = image_width / aspect_ratio;

	ttimer.start();
	Eigen::Affine3f view_affine = Eigen::Affine3f::Identity();
	view_affine.translate(Eigen::Vector3f(cameraPosition.x(), cameraPosition.y(), cameraPosition.z()));
	view_affine.rotate(Eigen::Quaternionf(cameraRotation.scalar(), cameraRotation.x(), cameraRotation.y(), cameraRotation.z()));
	Eigen::Matrix4f view_matrix = view_affine.matrix().inverse();

	Eigen::Matrix4f mdv_inv_trasp = view_matrix.inverse().transpose();
	Eigen::Matrix3f normal_matrix = mdv_inv_trasp.block(0, 0, 3, 3);
	knt_set_normal_matrix(normal_matrix.data());

	if (altPressed)
		lightDirection = (lightRotation * QVector3D(0, -1, -1)).normalized();

	float4 lightData = make_float4(lightDirection.x(), lightDirection.y(), lightDirection.z(), lightSpecPower);
	knt_set_light(lightData);
	
	//system("CLS");
	//std::cout << view_matrix << std::endl << std::endl;
	//qDebug() << "c: " << cameraPosition << '\t' << cameraRotationAxis << "  l: " << lightRotationAxis << " : " << lightSpecPower;
	//QMatrix4x4 view_matrix;
	//view_matrix.translate(cameraPosition);
	//view_matrix.rotate(cameraRotation);
	//QMatrix4x4 view_matrix_inv = view_matrix.transposed().inverted();

	
	knt_cuda_raycast(KINECT_V2_FOVY, KINECT_V2_DEPTH_ASPECT_RATIO, view_matrix.data());
	//knt_cuda_raycast(KINECT_V2_FOVY, KINECT_V2_DEPTH_ASPECT_RATIO, view_matrix_inv.data());
	//ttimer.print_interval("Raycast             : ");

	//ttimer.start();
	knt_cuda_copy_image_device_to_host(*(uchar4*)raycastImage.bits());
	//ttimer.print_interval("Copy Img to host    : ");
}




void QRaycastImageWidget::setup(const std::string& filepath, ushort vx_count, ushort vx_size)
{
	vol_size = vx_count * vx_size;
	float half_vol_size = vol_size * 0.5f;

	weelSpeed = vx_size * 0.01f;

	this->voxel_count = vx_count;
	this->voxel_size = vx_size;
	this->fovy = KINECT_V2_FOVY;

	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3i volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
	ulong total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();


	std::cout << std::fixed
		<< "Voxel Count  : " << voxel_count.transpose() << std::endl
		<< "Voxel Size   : " << voxel_size.transpose() << std::endl
		<< "Volume Size  : " << volume_size.transpose() << std::endl
		<< "Total Voxels : " << total_voxels << std::endl
		<< std::endl;

	ttimer.start();
	KinectFrame knt(filepath);
	ttimer.print_interval("Importing knt frame : ");

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
	unsigned short image_width = g_width;
	unsigned short image_height = image_width / aspect_ratio;

	knt_cuda_setup(
		vx_count, vx_size,
		grid_matrix.data(),
		projection.data(),
		projection_inverse.data(),
		*grid_voxels_params.data()->data(),
		g_width,
		g_height,
		KINECT_V2_DEPTH_MIN,
		KINECT_V2_DEPTH_MAX,
		vertices.data()[0],
		normals.data()[0],
		image_width,
		image_height
		);

	ttimer.start();
	knt_cuda_allocate();
	knt_cuda_init_grid();
	ttimer.print_interval("Allocating gpu      : ");

	//
	// use this for box 
	// 
	//cameraPosition.setX(0);
	//cameraPosition.setY(0);
	//cameraPosition.setZ(2.f * vol_size);
	//
	// use this for volume
	// 
	cameraPosition.setX(-vol_size * 0.5f);
	cameraPosition.setY(-vol_size * 0.5f);
	cameraPosition.setZ(0);

#if 0
	//
	// use this with vx_count = 3 and vx_size = 1 to debug
	// 
	std::vector<Eigen::Vector2f> tsdf(total_voxels, Eigen::Vector2f::Ones());

	// bottom face
	for (ulong z = 0; z < vx_count; ++z)
	{
		for (ulong x = 0; x < vx_count; ++x)
		{
			tsdf.at(z * vx_count * vx_count + x)[0] = -1.f;
		}
	}

	// half front face
	ulong half_side = vx_count * vx_count / 2;
	for (ulong x = 0; x < half_side; ++x)
	{
		tsdf.at(x)[0] = -1.f;
	}

	// right face
	ulong vx_count_2 = vx_count * vx_count;
	for (ulong z = 0; z < total_voxels; z += vx_count_2)
	{
		ulong begin = z + vx_count - 1;
		ulong end = z + vx_count_2;
		for (ulong i = begin; i < end; i += vx_count)
		{
			tsdf.at(i)[0] = -1.f;
		}
	}

	// last voxel
	tsdf.at(vx_count*vx_count*vx_count - 1)[0] = -1.f;

	knt_cuda_grid_sample_test(tsdf.data()->data(), tsdf.size());
#else
	ttimer.start();
	knt_cuda_copy_host_to_device();
	knt_cuda_copy_depth_buffer_to_device(knt.depth.data());
	ttimer.print_interval("Copy host to device : ");

	ttimer.start();
	knt_cuda_normal_estimation();
	ttimer.print_interval("Normal estimation   : ");

	ttimer.start();
	knt_cuda_update_grid(view_matrix.data());
	ttimer.print_interval("Update grid         : ");
#endif

	raycastImage = QImage(image_width, image_height, QImage::Format::Format_RGBA8888);

	////ttimer.start();
	//Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	//float cam_z = -half_vol_size;
	//camera_to_world.scale(Eigen::Vector3f(1, 1, -1));
	//camera_to_world.translate(Eigen::Vector3f(half_vol_size, half_vol_size, cam_z));

	//Eigen::Matrix4f camera_to_world_matrix = camera_to_world.matrix().inverse();

	//knt_cuda_raycast(KINECT_V2_FOVY, KINECT_V2_DEPTH_ASPECT_RATIO, camera_to_world_matrix.data());
	////ttimer.print_interval("Raycast             : ");

	////ttimer.start();
	//knt_cuda_copy_image_device_to_host(*(uchar4*)raycastImage.bits());
	////ttimer.print_interval("Copy Img to host    : ");
}


void QRaycastImageWidget::cleanup()
{
	ttimer.start();
	knt_cuda_free();
	ttimer.print_interval("Cleanup gpu         : ");
}

void QRaycastImageWidget::paintEvent(QPaintEvent* event)
{
	computeRaycast();

	setPixmap(QPixmap::fromImage(raycastImage).scaled(width(), height(), Qt::KeepAspectRatio));
	QLabel::paintEvent(event); //<<<<<<<<<<<<<<<
}










void QRaycastImageWidget::mousePressEvent(QMouseEvent *e)
{
	// Save mouse press position
	mousePressPosition = QVector2D(e->localPos());
	angularSpeed = 0.0;
}


void QRaycastImageWidget::mouseMoveEvent(QMouseEvent *e)
{
	// Mouse release position - mouse press position
	QVector2D diff = QVector2D(e->localPos()) - mousePressPosition;

	// Rotation axis is perpendicular to the mouse position difference
	// vector
	QVector3D n = QVector3D(diff.y(), diff.x(), 0.0).normalized();

	// Accelerate angular speed relative to the length of the mouse sweep
	qreal acc = diff.length() / 50.0;

	// Calculate new rotation axis as weighted sum
	//rotationAxis = (rotationAxis * angularSpeed + n * acc).normalized();
	if (!altPressed)
		cameraRotationAxis = n.normalized() * acc;
	else
		lightRotationAxis = n.normalized() * acc;

	// Increase angular speed
	//angularSpeed += acc;
	angularSpeed = acc * 10.0;

	mousePressPosition = QVector2D(e->localPos());
}


void QRaycastImageWidget::mouseReleaseEvent(QMouseEvent *e)
{
	e->accept();
}


void QRaycastImageWidget::wheelEvent(QWheelEvent* event)
{
	if (!altPressed)
		cameraPosition.setZ(cameraPosition.z() - event->delta() * weelSpeed);
	else
		lightSpecPower = fmax(0.f, fmin(1.0, (lightSpecPower - event->delta() * 0.0001f)));

	event->accept();

	update();
}


void QRaycastImageWidget::timerEvent(QTimerEvent *)
{
	// Decrease angular speed (friction)
	angularSpeed *= 0.5;

	// Stop rotation when speed goes below threshold
	if (angularSpeed < 0.01) 
	{
		angularSpeed = 0.0;
	}
	else 
	{
		// Update rotation
		if (!altPressed)
			cameraRotation = QQuaternion::fromAxisAndAngle(cameraRotationAxis, angularSpeed) * cameraRotation;
		else
			lightRotation = QQuaternion::fromAxisAndAngle(lightRotationAxis, angularSpeed) * lightRotation;

		// Request an update
		update();
	}
}

void QRaycastImageWidget::keyPressEvent(QKeyEvent *e)
{
	altPressed = (e->modifiers() == Qt::AltModifier);
}

void QRaycastImageWidget::keyReleaseEvent(QKeyEvent *e)
{
	altPressed = false;


	if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_S)
	{
		const QString& filename = QFileDialog::getSaveFileName(this,
			tr("Save Image File"), "../../data/",
			tr("Images (*.png);;Images (*.jpg)"));

		this->save(filename);
		return;
	}

	switch (e->key())
	{
		case Qt::Key_Q:
		case Qt::Key_Escape:
		{
			this->close();
			break;
		}

		case Qt::Key_W:
		{
			cameraPosition.setY(cameraPosition.y() - vol_size * 0.05f);
			break;
		}

		case Qt::Key_S:
		{
			cameraPosition.setY(cameraPosition.y() + vol_size * 0.05f);
			break;
		}

		case Qt::Key_A:
		{
			cameraPosition.setX(cameraPosition.x() + vol_size * 0.05f);
			break;
		}
		case Qt::Key_D:
		{
			cameraPosition.setX(cameraPosition.x() - vol_size * 0.05f);
			break;
		}
		case Qt::Key_Left:
		{
			lightDirection = QVector3D(-1, -1, 0);
			break;
		}
		case Qt::Key_Right:
		{
			lightDirection = QVector3D(1, -1, 0);
			break;
		}
		case Qt::Key_Up:
		{
			lightDirection = QVector3D(0, -1, -1);
			break;
		}
		case Qt::Key_Down:
		{
			lightDirection = QVector3D(0, -1, 1);
			break;
		}
		case Qt::Key_Insert:
		{
			lightDirection = QVector3D(0, 1, -1);
			break;
		}
		case Qt::Key_Delete:
		{
			lightDirection = QVector3D(0, 1, 1);
			break;
		}
		default:
		{
			e->ignore(); // let the base class handle this event
		}
	}
	qDebug() << "Light dir: " << lightDirection;
}



void QRaycastImageWidget::resizeEvent(QResizeEvent *event)
{
	projection.setToIdentity();	// Reset projection
	projection.perspective(
		fovy, 
		(float)this->width() / (float)this->height(), 
		nearPlane, 
		farPlane);	// Set perspective projection
}
