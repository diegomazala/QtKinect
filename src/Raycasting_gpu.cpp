#include <QApplication>
#include "QImageWidget.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "RayBox.h"
#include "Timer.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#include "helper_cuda.h"
#include "helper_image.h"
#include "Kernels/RaycastKernels.h"

#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.

// Usage: ./Raycastingd.exe 3 1 1.5 1.5 -15 1.5 1.5 -10
int main(int argc, char **argv)
{
	int vol_size = atoi(argv[1]);
	int vx_size = atoi(argv[2]);

	Eigen::Vector3i voxel_count(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

	Eigen::Vector3f origin((float)atof(argv[3]), (float)atof(argv[4]), (float)atof(argv[5]));
	Eigen::Vector3f target((float)atof(argv[6]), (float)atof(argv[7]), (float)atof(argv[8]));
	Eigen::Vector3f direction = (target - origin).normalized();

	
	Timer t;
	t.start();
	raycast_one(origin.data(), direction.data(), voxel_count.data(), voxel_size.data());
	std::cout << "Time : " << t.diff_msec() << std::endl;

	int box_size = 1;
	Eigen::Affine3f box_transform = Eigen::Affine3f::Identity();
	box_transform.rotate(Eigen::AngleAxisf(DegToRad(30), Eigen::Vector3f::UnitY()));
	box_transform.rotate(Eigen::AngleAxisf(DegToRad(-20), Eigen::Vector3f::UnitX()));
	box_transform.rotate(Eigen::AngleAxisf(DegToRad(-10), Eigen::Vector3f::UnitZ()));
	box_transform.translate(Eigen::Vector3f(0.0f, 0.0f, 5.0f));

	float fovy = 60.0f;
	if (argc > 9)
		fovy = atof(argv[9]);

	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();

	if (argc > 10)
		camera_to_world.translate(Eigen::Vector3f(0, 0, atoi(argv[10])));


	int width = 256;
	int height = 256;
	uchar* image_data = new uchar[width * height * 3]; // rgb
	raycast_box(image_data, width, height, box_size, fovy, camera_to_world.matrix().data(), box_transform.matrix().data());

	QImage image(image_data, width, height, QImage::Format_RGB888);
	//image.fill(Qt::white);
	//image.save("../../data/raycast_box.png");
	

	QApplication app(argc, argv);
	QImageWidget widget;
	widget.setImage(image);
	widget.show();

	return app.exec();
}
