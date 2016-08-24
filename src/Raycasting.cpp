
#include <QApplication>
#include <QKeyEvent>
#include <QPushButton>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "Interpolator.hpp"
#include "RayBox.h"
#include "Grid.h"
#include "Timer.h"
#include "RayIntersection.h"



template <typename Type>
void raycast_grid_test(
	const Eigen::Matrix<Type, 3, 1>& ray_origin,
	const Eigen::Matrix<Type, 3, 1>& ray_direction,
	const Eigen::Matrix<Type, 3, 1>& volume_center,
	const Eigen::Matrix<Type, 3, 1>& volume_size,
	const Eigen::Matrix<Type, 3, 1>& voxel_size
	)
{
	Eigen::Matrix<Type, 3, 1> hit1;
	Eigen::Matrix<Type, 3, 1> hit2;
	Eigen::Matrix<Type, 3, 1> hit1_normal;
	Eigen::Matrix<Type, 3, 1> hit2_normal;

	int intersections_count = box_intersection<Type>(
		ray_origin, 
		ray_direction, 
		volume_center, 
		volume_size.x(),
		volume_size.y(),
		volume_size.z(),
		hit1, 
		hit2, 
		hit1_normal, 
		hit2_normal);

	std::cout
		<< std::fixed << std::endl
		<< "Intersections: " << intersections_count << std::endl
		<< "Origin       : " << ray_origin.transpose() << std::endl
		<< "Direction    : " << ray_direction.transpose() << std::endl
		<< "Hit1         : " << hit1.transpose() << std::endl
		<< "Hit2         : " << hit2.transpose() << std::endl
		<< "Hit1 Normal  : " << hit1_normal.transpose() << std::endl
		<< "Hit2 Normal  : " << hit2_normal.transpose() << std::endl
		<< std::endl;


}



template <typename Type>
void raycast_test(const Eigen::Matrix<Type, 3, 1>& o,
	const Eigen::Matrix<Type, 3, 1>& d,
	const Eigen::Matrix<Type, 3, 1>& box_center,
	const Eigen::Matrix<Type, 3, 1>& voxel_size,
	Eigen::Matrix<Type, 3, 1>& hit1,
	Eigen::Matrix<Type, 3, 1>& hit2,
	Eigen::Matrix<Type, 3, 1>& hit1_normal,
	Eigen::Matrix<Type, 3, 1>& hit2_normal)
{
	Timer t;
	t.start();
	int count = box_intersection<Type>(o, d, box_center, voxel_size.x(), voxel_size.y(), voxel_size.z(), hit1, hit2, hit1_normal, hit2_normal);
	std::cout << "Time : " << t.diff_msec() << std::endl;

	std::cout
		<< "Intersections: " << count << std::endl
		<< "Origin       : " << o.transpose() << std::endl
		<< "Direction    : " << d.transpose() << std::endl
		<< "Hit1         : " << hit1.transpose() << std::endl
		<< "Hit2         : " << hit2.transpose() << std::endl
		<< "Hit1 Normal  : " << hit1_normal.transpose() << std::endl
		<< "Hit2 Normal  : " << hit2_normal.transpose() << std::endl
		<< std::endl;


	Ray<Type> r(o, d);
	Eigen::Matrix<Type, 3, 1> bmin((box_center - voxel_size) * (Type)0.5);
	Eigen::Matrix<Type, 3, 1> bmax((box_center + voxel_size) * (Type)0.5);
	Box<Type> box(bmin, bmax);
	std::cout << "--------------" << std::endl
		<< box.intersect(r, 0, 100) << std::endl;
}


// Usage: ./Raycastingd.exe 2 1 0 2 - 3 0.72 - 1.2 2 7
int main(int argc, char **argv)
{

	int vol_size = atoi(argv[1]); //16;
	int vx_size = atoi(argv[2]); //1;
	
	Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);

	Grid<double> grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());
	
	Eigen::Vector3d half_voxel(vx_size * 0.5, vx_size * 0.5, vx_size * 0.5);
	
	float ray_near = 0; // atof(argv[1]);
	float ray_far = 100; // atof(argv[2]);

	Eigen::Vector3d cube(0, 0, 0);

	Eigen::Vector3d origin(atof(argv[3]), atof(argv[4]), atof(argv[5]));
	Eigen::Vector3d target(atof(argv[6]), atof(argv[7]), atof(argv[8]));
	Eigen::Vector3d direction = (target - origin).normalized();
	
	
	
	Eigen::Vector3d center(0, 0, 0);

	Eigen::Matrix<double, 3, 1> hit1;
	Eigen::Matrix<double, 3, 1> hit2;
	Eigen::Matrix<double, 3, 1> hit1_normal;
	Eigen::Matrix<double, 3, 1> hit2_normal;
	raycast_grid_test<double>(origin, direction, center, volume_size, voxel_size);
	return 0;


	int voxel_index = atoi(argv[9]);
	std::vector<int> intersections;
	
	Timer t;
	t.start();
	grid.recursive_raycast(-1, voxel_index, origin, direction, ray_near, ray_far, intersections);
	std::cout << "\nIntersections Recursive: " << intersections.size() << std::endl;
	for (const auto i : intersections)
		std::cout << i << " : " << grid.data[i].point.transpose() << std::endl;
	std::cout << "Time : " << t.diff_msec() << std::endl;

	t.start();
	intersections = grid.raycast_all(origin, direction, ray_near, ray_far);
	std::cout << "\nIntersections All: " << intersections.size() << std::endl;
	for (const auto i : intersections)
		std::cout << i << " : " << grid.data[i].point.transpose() << std::endl;
	std::cout << "Time : " << t.diff_msec() << std::endl;

	t.start();
	//intersections = Grid::find_intersections(grid.data, grid.volume_size, grid.voxel_size, grid.transformation, origin, direction, ray_near, ray_far);
	intersections = Grid<double>::find_intersections(grid.data, grid.volume_size, grid.voxel_size, Eigen::Matrix4d::Identity(), origin, direction, ray_near, ray_far);
	std::cout << "\nIntersections Find: " << intersections.size() << std::endl;
	for (const auto i : intersections)
		std::cout << i << " : " << grid.data[i].point.transpose() << std::endl;
	std::cout << "Time : " << t.diff_msec() << std::endl;

	t.start();
	Grid<double>::sort_intersections(intersections, grid.data, origin);
	std::cout << "\nIntersections Find Sorted: " << intersections.size() << std::endl;
	for (const auto i : intersections)
		std::cout << i << " : " << grid.data[i].point.transpose() << std::endl;
	std::cout << "Time : " << t.diff_msec() << std::endl;

	return 0;
}


#if 0
const float window_width = 512.0f;
const float window_height = 424.0f;
QImage image(window_width, window_height, QImage::Format_RGB888);
image.fill(Qt::white);

Eigen::Vector3d window_coord_norm;

std::cout << "Ray casting to image..." << std::endl;
// sweeping the image
for (int y = 0; y < window_height; ++y)
{
	std::cout << "Ray casting to image... " << (double)y / window_height * 100 << "%" << std::endl;
	for (int x = 0; x < window_width; ++x)
	{
		window_coord_norm.x() = ((double)x / window_width * 2.0) - 1.0;
		window_coord_norm.y() = ((double)y / window_height * 2.0) - 1.0;;
		window_coord_norm.z() = origin.z() + near_plane;

		direction = (window_coord_norm - origin).normalized();

		for (const Voxeld v : volume)
		{
			Eigen::Vector3d corner_min = (volume_transformation * (v.point - half_voxel).homogeneous()).head<3>();
			Eigen::Vector3d corner_max = (volume_transformation * (v.point + half_voxel).homogeneous()).head<3>();

			Box box(corner_min, corner_max);
			Ray ray(origin, direction);


			if (box.intersect(ray, ray_near, ray_far))
			{
				//std::cout << "Box Intersected: " << v.point.transpose() << std::endl;
				//image.setPixel(QPoint(x, window_height - y), qRgb(255, 0, 0)); 
				image.setPixel(QPoint(x, y), qRgb(255, 0, 0));
			}
		}
	}
}

std::cout << "Saving to image..." << std::endl;
image.save("raycasting.png");
#endif
