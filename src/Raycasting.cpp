
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


void test_quad_intersection()
{
	Eigen::Vector3f R1(0.0f, 0.0f, -1.0f);
	Eigen::Vector3f R2(0.0f, 0.0f, 1.0f);

	Eigen::Vector3f S1(-1.0f, 1.0f, 0.0f);
	Eigen::Vector3f S2(1.0f, 1.0f, 0.0f);
	Eigen::Vector3f S3(-1.0f, -1.0f, 0.0f);

	if (!quad_intersection<float>(R1, R2, S1, S2, S3))
	{
		std::cout << "something is wrong" << std::endl;
	}


	R1 = Eigen::Vector3f(1.5f, 1.5f, -1.0f);
	R2 = Eigen::Vector3f(1.5f, 1.5f, 1.0f);

	if (quad_intersection<float>(R1, R2, S1, S2, S3))
	{
		std::cout << "something is wrong" << std::endl;
	}
}

static Eigen::Vector3f eigen_fminf(Eigen::Vector3f a, Eigen::Vector3f b)
{
	return Eigen::Vector3f(fminf(a[0], b[0]), fminf(a[1], b[1]), fminf(a[2], b[2]));
};


static Eigen::Vector3f eigen_fmaxf(Eigen::Vector3f a, Eigen::Vector3f b)
{
	return Eigen::Vector3f(fmaxf(a[0], b[0]), fmaxf(a[1], b[1]), fmaxf(a[2], b[2]));
};

static Eigen::Vector3f eigen_mulf(Eigen::Vector3f a, Eigen::Vector3f b)
{
	return Eigen::Vector3f(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
};

int intersectBox(Ray<float> r, Eigen::Vector3f boxmin, Eigen::Vector3f boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	Eigen::Vector3f tbot = eigen_mulf(r.inv_direction, (boxmin - r.origin));
	Eigen::Vector3f ttop = eigen_mulf(r.inv_direction, (boxmax - r.origin));

	// re-order intersections to find smallest and largest on each axis
	Eigen::Vector3f tmin = eigen_fminf(ttop, tbot);
	Eigen::Vector3f tmax = eigen_fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x(), tmin.y()), fmaxf(tmin.x(), tmin.z()));
	float smallest_tmax = fminf(fminf(tmax.x(), tmax.y()), fminf(tmax.x(), tmax.z()));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}


static int get_index_test(Eigen::Vector3i pt, Eigen::Vector3i voxel_count)
{
	return pt.z() * voxel_count.x() * voxel_count.y() + pt.y() * voxel_count.y() + pt.x();
}

template <typename Type>
void raycast_test(
	const Eigen::Matrix<Type, 3, 1>& ray_origin,
	const Eigen::Matrix<Type, 3, 1>& ray_direction,
	const Eigen::Vector3i& voxel_count,
	const Eigen::Vector3i& voxel_size
	)
{

	Eigen::Vector3i volume_size(
		voxel_count.x() * voxel_size.x(), 
		voxel_count.y() * voxel_size.y(), 
		voxel_count.z() * voxel_size.z());

	Eigen::Matrix<Type, 3, 1> half_volume_size = volume_size.cast<Type>() * (Type)0.5;
	int total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();
	Type half_total_voxels = total_voxels / (Type)2.0;

	Eigen::Matrix<Type, 3, 1> half_voxel_size = voxel_size.cast<Type>() * (Type)0.5;
	Eigen::Matrix<Type, 3, 1> to_origin = (-volume_size.cast<Type>() * (Type)0.5);

	Eigen::Matrix<Type, 3, 1> hit1;
	Eigen::Matrix<Type, 3, 1> hit2;
	Eigen::Matrix<Type, 3, 1> hit1_normal;
	Eigen::Matrix<Type, 3, 1> hit2_normal;

	//
	// Check intersection with the whole volume
	//
	int intersections_count = box_intersection<Type>(
		ray_origin,
		ray_direction,
		half_volume_size,	//volume_center,
		volume_size.x(),
		volume_size.y(),
		volume_size.z(),
		hit1,
		hit2,
		hit1_normal,
		hit2_normal);

	Eigen::Vector3i hit_int = hit1.cast<int>();
	int voxel_index = get_index_test(hit_int, voxel_count);
	Eigen::Matrix<Type, 3, 1> last_voxel = hit_int.cast<Type>();

#if 0
	std::cout
		<< std::fixed << std::endl
		<< "Intersections   : " << intersections_count << std::endl
		<< "Origin          : " << ray_origin.transpose() << std::endl
		<< "Direction       : " << ray_direction.transpose() << std::endl
		<< std::endl
		<< "Hit In          : " << hit1.transpose() << std::endl
		<< "Hit Out         : " << hit2.transpose() << std::endl
		<< std::endl
		<< "Hit In Normal   : " << hit1_normal.transpose() << std::endl
		<< "Hit Out Normal  : " << hit2_normal.transpose() << std::endl
		<< std::endl;
#endif

	int loop_count = 0;
	std::cout << "First Intersected : " << voxel_index << std::endl;

	// 
	// Check intersection with each box inside of volume
	// 
	while (voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
	{
		intersections_count = box_intersection<Type>(
			hit1,
			ray_direction,
			last_voxel + half_voxel_size,
			voxel_size.x(),
			voxel_size.y(),
			voxel_size.z(),
			hit1,
			hit2,
			hit1_normal,
			hit2_normal);

		hit_int = hit1.cast<int>();
		voxel_index = get_index_test(hit_int, voxel_count);
		last_voxel = hit_int.cast<Type>();
		loop_count++;

#if 0
		std::cout
			<< std::fixed << std::endl
			<< "Intersections   : " << intersections_count << std::endl
			<< "Origin          : " << ray_origin.transpose() << std::endl
			<< "Direction       : " << ray_direction.transpose() << std::endl
			<< std::endl
			<< "Hit In          : " << hit1.transpose() << std::endl
			<< "Hit Out         : " << hit2.transpose() << std::endl
			<< std::endl
			<< "Hit In Normal   : " << hit1_normal.transpose() << std::endl
			<< "Hit Out Normal  : " << hit2_normal.transpose() << std::endl
			<< std::endl;
#endif
		std::cout << "Voxel Intersected : " << voxel_index << std::endl;
	}

}


static Eigen::Vector3i index_3d_from_array(
	int array_index, 
	const Eigen::Vector3i& voxel_count)
{
	return Eigen::Vector3i(
		int(std::fmod(array_index, voxel_count.x())),
		int(std::fmod(array_index / voxel_count.y(), voxel_count.y())),
		int(array_index / (voxel_count.x() * voxel_count.y())));
}

// 
// Face Index
// 0-Top, 1-Bottom, 2-Front, 3-Back, 4-Left, 5-Right
//
static int get_index_from_face(int face, int last_index, Eigen::Vector3i voxel_count)
{
	
	switch (face)
	{
		case 0: return last_index + voxel_count.x();					// Top
		case 1: return last_index - voxel_count.x();					// Bottom
		case 2: return last_index - voxel_count.x() * voxel_count.y();	// Front
		case 3: return last_index + voxel_count.x() * voxel_count.y();	// Back
		case 4: return last_index - 1;									// Left
		case 5: return last_index + 1;									// Right
		default: return -1;
	}
}

template <typename Type>
void raycast_test2(
	const Eigen::Matrix<Type, 3, 1>& ray_origin,
	const Eigen::Matrix<Type, 3, 1>& ray_direction,
	const Eigen::Vector3i& voxel_count,
	const Eigen::Vector3i& voxel_size
	)
{

	Eigen::Vector3i volume_size(
		voxel_count.x() * voxel_size.x(),
		voxel_count.y() * voxel_size.y(),
		voxel_count.z() * voxel_size.z());

	Eigen::Matrix<Type, 3, 1> half_volume_size = volume_size.cast<Type>() * (Type)0.5;
	int total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();
	Type half_total_voxels = total_voxels / (Type)2.0;

	Eigen::Matrix<Type, 3, 1> half_voxel_size = voxel_size.cast<Type>() * (Type)0.5;
	Eigen::Matrix<Type, 3, 1> to_origin = (-volume_size.cast<Type>() * (Type)0.5);

	Eigen::Matrix<Type, 3, 1> hit1;
	Eigen::Matrix<Type, 3, 1> hit2;
	Eigen::Matrix<Type, 3, 1> hit1_normal;
	Eigen::Matrix<Type, 3, 1> hit2_normal;

	//
	// Check intersection with the whole volume
	//
	int intersections_count = box_intersection<Type>(
		ray_origin,
		ray_direction,
		half_volume_size,	//volume_center,
		volume_size.x(),
		volume_size.y(),
		volume_size.z(),
		hit1,
		hit2,
		hit1_normal,
		hit2_normal);

	Eigen::Vector3i hit_int = hit1.cast<int>();
	int voxel_index = get_index_test(hit_int, voxel_count);
	Eigen::Matrix<Type, 3, 1> last_voxel = hit_int.cast<Type>();


	int loop_count = 0;
	std::cout << "First Intersected : " << voxel_index << std::endl;

	// 
	// Check intersection with each box inside of volume
	// 
	while (voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
	{

		int face = -1;
		intersections_count = box_intersection<Type>(
			ray_origin,
			ray_direction,
			last_voxel + half_voxel_size,
			voxel_size.x(),
			voxel_size.y(),
			voxel_size.z(),
			hit1_normal,
			hit2_normal,
			face);


		voxel_index = get_index_from_face(face, voxel_index, voxel_count);
		Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count);
		loop_count++;

#if 0
		std::cout
			<< std::fixed << std::endl
			<< "Intersections   : " << intersections_count << std::endl
			<< "Origin          : " << ray_origin.transpose() << std::endl
			<< "Direction       : " << ray_direction.transpose() << std::endl
			<< std::endl
			<< "Hit In Normal   : " << hit1_normal.transpose() << std::endl
			<< "Hit Out Normal  : " << hit2_normal.transpose() << std::endl
			<< "Face            : " << face << std::endl
			<< std::endl;
#endif
		std::cout << "Voxel Intersected : " << voxel_index << std::endl;
	}

}

// Usage: ./Raycastingd.exe 3 1 1.5 1.5 -15 1.5 1.5 -10
int main(int argc, char **argv)
{
	int vol_size = atoi(argv[1]); 
	int vx_size = atoi(argv[2]);
	
	Eigen::Vector3i voxel_count(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);
	
	Eigen::Vector3f origin(atof(argv[3]), atof(argv[4]), atof(argv[5]));
	Eigen::Vector3f target(atof(argv[6]), atof(argv[7]), atof(argv[8]));
	Eigen::Vector3f direction = (target - origin).normalized();
	
	Timer t;
	t.start();
	raycast_test(origin, direction, voxel_count, voxel_size);
	double t1 = t.diff_msec();
	t.start();
	raycast_test2(origin, direction, voxel_count, voxel_size);
	double t2 = t.diff_msec();

	std::cout << "Time using triangle intersection : " << t1 << std::endl;
	std::cout << "Time using quad intersection     : " << t2 << std::endl;

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
