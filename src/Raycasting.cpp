
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


#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.

const float fov_y = 70.0f;
const float window_width = 512.0f / 12.0f;
const float window_height = 424.0f / 12.0f;
const float near_plane = 0.1f; // 0.1f;
const float far_plane = 512.0f; // 10240.0f;
const float aspect_ratio = window_width / window_height;
Eigen::Matrix4d	K(Eigen::Matrix4d::Zero());
std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Zero(), Eigen::Matrix4d::Zero());


static bool import_obj(const std::string& filename, std::vector<Eigen::Vector3d>& points3D, int max_point_count = INT_MAX)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	points3D.clear();

	int i = 0;
	while (inFile)
	{
		std::string str;

		if (!std::getline(inFile, str))
		{
			if (inFile.eof())
				return true;

			std::cerr << "Error: Problems when reading obj file: " << filename << std::endl;
			return false;
		}

		if (str[0] == 'v')
		{
			std::stringstream ss(str);
			std::vector <std::string> record;

			char c;
			double x, y, z;
			ss >> c >> x >> y >> z;

			Eigen::Vector3d p(x, y, z);
			points3D.push_back(p);
		}

		if (i++ > max_point_count)
			break;
	}

	inFile.close();
	return true;
}


static Eigen::Matrix4d perspective_matrix(double fovy, double aspect_ratio, double near_plane, double far_plane)
{
	Eigen::Matrix4d out = Eigen::Matrix4d::Zero();

	const double	y_scale = 1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
	const double	x_scale = y_scale / aspect_ratio;
	const double	depth_length = far_plane - near_plane;

	out(0, 0) = x_scale;
	out(1, 1) = y_scale;
	out(2, 2) = -((far_plane + near_plane) / depth_length);
	out(3, 2) = -1.0;
	out(2, 3) = -((2 * near_plane * far_plane) / depth_length);

	return out;
}


static Eigen::Matrix4d perspective_matrix_inverse(double fovy, double aspect_ratio, double near_plane, double far_plane)
{
	Eigen::Matrix4d out = Eigen::Matrix4d::Zero();

	const double	y_scale = 1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
	const double	x_scale = y_scale / aspect_ratio;
	const double	depth_length = far_plane - near_plane;

	out(0, 0) = 1.0 / x_scale;
	out(1, 1) = 1.0 / y_scale;
	out(2, 3) = -1.0;
	out(3, 2) = -1.0 / ((2 * near_plane * far_plane) / depth_length);
	out(3, 3) = ((far_plane + near_plane) / depth_length) / ((2 * near_plane * far_plane) / depth_length);

	return out;
}



Eigen::Vector3d vertex_to_window_coord(Eigen::Vector4d p3d, double fovy, double aspect_ratio, double near_plane, double far_plane, int window_width, int window_height)
{
	const Eigen::Matrix4d proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

	const Eigen::Vector4d clip = proj * p3d;

	const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();

	Eigen::Vector3d pixel;
	pixel.x() = window_width / 2.0 * ndc.x() + window_width / 2.0;
	pixel.y() = window_height / 2.0 * ndc.y() + window_height / 2.0;
	pixel.z() = (far_plane - near_plane) / 2.0 * ndc.z() + (far_plane + near_plane) / 2.0;

	return pixel;
}


Eigen::Vector3d window_coord_to_3d(Eigen::Vector2d pixel, double depth, double fovy, double aspect_ratio, double near_plane, double far_plane, int window_width, int window_height)
{
	Eigen::Vector3d ndc;
	ndc.x() = (pixel.x() - (window_width / 2.0)) / (window_width / 2.0);
	ndc.y() = (pixel.y() - (window_height / 2.0)) / (window_height / 2.0);
	ndc.z() = -1.0f;

	const Eigen::Vector3d clip = ndc * depth;

	const Eigen::Matrix4d proj_inv = perspective_matrix_inverse(fovy, aspect_ratio, near_plane, far_plane);
	const Eigen::Vector4d vertex_proj_inv = proj_inv * clip.homogeneous();

	Eigen::Vector3d p3d_final;
	p3d_final.x() = -vertex_proj_inv.x();
	p3d_final.y() = -vertex_proj_inv.y();
	p3d_final.z() = depth;

	return p3d_final;
}


class Ray 
{
public:
	Ray(Eigen::Vector3d &o, Eigen::Vector3d &d)
	{
		origin = o;
		direction = d;
		inv_direction = Eigen::Vector3d(1 / d.x(), 1 / d.y(), 1 / d.z());
		sign[0] = (inv_direction.x() < 0);
		sign[1] = (inv_direction.y() < 0);
		sign[2] = (inv_direction.z() < 0);
	}
	Eigen::Vector3d origin;
	Eigen::Vector3d direction;
	Eigen::Vector3d inv_direction;
	int sign[3];
};


class Box
{
public:
	Box(const Eigen::Vector3d &vmin, const Eigen::Vector3d &vmax)
	{
		bounds[0] = vmin;
		bounds[1] = vmax;
	}

	bool intersect(const Ray &r, float t0, float t1) const
	{
		float tmin, tmax, tymin, tymax, tzmin, tzmax;
		tmin = (bounds[r.sign[0]].x() - r.origin.x()) * r.inv_direction.x();
		tmax = (bounds[1 - r.sign[0]].x() - r.origin.x()) * r.inv_direction.x();
		tymin = (bounds[r.sign[1]].y() - r.origin.y()) * r.inv_direction.y();
		tymax = (bounds[1 - r.sign[1]].y() - r.origin.y()) * r.inv_direction.y();
		if ((tmin > tymax) || (tymin > tmax))
			return false;
		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;
		tzmin = (bounds[r.sign[2]].z() - r.origin.z()) * r.inv_direction.z();
		tzmax = (bounds[1 - r.sign[2]].z() - r.origin.z()) * r.inv_direction.z();
		if ((tmin > tzmax) || (tzmin > tmax))
			return false;
		if (tzmin > tmin)
			tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;
		return ((tmin < t1) && (tmax > t0));
	}

	Eigen::Vector3d bounds[2];
};



template<typename Type>
struct Voxel
{
	Eigen::Matrix<Type, 3, 1> point;
	Eigen::Matrix<Type, 3, 1> rgb;
	Type tsdf;
	Type weight;

	Voxel() :tsdf(FLT_MAX), weight(0.0){}
};
typedef Voxel<double> Voxeld;
typedef Voxel<float> Voxelf;


void createVolume(const Eigen::Vector3d& volume_size, const Eigen::Vector3d& voxel_size, std::vector<Voxeld>& volume)
{
	// Creating volume
	
	Eigen::Vector3i voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	std::size_t slice_size = (voxel_count.x() + 1) * (voxel_count.y() + 1);
	volume.clear();
	volume.resize((voxel_count.x() + 1) * (voxel_count.y() + 1) * (voxel_count.z() + 1));
	Eigen::Matrix4d volume_transformation = Eigen::Matrix4d::Identity();
	volume_transformation.col(3) << -(volume_size.x() / 2.0), -(volume_size.y() / 2.0), -(volume_size.z() / 2.0), 1.0;	// set translate


	int i = 0;
	for (int z = 0; z <= volume_size.z(); z += voxel_size.z())
	{
		for (int y = 0; y <= volume_size.y(); y += voxel_size.y())
		{
			for (int x = 0; x <= volume_size.x(); x += voxel_size.x(), i++)
			{
				volume[i].point = Eigen::Vector3d(x, y, z);
				volume[i].rgb = Eigen::Vector3d(0, 0, 0);
				volume[i].weight = i;
			}
		}
	}
}


int main(int argc, char **argv)
{
	int vol_size = 16; 
	int vx_size = 1; 

	std::vector<Voxeld> volume;
	createVolume(Eigen::Vector3d(vol_size, vol_size, vol_size), Eigen::Vector3d(vx_size, vx_size, vx_size), volume);
	Eigen::Vector3d half_voxel(vx_size * 0.5, vx_size * 0.5, vx_size * 0.5);

	float t0 = atof(argv[1]);
	float t1 = atof(argv[2]);

	Eigen::Vector3d cube(0, 0, 0);

	Eigen::Vector3d origin(0, 0, -32);
	Eigen::Vector3d target(0, 0, -20);
	//Eigen::Vector3d origin(-12, 0, -27);
	//Eigen::Vector3d target(6.1, 1.01, -20);
	Eigen::Vector3d direction = (target - origin).normalized();

	Eigen::Matrix4d volume_transformation = Eigen::Matrix4d::Identity();
	volume_transformation.col(3) << -vol_size / 2.0, -vol_size / 2.0, -vol_size / 2.0, 1.0;	// set translate

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
			window_coord_norm.x() = ((double)x / window_width * 2.0) -1.0;
			window_coord_norm.y() = ((double)y / window_height * 2.0) - 1.0;;
			window_coord_norm.z() = origin.z() + near_plane;

			direction = (window_coord_norm - origin).normalized();


			for (const Voxeld v : volume)
			{
				Eigen::Vector3d corner_min = (volume_transformation * (v.point - half_voxel).homogeneous()).head<3>();
				Eigen::Vector3d corner_max = (volume_transformation * (v.point + half_voxel).homogeneous()).head<3>();

				Box box(corner_min, corner_max);
				Ray ray(origin, direction);


				if (box.intersect(ray, t0, t1))
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


	return 0;
}


