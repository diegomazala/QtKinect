
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


#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.

const float fov_y = 70.0f;
const float window_width = 512.0f;
const float window_height = 424.0f;
const float near_plane = 0.1f; // 0.1f;
const float far_plane = 512.0f; // 10240.0f;
const float aspect_ratio = window_width / window_height;
Eigen::Matrix4d	K(Eigen::Matrix4d::Zero());
std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Zero(), Eigen::Matrix4d::Zero());



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

static void export_obj(const std::string& filename, const std::vector<Eigen::Vector3d>& points3D)
{
	std::ofstream file;
	file.open(filename);
	for (const auto X : points3D)
	{
		file << std::fixed << "v " << X.transpose() << std::endl;
	}
	file.close();
}



static void export_volume(const std::string& filename, const std::vector<Voxeld>& volume, const Eigen::Matrix4d& transformation = Eigen::Matrix4d::Identity())
{
	Eigen::Affine3d rotation;
	Eigen::Vector4d rgb;
	std::ofstream file;
	file.open(filename);
	for (const auto v : volume)
	{
		//rotation = Eigen::Affine3d::Identity();
		//rotation.rotate(Eigen::AngleAxisd(DegToRad(v.tsdf * 180.0), Eigen::Vector3d::UnitZ()));		// 90º
		//rgb = rotation.matrix() * (-Eigen::Vector4d::UnitX());
		//if (v.tsdf > -0.1 && v.tsdf < 0.1)
		file << std::fixed << "v " << (transformation * v.point.homogeneous()).head<3>().transpose() << ' ' << v.rgb.transpose() << std::endl;
	}
	file.close();
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


// Usage: ./Volumetricd.exe ../../data/plane.obj 256 4
int main(int argc, char **argv)
{
	const std::string filepath = argv[1];
	int vol_size = atoi(argv[2]);
	int vx_size = atoi(argv[3]);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3i volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());

	// Projection Matrix
	Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);

	// Modelview Matrix
	Eigen::Affine3d affine = Eigen::Affine3d::Identity();
	affine.translate(Eigen::Vector3d(0, 0, -192));
	T.first = affine.matrix();
	affine.rotate(Eigen::AngleAxisd(DegToRad(45.0), Eigen::Vector3d::UnitY()));		// 90º
	T.second = affine.matrix();

	std::vector<Eigen::Vector3d> points3D, points3DT;
	import_obj(filepath, points3D);

	// Creating depth buffer
	std::vector<double> depth_buffer(int(window_width * window_height), -1.0);
	for (Eigen::Vector3d p3d : points3D)
	{
		Eigen::Vector4d v = T.first * p3d.homogeneous();
		v /= v.w();

		points3DT.push_back(v.head<3>());

		const Eigen::Vector4d clip = K * v;
		const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();

		if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
			continue;

		Eigen::Vector3d pixel;
		pixel.x() = window_width / 2.0 * ndc.x() + window_width / 2.0;
		pixel.y() = window_height / 2.0 * ndc.y() + window_height / 2.0;
		pixel.z() = (far_plane - near_plane) / 2.0 * ndc.z() + (far_plane + near_plane) / 2.0;


		if (pixel.x() > 0 && pixel.x() < window_width &&
			pixel.y() > 0 && pixel.y() < window_height)
		{
			const int depth_index = (int)(pixel.y() * window_width + pixel.x());
			depth_buffer.at(depth_index) = v.z();
		}
	}

	// Creating volume
	std::size_t slice_size = (voxel_count.x() + 1) * (voxel_count.y() + 1);
	std::vector<Voxeld> tsdf_volume((voxel_count.x() + 1) * (voxel_count.y() + 1) * (voxel_count.z() + 1));
	Eigen::Matrix4d volume_transformation = Eigen::Matrix4d::Identity();
	volume_transformation.col(3) << -(volume_size.x() / 2.0), -(volume_size.y() / 2.0), -(volume_size.z() / 2.0), 1.0;	// set translate

	
	int i = 0;
	for (int z = 0; z <= volume_size.z(); z += voxel_size.z())
	{
		for (int y = 0; y <= volume_size.y(); y += voxel_size.y())
		{
			for (int x = 0; x <= volume_size.x(); x += voxel_size.x(), i++)
			{
				tsdf_volume[i].point = Eigen::Vector3d(x, y, z);
				tsdf_volume[i].rgb = Eigen::Vector3d(0, 0, 0);
				tsdf_volume[i].weight = i;
			}
		}
	}


	Eigen::Vector4d ti = T.first.col(3);

	double half_voxel = voxel_size.x() * 0.5;
	std::vector<Eigen::Vector3d> points3DVg, points3DV, pointsAll;

	// Sweeping volume
	for (auto it_volume = tsdf_volume.begin(); it_volume != tsdf_volume.end(); it_volume += slice_size)
	{
		auto z_slice_begin = it_volume;
		auto z_slice_end = it_volume + slice_size;

		for (auto it = z_slice_begin; it != z_slice_end; ++it)
		{

			// to world space
			Eigen::Vector4d vg = volume_transformation * it->point.homogeneous();	
			
			// to camera space
			Eigen::Vector4d v = T.first * vg;	// se for inversa não fica no clip space							
			//Eigen::Vector4d v = vg;
			v /= v.w();

			pointsAll.push_back(v.head<3>());

			// to camera space
			Eigen::Vector3d pixel = vertex_to_window_coord(v, fov_y, window_width / window_height, near_plane, far_plane, (int)window_width, (int)window_height);

			const Eigen::Vector4d clip = K * v;
			const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();
			
			if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
				continue;

			Eigen::Vector3d p_window;
			p_window.x() = window_width / 2.0 * ndc.x() + window_width / 2.0;
			p_window.y() = window_height / 2.0 * ndc.y() + window_height / 2.0;
			p_window.z() = (far_plane - near_plane) / 2.0 * ndc.z() + (far_plane + near_plane) / 2.0;

			//double sdf = (ti - vg).norm();
			double sdf = v.z();

			const int depth_pixel_index = (int)(p_window.y() * window_width + p_window.x());

			if (depth_pixel_index > depth_buffer.size())
				continue;


			const double Dp = depth_buffer.at(depth_pixel_index);

			double dist_v_cam = v.z();


			it->tsdf = dist_v_cam - Dp;
			
			if (it->tsdf > -half_voxel && it->tsdf < half_voxel)
				it->rgb = Eigen::Vector3d(0, 255, 0);
			else if (it->tsdf > 0)
				it->rgb = Eigen::Vector3d(0, 128, 255);
			else
				it->rgb = Eigen::Vector3d(255, 0, 0);
		}

	}

	export_volume("../../data/volume.obj", tsdf_volume, T.first * volume_transformation);

	return 0;
}





#if 0

template <typename T> T  lerp(const T& x0, const T& x1, const T& t) 
{
	return (1 - t) * x0 + t * x1;
}

template <typename T> T lerp(const T& y0, const T& y1, const T& x0, const T& x1, const T& x)
{
	return y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
}


double max_z = -9999999;
double min_z = FLT_MAX;

QImage color(window_width, window_height, QImage::Format_RGB888);
color.fill(Qt::white);

for (const Eigen::Vector3d p3d : points3D)
{
	Eigen::Vector4d clip = K * T.first * p3d.homogeneous();
	Eigen::Vector3d ndc = (clip / clip.w()).head<3>();

	if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
		continue;

	Eigen::Vector3d p_window;
	p_window.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
	p_window.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
	p_window.z() = (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

	double z = (ndc.z() * 0.5 + 0.5);

	if (max_z < z)
		max_z = z;

	if (min_z > z)
		min_z = z;

}

for (const Eigen::Vector3d p3d : points3D)
{
	Eigen::Vector4d clip = K * T.first * p3d.homogeneous();
	Eigen::Vector3d ndc = (clip / clip.w()).head<3>();

	if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
		continue;

	Eigen::Vector3d p_window;
	p_window.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
	p_window.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
	p_window.z() = (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

	double depth_z = (ndc.z() * 0.5 + 0.5);
	double z = lerp(0.0, 1.0, min_z, max_z, depth_z) * 255.0;

	if (p_window.x() > 0 && p_window.x() < window_width && p_window.y() > 0 && p_window.y() < window_height)
	{
		color.setPixel(QPoint(p_window.x(), window_height - p_window.y()), qRgb(255 - z, 0, 0));
	}
}

std::cout << "min max z: " << min_z << ", " << max_z << std::endl;
color.save("../../data/monkey_depth.png");
return 0;
#endif


#if 0
QImage color(window_width, window_height, QImage::Format_RGB888);
color.fill(Qt::white);
int count_off = 0;

// Sweeping volume
for (auto it_volume = tsdf_volume.begin(); it_volume != tsdf_volume.end(); it_volume += slice_size)
{
	auto z_slice_begin = it_volume;
	auto z_slice_end = it_volume + slice_size - 1;

	for (auto it = z_slice_begin; it != z_slice_end; ++it)
	{
		Eigen::Vector4d clip = K * T.first * (volume_transformation * it->point.homogeneous());
		Eigen::Vector3d ndc = (clip / clip.w()).head<3>();

		if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
		{
			std::cout << "out " << ndc.transpose() << std::endl;
			++count_off;
			continue;
		}

		Eigen::Vector3d p_window;
		p_window.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
		p_window.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
		p_window.z() = (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

		if (p_window.x() > 0 && p_window.x() < window_width && p_window.y() > 0 && p_window.y() < window_height)
		{
			color.setPixel(QPoint(p_window.x(), window_height - p_window.y()), qRgb(255, 0, 0));
		}
	}
}

for (const auto vx : tsdf_volume)
{
	Eigen::Vector4d clip = K * T.first * (volume_transformation * vx.point.homogeneous());
	Eigen::Vector3d ndc = (clip / clip.w()).head<3>();

	if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
	{
		std::cout << "out " << ndc.transpose() << std::endl;
		++count_off;
		continue;
	}

	Eigen::Vector3d p_window;
	p_window.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
	p_window.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
	p_window.z() = (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

	if (p_window.x() > 0 && p_window.x() < window_width && p_window.y() > 0 && p_window.y() < window_height)
	{
		color.setPixel(QPoint(p_window.x(), window_height - p_window.y()), qRgb(255, 0, 0));
	}
}

std::cout << "Count off " << count_off << std::endl;
color.save("../../data/volume_color.png");
return 0;
#endif


#if 0
std::vector<Eigen::Vector3d> points3DT, points3DProj;
double max_Z = -9999999;
double min_Z = FLT_MAX;
double max_ndc_Z = -9999999;
double min_ndc_Z = FLT_MAX;

// Creating depth buffer
std::vector<double> depth_buffer(int(window_width * window_height), -1.0);
for (Eigen::Vector3d p3d : points3D)
{
	Eigen::Vector4d v = T.first * p3d.homogeneous();
	v /= v.w();

	if (max_Z < v.z())
		max_Z = v.z();

	if (min_Z > v.z())
		min_Z = v.z();

	points3DT.push_back(v.head<3>());

	const Eigen::Vector4d clip = K * v;
	const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();

	if (max_ndc_Z < ndc.z())
		max_ndc_Z = ndc.z();

	if (min_ndc_Z > ndc.z())
		min_ndc_Z = ndc.z();

	points3DProj.push_back(ndc);

	Eigen::Vector3d pixel = vertex_to_window_coord(v, fov_y, window_width / window_height, near_plane, far_plane, (int)window_width, (int)window_height);
	if (pixel.x() > 0 && pixel.x() < window_width &&
		pixel.y() > 0 && pixel.y() < window_height)
	{
		const int depth_index = (int)(pixel.y() * window_width + pixel.x());
		depth_buffer.at(depth_index) = -v.z();
	}
}

std::cout << "min max Z     " << min_Z << ", " << max_Z << std::endl;
std::cout << "min max ndc Z " << min_ndc_Z << ", " << max_ndc_Z << std::endl;

export_obj("../../data/monkey_T.obj", points3DT);
export_obj("../../data/monkey_proj.obj", points3DProj);
#endif