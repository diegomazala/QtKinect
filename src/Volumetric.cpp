
#include <QApplication>
#include <QKeyEvent>
#include <QPushButton>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include "Grid.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "Timer.h"
#include "Projection.h"


#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.

const double fov_y = 70.0f;
const double window_width = 512.0f;
const double window_height = 424.0f;
const double near_plane = 0.1f; // 0.1f;
const double far_plane = 512.0f; // 10240.0f;
const double aspect_ratio = window_width / window_height;
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

static void export_obj(const std::string& filename, const std::vector<Eigen::Vector3d>& points3D)
{
	std::ofstream file;
	file.open(filename);
	for (const auto X : points3D)
	{
		if (!X.isApprox(Eigen::Vector3d(0,0,0)))
			file << std::fixed << "v " << X.transpose() << std::endl;
	}
	file.close();
}


static void export_obj(const std::string& filename, const std::vector<Eigen::Vector3d>& points3D, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view)
{
	std::ofstream file;
	file.open(filename);
	for (const auto X : points3D)
	{
		Eigen::Vector4d clip = proj * view * X.homogeneous();
		const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();
		Eigen::Vector3f pixel;
		pixel.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
		pixel.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
		pixel.z() = 0.0; // (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

		file << std::fixed << "v " << pixel.transpose() << std::endl;
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
		//int r = v.rgb.x();
		//int b = v.rgb.z();
		//if (r != 255 && b != 255)
		file << std::fixed << "v " << (transformation * v.point.homogeneous()).head<3>().transpose() << ' ' << v.rgb.transpose() << std::endl;
	}
	file.close();
}



static void export_depth_buffer(const std::string& filename, const std::vector<double>& depth_buffer)
{
	std::ofstream file;
	file.open(filename);
	int i = 0;
	for (int y = 0; y < window_height; ++y)
	{
		for (int x = 0; x < window_width; ++x)
		{
			file << std::fixed << "v " << x << ' ' << y << ' ' << depth_buffer.at(i) << std::endl;
			++i;
		}
	}
	file.close();
}


void create_depth_buffer(std::vector<double>& depth_buffer, const std::vector<Eigen::Vector3d>& points3D, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view, std::vector<Eigen::Vector3d>& depth_buffer_3d)
{
	// Creating depth buffer
	depth_buffer.clear();
	depth_buffer.resize(int(window_width * window_height), -1.0);


	for (const Eigen::Vector3d p3d : points3D)
	{
		Eigen::Vector4d v = view * p3d.homogeneous();
		v /= v.w();

		const Eigen::Vector4d clip = proj * view * p3d.homogeneous();
		const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();
		if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
			continue;

		Eigen::Vector3d pixel;
		pixel.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
		pixel.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
		pixel.z() = (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

		if (pixel.x() > 0 && pixel.x() < window_width &&
			pixel.y() > 0 && pixel.y() < window_height)
		{
			const int depth_index = (int)((int)pixel.y() * (int)window_width + (int)pixel.x());
			depth_buffer.at(depth_index) = v.z();
		}
	}
}

void create_plane(std::vector<Eigen::Vector3d>& points3D, float width, float height, float cell_size)
{
	points3D.clear();
	for (float y = -height * 0.5f; y < height * 0.5f; y += cell_size)
	{
		for (float x = -width * 0.5f; x < width * 0.5f; x += cell_size)
		{
			points3D.push_back(Eigen::Vector3d(x, y, 0));
		}
	}

}


// Usage: ./Volumetricd.exe ../../data/plane.obj 256 4
int main(int argc, char **argv)
{
	Timer timer;
	const std::string filepath = argv[1];
	int vol_size = atoi(argv[2]);
	int vx_size = atoi(argv[3]);
	Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());

	// Projection Matrix
	Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);

	// Modelview Matrix
	Eigen::Affine3d affine = Eigen::Affine3d::Identity();
	affine.translate(Eigen::Vector3d(0, 0, -192));
	T.first = affine.matrix();
	affine.rotate(Eigen::AngleAxisd(DegToRad(45.0), Eigen::Vector3d::UnitY()));		// 45º
	T.second = affine.matrix();

	std::vector<Eigen::Vector3d> points3D, points3DDepth;
	timer.start();
	//create_plane(points3D, 128, 128, 0.5);
	//export_obj("../../data/plane_128_128_05.obj", points3D);
	import_obj(filepath, points3D);
	timer.print_interval("Import .obj         : ");


	//timer.start();
	//export_obj("../../data/mvp.obj", points3D, K, T.first);
	//timer.print_interval("Export mvp          : ");
	//return 0;

	// Creating depth buffer
	std::vector<double> depth_buffer;
	timer.start();
	create_depth_buffer(depth_buffer, points3D, K, T.first, points3DDepth);
	timer.print_interval("Create depth buffer : ");

	//timer.start();
	//export_depth_buffer("../../data/depth_buffer.obj", depth_buffer);
	//export_obj("../../data/depth_buffer_3d.obj", points3DDepth);
	//timer.print_interval("Export depth buffer : ");



	// Creating volume

	Grid grid(volume_size, voxel_size);
	//export_volume("../../data/grid_volume_clean.obj", grid.data, T.first * grid.transformation);

	


	Eigen::Vector4d ti = T.first.col(3);
	double half_voxel = voxel_size.x() * 0.5;
	std::vector<Eigen::Vector3d> pointsAll;


	
	timer.start();

	// Sweeping volume
	const std::size_t slice_size = (voxel_count.x() + 1) * (voxel_count.y() + 1);
	for (auto it_volume = grid.data.begin(); it_volume != grid.data.end(); it_volume += slice_size)
	{
		auto z_slice_begin = it_volume;
		auto z_slice_end = it_volume + slice_size;

		for (auto it = z_slice_begin; it != z_slice_end; ++it)
		{
			// to world space
			Eigen::Vector4d vg = grid.transformation * it->point.homogeneous();

			
			// to camera space
			Eigen::Vector4d v = T.first * vg;	// se for inversa não fica no clip space							
			v /= v.w();

			pointsAll.push_back(v.head<3>());

			// to ndc space
			const Eigen::Vector4d clip = K * v;
			const Eigen::Vector3d ndc = (clip / clip.w()).head<3>();
			
			// check if it is out of ndc space
			if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
				continue;

			// to screen space
			Eigen::Vector3d pixel;
			pixel.x() = window_width / 2.0 * ndc.x() + window_width / 2.0;
			pixel.y() = window_height / 2.0 * ndc.y() + window_height / 2.0;
			//pixel.z() = (far_plane - near_plane) / 2.0 * ndc.z() + (far_plane + near_plane) / 2.0;


			// get depth buffer value at pixel where the current vertex has been projected
			const int depth_pixel_index = (int)(pixel.y() * window_width + pixel.x());
			if (depth_pixel_index > depth_buffer.size())
			{
				std::cout << "Depth pixel out of range: " << depth_pixel_index << " > " << pixel.transpose() << std::endl;
				continue;
			}
						
			const double Dp = std::abs(depth_buffer.at(depth_pixel_index));
			
			//double dist_v_cam = v.z();
			//it->tsdf = dist_v_cam - Dp;
			//if (it->tsdf > -half_voxel && it->tsdf < half_voxel)
			//	it->rgb = Eigen::Vector3d(0, 255, 0);
			//else if (it->tsdf > 0)
			//	it->rgb = Eigen::Vector3d(0, 128, 255);
			//else
			//	it->rgb = Eigen::Vector3d(255, 0, 0);


			double distance_vertex_camera = (ti - vg).norm();
			it->tsdf = distance_vertex_camera - Dp;

			if (it->tsdf > voxel_size.x() && it->tsdf < voxel_size.x())
				it->rgb = Eigen::Vector3d(0, 255, 0);
			else if (it->tsdf > 0)
				it->rgb = Eigen::Vector3d(0, 128, 255);
			else if (it->tsdf < 0)
				it->rgb = Eigen::Vector3d(255, 0, 0);
			else
				it->rgb = Eigen::Vector3d(128, 255, 0);

		}

	}

	timer.print_interval("Filling volume      : ");

#if 0	

	//timer.start();
	//export_volume("../../data/volume_tsdf_wip.obj", grid.data, T.first * grid.transformation);
	//timer.print_interval("Exporting volume    : ");

	timer.start();

	Eigen::Vector3d origin = T.first.col(3).head<3>();
	Eigen::Vector3d window_coord_norm;

	std::vector<Eigen::Vector3d> output_cloud;

	// Sweep the volume looking for the zero crossing
	for (int y = 0; y < window_height * 0.1; ++y)
	{
		std::cout << "Ray casting to image... " << (double)y / window_height * 100 << "%" << std::endl;

		for (int x = 0; x < window_width * 0.1; ++x)
		{
			window_coord_norm.x() = ((double)x / window_width * 2.0) - 1.0;
			window_coord_norm.y() = ((double)y / window_height * 2.0) - 1.0;
			window_coord_norm.z() = origin.z() + near_plane;
			Eigen::Vector3d direction = (window_coord_norm - origin).normalized();

			std::vector<int> intersections = Grid::find_intersections(grid.data, volume_size, voxel_size, grid.transformation, origin, direction, near_plane, far_plane);
			Grid::sort_intersections(intersections, grid.data, origin);

			for (int i = 1; i < intersections.size(); ++i)
			{
				const Voxeld& prev = grid.data.at(i - 1);
				const Voxeld& curr = grid.data.at(i);

				const bool& same_sign = ((prev.tsdf < 0) == (curr.tsdf < 0));
				
				if (!same_sign)		// it is a zero-crossing
				{
					output_cloud.push_back(curr.point);
				}
			}
		}
	}

	timer.print_interval("Raycasting volume   : ");


	export_obj("../../data/output_cloud.obj", output_cloud);
#endif

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