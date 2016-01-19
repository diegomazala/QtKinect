
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

const double MinTruncation = 0.5;
const double MaxTruncation =  1.1;
const double MaxWeight = 5.0;

const double fov_y = 70.0f;
const double window_width = 512.0f;
const double window_height = 424.0f;
const double near_plane = 0.1f; // 0.1f;
const double far_plane = 512.0f; // 10240.0f;
const double aspect_ratio = window_width / window_height;
Eigen::Matrix4d	K(Eigen::Matrix4d::Zero());
std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Zero(), Eigen::Matrix4d::Zero());

typedef std::vector<Eigen::Vector3d> PointCloud;
std::vector<PointCloud> cloud_array_points;
std::vector<Eigen::Matrix4d> cloud_array_matrix;


#if 0
void raycast_volume()
{
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
}
#endif


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

		Eigen::Vector3i rgb(255, 255, 255);
		if (v.tsdf > 0.1)
			rgb = Eigen::Vector3i(0, 255, 0);
		else if (v.tsdf < -0.1)
			rgb = Eigen::Vector3i(255, 0, 0);
		
		file << std::fixed << "v " << (transformation * v.point.homogeneous()).head<3>().transpose() << ' ' << rgb.transpose() << std::endl;
	}
	file.close();
}



static void export_depth_buffer(const std::string& filename, const std::vector<double>& depth_buffer, Eigen::Vector3i rgb = Eigen::Vector3i::Zero())
{
	std::ofstream file;
	file.open(filename);
	int i = 0;
	for (int y = 0; y < window_height; ++y)
	{
		for (int x = 0; x < window_width; ++x)
		{
			file << std::fixed << "v " << x << ' ' << y << ' ' << depth_buffer.at(i) << ' ' << rgb.transpose() << std::endl;
			++i;
		}
	}
	file.close();
}

static void export_image_from_depth_buffer(const std::string& filename, const std::vector<double>& depth_buffer)
{
	QImage depth(window_width, window_height, QImage::Format_RGB888);
	depth.fill(Qt::white);

	int x = 0;
	int y = 0;
	int i = 0;
	for (double d : depth_buffer)
	{
		int gray = 255 - (d / far_plane * 255.0);
		x = i % (int)window_width;
		y = i / (int)window_width;
		depth.setPixel(QPoint(x, window_height - y - 1), qRgb(gray, gray, gray));
		++i;
	}

	depth.save(filename.c_str());
}

void create_plane(std::vector<Eigen::Vector3d>& points3D, float width, float height, float z, float cell_size)
{
	points3D.clear();
	for (float y = -height * 0.5f; y <= height * 0.5f; y += cell_size)
	{
		for (float x = -width * 0.5f; x <= width * 0.5f; x += cell_size)
		{
			points3D.push_back(Eigen::Vector3d(x, y, z));
		}
	}

}

void build_cloud_array_points_of_planes(int count, float rotation_interval, float width, float height, float z, float cell_size)
{
	std::vector<Eigen::Vector3d> points3D;
	for (float y = -height * 0.5f; y <= height * 0.5f; y += cell_size)
	{
		for (float x = -width * 0.5f; x <= width * 0.5f; x += cell_size)
		{
			points3D.push_back(Eigen::Vector3d(x, y, z));
		}
	}

	// inserting first cloud
	cloud_array_points.push_back(points3D);
	cloud_array_matrix.push_back(Eigen::Matrix4d::Identity());

	for (int i = 1; i < count; ++i)
	{
		Eigen::Affine3d affine = Eigen::Affine3d::Identity();
		affine.translate(Eigen::Vector3d(0, 0, z));
		affine.rotate(Eigen::AngleAxisd(DegToRad(i * rotation_interval), Eigen::Vector3d::UnitY()));
		affine.translate(Eigen::Vector3d(0, 0, -z));

		std::vector<Eigen::Vector3d> points3DRot;

		for (const Eigen::Vector3d p3d : points3D)
		{
			Eigen::Vector4d rot = affine.matrix() * p3d.homogeneous();
			rot /= rot.w();

			points3DRot.push_back(rot.head<3>());
		}

		// inserting clouds rotated
		cloud_array_points.push_back(points3DRot);
		cloud_array_matrix.push_back(affine.matrix());
	}

	//int cc = 0;
	//for (auto c : cloud_array_points)
	//{
	//	std::stringstream ss;
	//	ss << "../../data/cloud_10_0" << cc << ".obj";
	//	export_obj(ss.str(), c);
	//	cc++;
	//}
}

void create_depth_buffer(std::vector<double>& depth_buffer, const std::vector<Eigen::Vector3d>& points3D, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view, double far_plane)
{
	// Creating depth buffer
	depth_buffer.clear();
	depth_buffer.resize(int(window_width * window_height), far_plane);


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
		//pixel.z() = (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

#if 0
		if (pixel.x() > 0 && pixel.x() < window_width &&
			pixel.y() > 0 && pixel.y() < window_height)
		{
			const int depth_index = (int)pixel.y() * (int)window_width + (int)pixel.x();
			const double& curr_depth = std::abs(depth_buffer.at(depth_index));
			const double& new_depth = std::abs(v.z());
			if (new_depth < curr_depth) 
				depth_buffer.at(depth_index) = new_depth;
		}
#else
		const int depth_index = (int)pixel.y() * (int)window_width + (int)pixel.x();
		if (depth_index > 0 && depth_index < depth_buffer.size())
		{
			const double& curr_depth = std::abs(depth_buffer.at(depth_index));
			const double& new_depth = std::abs(v.z());
			if (new_depth < curr_depth)
				depth_buffer.at(depth_index) = new_depth;
		}

#endif
	}
}

double compute_tsdf(const Eigen::Vector3d& pt, std::vector<double>& depth_buffer, const Eigen::Matrix4d& view)
{
	Eigen::Vector4d vg = pt.homogeneous();
	Eigen::Vector4d v = view * vg;	// se for inversa não fica no clip space		
	v /= v.w();

	// to screen space
	const Eigen::Vector3i pixel = vertex_to_window_coord(v, fov_y, aspect_ratio, near_plane, far_plane, (int)window_width, (int)window_height).cast<int>();
	

	// get depth buffer value at pixel where the current vertex has been projected
	const int depth_pixel_index = pixel.y() * int(window_width) + pixel.x();
		

	if (depth_pixel_index < 0 || depth_pixel_index > depth_buffer.size())
	{
		//std::cout << "Depth pixel out of range: " << depth_pixel_index << " > " << pixel.transpose() << std::endl;
		return -far_plane;
	}

	const double Dp = depth_buffer.at(depth_pixel_index);
	double distance_vertex_camera = std::abs(v.z());

	std::cout << std::endl << pixel.x() << ' ' << pixel.y() << " ---> " << depth_pixel_index << " : " << Dp << std::endl;



	int xx = depth_pixel_index % int(window_width);
	int yy = depth_pixel_index / int(window_width);
	std::cout << xx << ' ' << yy << " : " << yy * window_width + xx << std::endl;





	std::cout << std::endl << distance_vertex_camera << " - " << Dp << " = " << distance_vertex_camera - Dp << std::endl;
	
	return distance_vertex_camera - Dp;
}


void update_volume(Grid& grid, std::vector<Eigen::Vector3d>& points3DGrid, std::vector<double>& depth_buffer, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view)
{
	const Eigen::Vector4d& ti = view.col(3);

	//
	// Sweeping volume
	//
	const std::size_t slice_size = (grid.voxel_count.x() + 1) * (grid.voxel_count.y() + 1);
	for (auto it_volume = grid.data.begin(); it_volume != grid.data.end(); it_volume += slice_size)
	{
		auto z_slice_begin = it_volume;
		auto z_slice_end = it_volume + slice_size;

		for (auto it = z_slice_begin; it != z_slice_end; ++it)
		{
			// to world space
			Eigen::Vector4d vg = it->point.homogeneous();

			// to camera space
			Eigen::Vector4d v = view.inverse() * vg;	// se for inversa não fica no clip space		
			v /= v.w();

			// to screen space
			const Eigen::Vector3i pixel = vertex_to_window_coord(v, fov_y, aspect_ratio, near_plane, far_plane, (int)window_width, (int)window_height).cast<int>();

			// get depth buffer value at pixel where the current vertex has been projected
			const int depth_pixel_index = pixel.y() * int(window_width) + pixel.x();
			if (depth_pixel_index < 0 || depth_pixel_index > depth_buffer.size())
			{
				// default values for voxels out of frustum
				it->weight = MaxWeight;
				it->tsdf = MaxTruncation;
				continue;
			}

			const double Dp = std::abs(depth_buffer.at(depth_pixel_index));

			double distance_vertex_camera = (ti - vg).norm();

			const double sdf = Dp - distance_vertex_camera;
			
			const double prev_weight = it->weight;
			const double prev_tsdf = it->tsdf;
			double tsdf = sdf;

			if (sdf > 0)
			{
				tsdf = std::fmin(1.0, sdf / MaxTruncation);
			}
			else
			{
				tsdf = std::fmax(-1.0, sdf / MinTruncation);
			}

			const double weight = std::fmin(MaxWeight, prev_weight + 1);
			const double tsdf_avg = (prev_tsdf * prev_weight + tsdf * weight) / (prev_weight + weight);

			it->weight = weight;
			it->tsdf = tsdf_avg;

			//std::cout << it->point.transpose() << " : " << it->tsdf << " : " << tsdf << " : " << it->weight << std::endl;

			points3DGrid.push_back(v.head<3>());
		}

	}
}


// Usage: ./Volumetricd.exe ../../data/plane.obj 256 4
int main(int argc, char **argv)
{
	if (argc < 6)
	{
		std::cerr << "Missing parameters. Abort." 
			<< std::endl
			<< "Usage:  ./Volumetricd.exe ../../data/monkey.obj 256 5 5 10"
			<< std::endl;
		return EXIT_FAILURE;
	}
	Timer timer;
	const std::string filepath = argv[1];
	const int vol_size = atoi(argv[2]);
	const int vx_size = atoi(argv[3]);
	const int cloud_count = atoi(argv[4]);
	const int rot_interval = atoi(argv[5]);
	Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());


	//
	// Projection Matrix
	//
	Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);
	//
	// Modelview Matrix
	//
	Eigen::Affine3d affine = Eigen::Affine3d::Identity();
	//affine.translate(Eigen::Vector3d(0, 0, -192));
	T.first = affine.matrix();
	affine.rotate(Eigen::AngleAxisd(DegToRad(30.0), Eigen::Vector3d::UnitY()));
	T.second = affine.matrix();
	


	//
	// Import .obj
	//
	std::vector<Eigen::Vector3d> points3DGrid;
	//timer.start();
	//create_plane(points3D, 128, 128, 0.5);
	//export_obj("../../data/plane_128_128_05.obj", points3D);
	//import_obj(filepath, points3D);
	//timer.print_interval("Import .obj         : ");
	//create_plane(points3D, 128, 128, -256, 0.5);
	//export_obj("../../data/plane_128_128_-256_01.obj", points3D);
	//
	timer.start();
	build_cloud_array_points_of_planes(cloud_count, rot_interval, 128, 128, -256, 1);
	timer.print_interval("Create cloud array  : ");


	std::vector<Eigen::Vector3d>& points3D = cloud_array_points.at(0);

	


	//
	// Creating depth buffer
	//
	std::vector<double> depth_buffer;
	//timer.start();
	//create_depth_buffer(depth_buffer, points3D, K, T.first, far_plane);
	//timer.print_interval("Create depth buffer : ");


	//Eigen::Vector3d pt(atof(argv[4]), atof(argv[5]), atof(argv[6]));
	//std::cout << std::fixed
	//	<< "tsdf (" << pt.transpose() << ") = "
	//	<< compute_tsdf(pt, depth_buffer, T.first)
	//	<< std::endl;
	//return 0;


	//timer.start();
	//export_depth_buffer("../../data/depth_buffer.obj", depth_buffer);
	//export_image_from_depth_buffer("../../data/depth_buffer.png", depth_buffer);
	////export_obj("../../data/depth_buffer_3d.obj", points3DDepth);
	//timer.print_interval("Export depth buffer : ");
	//return 0;


	//
	// Creating volume
	//
	Eigen::Affine3d grid_affine = Eigen::Affine3d::Identity();
	grid_affine.translate(Eigen::Vector3d(0, 0, -256));
	grid_affine.scale(Eigen::Vector3d(1, 1, -1));	// z is negative inside of screen
	
	
	Grid grid(volume_size, voxel_size, grid_affine.matrix());
	//Grid grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());
	//export_volume("../../data/grid_volume_clean.obj", grid.data);
	//return 0;

	timer.start();
	{
#if 1
		int i = atoi(argv[5]);
		for (int j = 0; j < cloud_count; ++j)
		{
			std::cout << std::endl << i << std::endl;
			Eigen::Affine3d affine = Eigen::Affine3d::Identity();
			affine.rotate(Eigen::AngleAxisd(DegToRad(i), Eigen::Vector3d::UnitY()));
			T.first = affine.matrix();

			timer.start();
			create_depth_buffer(depth_buffer, points3D, K, T.first, far_plane);

			//std::stringstream ss;
			//ss << "../../data/depth_buffer_00" << i << ".png";
			//export_image_from_depth_buffer(ss.str(), depth_buffer);
			timer.print_interval("Create depth buffer : ");

			timer.start();
			update_volume(grid, points3DGrid, depth_buffer, K, T.first.inverse());
			timer.print_interval("Update volume       : ");
		}
#else

		for (int i = 0; i < cloud_count; ++i)
		{
			std::cout << std::endl << i << std::endl;

			const Eigen::Matrix4d& mat = cloud_array_matrix[i];

			std::cout << std::endl << mat << std::endl << std::endl;

			timer.start();
			create_depth_buffer(depth_buffer, points3D, K, mat, far_plane);

			//std::stringstream ss;
			//ss << "../../data/depth_buffer_00" << i << ".png";
			//export_image_from_depth_buffer(ss.str(), depth_buffer);
			timer.print_interval("Create depth buffer : ");

			timer.start();
			update_volume(grid, points3DGrid, depth_buffer, K, mat.inverse());
			timer.print_interval("Update volume       : ");
		}

#endif
		
	}
	timer.print_interval("Filling volume      : ");

	//for (auto v : grid.data)
	//{
	//	std::cout << v.point.transpose() << " : " << v.tsdf << " : " << v.weight << std::endl;
	//}


	export_volume("../../data/grid_volume.obj", grid.data);
	return 0;



	
	std::vector<double> grid_depth_buffer;
	timer.start();
	{
		create_depth_buffer(grid_depth_buffer, points3DGrid, K, T.first, far_plane);
	}
	timer.print_interval("Grid depth buffer   : ");
	//export_depth_buffer("../../data/grid_depth_buffer.obj", grid_depth_buffer, Eigen::Vector3i(255, 0, 0));
	//return 0;


	timer.start();
	Eigen::Vector3i rgb(0, 255, 0);
	std::ofstream file;
	file.open("../../data/diff_depth_buffer.obj");
	int i = 0;
	for (int y = 0; y < window_height; ++y)
	{
		for (int x = 0; x < window_width; ++x)
		{
			const double diff = depth_buffer.at(i) - grid_depth_buffer.at(i);
#if 1
			Eigen::Vector3i rgb(255, 255, 255);
			if (diff > 1)
				rgb = Eigen::Vector3i(0, 255, 0);
			else if (diff < 1)
				rgb = Eigen::Vector3i(255, 0, 0);

			if (grid_depth_buffer.at(i) < far_plane)
				file << std::fixed << "v " << x << ' ' << y << ' ' << grid_depth_buffer.at(i) << ' ' << rgb.transpose() << std::endl;
#else
			if (diff < 1 && grid_depth_buffer.at(i) < far_plane)
				file << std::fixed << "v " << x << ' ' << y << ' ' << grid_depth_buffer.at(i) << ' ' << rgb.transpose() << std::endl;
#endif
			++i;
		}
	}
	file.close();
	timer.print_interval("Exporting Diff Buffer ");

	timer.start();
	//export_volume("../../data/volume_tsdf_wip.obj", grid.data, T.first * grid.transformation);
	//export_volume("../../data/volume_tsdf_wip.obj", grid.data, grid.transformation);
	//export_volume("../../data/volume_tsdf_wip.obj", grid.data);
	timer.print_interval("Exporting volume    : ");



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