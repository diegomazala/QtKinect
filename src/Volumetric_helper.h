
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
const double MaxWeight = 10.0;

const double fov_y = 70.0f;
const double window_width = 512.0f;
const double window_height = 424.0f;
const double near_plane = 0.1f; // 0.1f;
const double far_plane = 512.0f; // 10240.0f;
const double aspect_ratio = window_width / window_height;
//Eigen::Matrix4d	K(Eigen::Matrix4d::Zero());
//std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Zero(), Eigen::Matrix4d::Zero());

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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix3d& R, Eigen::Vector3d& t);
///
/// @brief	Compute the rotation and translation that transform a source point set to a target point set
///
/// @author	Diego
/// @date	07/10/2015
///
/// @param	src		   		The source point set.
/// @param	dst		   		The target point set.
/// @param [in,out]	pts_dst	The rotation matrix.
/// @param [in,out]	pts_dst	The translation vector.
/// @return	True if found the transformation, false otherwise.
////////////////////////////////////////////////////////////////////////////////////////////////////
static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.size() == dst.size());
	int pairSize = (int)src.size();
	Eigen::Vector3d center_src(0, 0, 0), center_dst(0, 0, 0);

	// 
	// Compute centroid
	//
	for (int i = 0; i<pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (double)pairSize;
	center_dst /= (double)pairSize;


	Eigen::MatrixXd S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i<pairSize; ++i)
	{
		S.row(i) = src[i] - center_src;
		D.row(i) = dst[i] - center_dst;
	}
	Eigen::MatrixXd Dt = D.transpose();
	Eigen::Matrix3d H = Dt * S;
	Eigen::Matrix3d W, U, V;

	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
		std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix3d Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst - R * center_src;
	
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	static bool ComputeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix3f& R, Eigen::Vector3f& t);
///
/// @brief	Compute the rotation and translation that transform a source point set to a target point set
///
/// @author	Diego
/// @date	07/10/2015
///
/// @param	src		   		The source point set.
/// @param	dst		   		The target point set.
/// @param [in,out]	pts_dst	The rotation matrix.
/// @param [in,out]	pts_dst	The translation vector.
/// @return	True if found the transformation, false otherwise.
////////////////////////////////////////////////////////////////////////////////////////////////////
static bool ComputeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.size() == dst.size());
	int pairSize = (int)src.size();
	Eigen::Vector4f center_src(0, 0, 0, 1), center_dst(0, 0, 0, 1);

	// 
	// Compute centroid
	//
	for (int i = 0; i<pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (float)pairSize;
	center_dst /= (float)pairSize;


	Eigen::MatrixXf S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i<pairSize; ++i)
	{
		const Eigen::Vector4f src4f = src[i] - center_src;
		const Eigen::Vector4f dst4f = dst[i] - center_dst;

		S.row(i) = (src4f / src4f.w()).head<3>();
		D.row(i) = (dst4f / dst4f.w()).head<3>();
	}
	Eigen::MatrixXf Dt = D.transpose();
	Eigen::Matrix3f H = Dt * S;
	Eigen::Matrix3f W, U, V;

	
	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::MatrixXf> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
		std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	Eigen::Vector3f center_src_3f = (center_src / center_src.w()).head<3>();
	Eigen::Vector3f center_dst_3f = (center_dst / center_dst.w()).head<3>();

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix3f Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst_3f - R * center_src_3f;

	return true;
}

static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix4d& mat)
{
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	if (ComputeRigidTransform(src, dst, R, t))
	{
		mat.block(0, 0, 3, 3) = R;
		mat.row(3).setZero();
		mat.col(3) = t.homogeneous();
		return true;
	}
	return false;
}

static bool ComputeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix4f& mat)
{
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	if (ComputeRigidTransform(src, dst, R, t))
	{
		mat.block(0, 0, 3, 3) = R;
		mat.row(3).setZero();
		mat.col(3) = t.homogeneous();
		return true;
	}
	return false;
}

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


static bool import_obj(const std::string& filename, std::vector<Eigen::Vector4f>& points3D, int max_point_count = INT_MAX)
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

			Eigen::Vector4f p(x, y, z, 1.0f);
			points3D.push_back(p);
		}

		if (i++ > max_point_count)
			break;
	}

	inFile.close();
	return true;
}



template<typename Type, int Rows>
static void export_obj_with_normals(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices, const std::vector<Eigen::Matrix<Type, Rows, 1>>& normals)
{
	std::ofstream file;
	file.open(filename);
	for (const auto v : vertices)
		file << std::fixed << "v " << v.transpose() << std::endl;
	for (const auto n : normals)
		file << std::fixed << "vn " << n.transpose() << std::endl;
	file.close();
}

template<typename Type, int Rows>
static void export_obj_with_colors(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices, const std::vector<Eigen::Vector3f>& rgb)
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < vertices.size(); ++i)
	{
		const auto& v = vertices[i];
		const auto& c = rgb[i];
		file << std::fixed << "v " << v.transpose() << '\t' << c.transpose() << std::endl;
	}
	file.close();
}


template<typename Type, int Rows>
static void export_obj(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices)
{
	std::ofstream file;
	file.open(filename);
	for (const auto v : vertices)
	{
		file << std::fixed << "v " << v.transpose() << std::endl;
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
		//rotation.rotate(Eigen::AngleAxisd(DegToRad(v.tsdf * 180.0), Eigen::Vector3d::UnitZ()));		// 90�
		//rgb = rotation.matrix() * (-Eigen::Vector4d::UnitX());
		//if (v.tsdf > -0.1 && v.tsdf < 0.1)
		//int r = v.rgb.x();
		//int b = v.rgb.z();
		//if (r != 255 && b != 255)

		Eigen::Vector3i rgb(255, 255, 255);
		if (v.tsdf > 0.1)
		{
			rgb = Eigen::Vector3i(0, 255, 0);
			file << std::fixed << "v " << (transformation * v.point.homogeneous()).head<3>().transpose() << ' ' << rgb.transpose() << std::endl;
		}
		else if (v.tsdf < -0.1)
		{
			rgb = Eigen::Vector3i(255, 0, 0);
			file << std::fixed << "v " << (transformation * v.point.homogeneous()).head<3>().transpose() << ' ' << rgb.transpose() << std::endl;
		}
		
		
		
	}
	file.close();
}

static void export_volume(const std::string& filename, const std::vector<Eigen::Vector4f>& points, const std::vector<Eigen::Vector2f>& params, const Eigen::Matrix4f& transformation = Eigen::Matrix4f::Identity())
{
	Eigen::Affine3f rotation;
	Eigen::Vector4f rgb;
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < points.size(); ++i)
	{
		const Eigen::Vector4f& v = points[i];
		const float tsdf = params[i][0];
		

		Eigen::Vector3i rgb(255, 255, 255);
		if (tsdf > 0.1f)
		{
			rgb = Eigen::Vector3i(0, 255, 0);
			file << std::fixed << "v " << (transformation * v).head<3>().transpose() << ' ' << rgb.transpose() << std::endl;
		}
		else if (tsdf < -0.1f)
		{
			rgb = Eigen::Vector3i(255, 0, 0);
			file << std::fixed << "v " << (transformation * v).head<3>().transpose() << ' ' << rgb.transpose() << std::endl;
		}
	}
	file.close();
}


template <typename Type>
static void export_depth_buffer(const std::string& filename, const std::vector<Type>& depth_buffer, Eigen::Vector3i rgb = Eigen::Vector3i::Zero())
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

//static void export_image_from_depth_buffer(const std::string& filename, const std::vector<double>& depth_buffer)
//{
//	QImage depth(window_width, window_height, QImage::Format_RGB888);
//	depth.fill(Qt::white);
//
//	int x = 0;
//	int y = 0;
//	int i = 0;
//	for (double d : depth_buffer)
//	{
//		int gray = 255 - (d / far_plane * 255.0);
//		x = i % (int)window_width;
//		y = i / (int)window_width;
//		depth.setPixel(QPoint(x, window_height - y - 1), qRgb(gray, gray, gray));
//		++i;
//	}
//
//	depth.save(filename.c_str());
//}

static void create_plane(std::vector<Eigen::Vector3d>& points3D, float width, float height, float z, float cell_size)
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

static void build_cloud_array_points_of_planes(int count, float rotation_interval, float width, float height, float z, float cell_size)
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


static void create_depth_buffer(std::vector<float>& depth_buffer, const std::vector<Eigen::Vector4f>& points3D, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, float far_plane)
{
	// Creating depth buffer
	depth_buffer.clear();
	depth_buffer.resize(int(window_width * window_height), far_plane);

	for (const Eigen::Vector4f p3d : points3D)
	{
		Eigen::Vector4f v = view * p3d;
		v /= v.w();

		const Eigen::Vector4f clip = proj * view * p3d;
		const Eigen::Vector3f ndc = (clip / clip.w()).head<3>();
		if (ndc.x() < -1 || ndc.x() > 1 || ndc.y() < -1 || ndc.y() > 1 || ndc.z() < -1 || ndc.z() > 1)
			continue;

		Eigen::Vector3f pixel;
		pixel.x() = window_width / 2.0f * ndc.x() + window_width / 2.0f;
		pixel.y() = window_height / 2.0f * ndc.y() + window_height / 2.0f;
		//pixel.z() = (far_plane - near_plane) / 2.0f * ndc.z() + (far_plane + near_plane) / 2.0f;

		const int depth_index = (int)pixel.y() * (int)window_width + (int)pixel.x();
		if (depth_index > 0 && depth_index < depth_buffer.size())
		{
			const double& curr_depth = std::abs(depth_buffer.at(depth_index));
			const double& new_depth = std::abs(v.z());
			if (new_depth < curr_depth)
				depth_buffer.at(depth_index) = new_depth;
		}
	}
}


static void create_depth_buffer(std::vector<double>& depth_buffer, const std::vector<Eigen::Vector3d>& points3D, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view, double far_plane)
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

		const int depth_index = (int)pixel.y() * (int)window_width + (int)pixel.x();
		if (depth_index > 0 && depth_index < depth_buffer.size())
		{
			const double& curr_depth = std::abs(depth_buffer.at(depth_index));
			const double& new_depth = std::abs(v.z());
			if (new_depth < curr_depth)
				depth_buffer.at(depth_index) = new_depth;
		}
	}
}

static double compute_tsdf(const Eigen::Vector3d& pt, std::vector<double>& depth_buffer, const Eigen::Matrix4d& view)
{
	Eigen::Vector4d vg = pt.homogeneous();
	Eigen::Vector4d v = view * vg;	// se for inversa n�o fica no clip space		
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


static void update_volume(Grid& grid, std::vector<double>& depth_buffer, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view)
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
			Eigen::Vector4d v = view.inverse() * vg;	
			v /= v.w();

			// to screen space
			const Eigen::Vector3i pixel = vertex_to_window_coord(v, fov_y, aspect_ratio, near_plane, far_plane, (int)window_width, (int)window_height).cast<int>();

			// get depth buffer value at pixel where the current vertex has been projected
			const int depth_pixel_index = pixel.y() * int(window_width) + pixel.x();
			if (depth_pixel_index < 0 || depth_pixel_index > depth_buffer.size() - 1)
			{
				continue;
			}

			const double Dp = std::abs(depth_buffer.at(depth_pixel_index));

			double distance_vertex_camera = (ti - vg).norm();

			const double sdf = Dp - distance_vertex_camera;
			
			const double half_voxel_size = grid.voxel_size.x();// *0.5;
			if (std::fabs(sdf) > half_voxel_size)
				continue;


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

#if 1	// Izadi
			const double weight = std::fmin(MaxWeight, prev_weight + 1);
			const double tsdf_avg = (prev_tsdf * prev_weight + tsdf * weight) / (prev_weight + weight);
#else	// Open Fusion
			const double weight = std::fmin(MaxWeight, prev_weight + 1);
			const double tsdf_avg = (prev_tsdf * prev_weight + tsdf * 1) / (prev_weight + 1);
#endif

			it->tsdf = tsdf_avg;
			it->weight = weight;
			

			it->sdf = sdf;
			it->tsdf_raw = tsdf;
		}
	}
}



static void update_volume(
	std::vector<Eigen::Vector4f>& points, 
	std::vector<Eigen::Vector2f>& params, 
	std::vector<float>& depth_buffer, 
	const Eigen::Matrix4f& proj, 
	const Eigen::Matrix4f& view,
	int volume_size,
	int voxel_size)
{
	const Eigen::Vector4f& ti = view.col(3);

	Eigen::Vector3i voxel_count(volume_size / voxel_size, volume_size / voxel_size, volume_size / voxel_size);

	//
	// Sweeping volume
	//
	const std::size_t slice_size = (voxel_count.x() + 1) * (voxel_count.y() + 1);
	int index = 0;
	for (auto it_volume = points.begin(); it_volume != points.end(); it_volume += slice_size)
	{
		auto z_slice_begin = it_volume;
		auto z_slice_end = it_volume + slice_size;

		for (auto it = z_slice_begin; it != z_slice_end; ++it)
		{
			// to world space
			Eigen::Vector4f vg = (*it);

			// to camera space
			Eigen::Vector4f v = view.inverse() * vg;
			v /= v.w();

			// to screen space
			const Eigen::Vector3i pixel = vertex_to_window_coord(v, (float)fov_y, (float)aspect_ratio, (float)near_plane, (float)far_plane, (int)window_width, (int)window_height).cast<int>();

			// get depth buffer value at pixel where the current vertex has been projected
			const int depth_pixel_index = pixel.y() * int(window_width) + pixel.x();
			if (depth_pixel_index < 0 || depth_pixel_index > depth_buffer.size() - 1)
			{
				continue;
			}

			const float Dp = std::abs(depth_buffer.at(depth_pixel_index));

			float distance_vertex_camera = (ti - vg).norm();

			const float sdf = Dp - distance_vertex_camera;

			const float half_voxel_size = voxel_size;// *0.5;
			if (std::fabs(sdf) > half_voxel_size)
				continue;

			
			const float prev_tsdf = params[index][0];
			const float prev_weight = params[index][1];
			
			double tsdf = sdf;

			if (sdf > 0)
			{
				tsdf = std::fmin(1.0f, sdf / MaxTruncation);
			}
			else
			{
				tsdf = std::fmax(-1.0f, sdf / MinTruncation);
			}

#if 1	// Izadi
			const double weight = std::fmin(MaxWeight, prev_weight + 1);
			const double tsdf_avg = (prev_tsdf * prev_weight + tsdf * weight) / (prev_weight + weight);
#else	// Open Fusion
			const double weight = std::fmin(MaxWeight, prev_weight + 1);
			const double tsdf_avg = (prev_tsdf * prev_weight + tsdf * 1) / (prev_weight + 1);
#endif

			
			params[index][0] = tsdf_avg;
			params[index][1] = weight;

			++index;
		}
	}
}