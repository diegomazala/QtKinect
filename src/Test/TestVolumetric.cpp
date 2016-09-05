// Including SDKDDKVer.h defines the highest available Windows platform.
// If you wish to build your application for a previous Windows platform, include WinSDKVer.h and
// set the _WIN32_WINNT macro to the platform you wish to support before including SDKDDKVer.h.
#include <SDKDDKVer.h>

#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include <algorithm>
#include <sstream>
#include <fstream>
#include <QApplication>
#include <QTimer>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include "Timer.h"
#include "RayBox.h"
#include "Grid.h"
#include "Projection.h"
#include "RayIntersection.h"
#include "Eigen/Eigen"

#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.

const double MinTruncation = 0.5;
const double MaxTruncation = 1.1;
const double MaxWeight = 10.0;

const double fov_y = 70.0f;
const double window_width = 512.0f;
const double window_height = 424.0f;
const double near_plane = 0.1f; // 0.1f;
const double far_plane = 512.0f; // 10240.0f;
const double aspect_ratio = window_width / window_height;


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




////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix3d& R, Eigen::Vector3d& t);
///
/// \brief	Compute the rotation and translation that transform a source point set to a target point set
///
/// \author	Diego
/// \date	07/10/2015
///
/// \param	src		   		The source point set.
/// \param	dst		   		The target point set.
/// \param [in,out]	pts_dst	The rotation matrix.
/// \param [in,out]	pts_dst	The translation vector.
/// \return	True if found the transformation, false otherwise.
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
		for (int j = 0; j<3; ++j)
			S(i, j) = src[i][j] - center_src[j];
		for (int j = 0; j<3; ++j)
			D(i, j) = dst[i][j] - center_dst[j];
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


static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix4d& mat)
{
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	if (ComputeRigidTransform(src, dst, R, t))
	{
		mat.block(0, 0, 3, 3) = R;
		mat.col(3) = t.homogeneous();
		return true;
	}
	return false;
}


static void export_volume(const std::string& filename, const std::vector<Voxeld>& volume, const Eigen::Matrix4d& transformation = Eigen::Matrix4d::Identity())
{
	Eigen::Affine3d rotation;
	Eigen::Vector4d rgb;
	std::ofstream file;
	file.open(filename);
	for (const auto v : volume)
	{
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


static void update_volume(Grid<double>& grid, std::vector<double>& depth_buffer, const Eigen::Matrix4d& proj, const Eigen::Matrix4d& view)
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

			it->weight = weight;
			it->tsdf = tsdf_avg;

			it->sdf = sdf;
			it->tsdf_raw = tsdf;
		}
	}
}




namespace TestVolumetric
{		
	TEST_CLASS(UnitTestMonkey)
	{
	public:
		
		TEST_METHOD(Test1)
		{
			int vol_size = 256;
			int vx_size = 32;
			int iterations = 2;

			Timer timer;

			//
			// Importing monkey obj
			//
			timer.start();
			std::vector<Eigen::Vector3d> points3DOrig, pointsTmp;
			import_obj("../../data/monkey.obj", points3DOrig);
			timer.print_interval("Importing monkey    : ");
			std::cout << "Monkey point count  : " << points3DOrig.size() << std::endl;

			// 
			// Translating and rotating monkey point cloud 
			std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> cloud;
			//
			Eigen::Affine3d translate = Eigen::Affine3d::Identity();
			translate.translate(Eigen::Vector3d(0, 0, -256));
			//
			Eigen::Affine3d rotate = Eigen::Affine3d::Identity();
			rotate.rotate(Eigen::AngleAxisd(DegToRad(90.0), Eigen::Vector3d::UnitY()));
			//
			for (Eigen::Vector3d p3d : points3DOrig)
			{
				Eigen::Vector4d trans = translate.matrix() * p3d.homogeneous();
				trans /= trans.w();

				Eigen::Vector4d rot = translate.matrix() * rotate.matrix() * p3d.homogeneous();
				rot /= rot.w();

				cloud.first.push_back(trans.head<3>());
				cloud.second.push_back(rot.head<3>());
			}

			Eigen::Matrix4d icp_mat;
			timer.start();
			ComputeRigidTransform(cloud.first, cloud.second, icp_mat);
			timer.print_interval("Compute rigid transf: ");


			//
			// Projection Matrix
			//
			Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);
			//
			// Setup T matrix : transformation between clouds
			//
			std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T;
			T.first = Eigen::Matrix4d::Identity();
			T.second = icp_mat;

			std::cout << std::fixed << std::endl << "icp_mat m1" << std::endl << icp_mat << std::endl;

			//
			// Creating volume
			//
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
			//
			Eigen::Affine3d grid_affine = Eigen::Affine3d::Identity();
			grid_affine.translate(Eigen::Vector3d(0, 0, -256));
			grid_affine.scale(Eigen::Vector3d(1, 1, -1));	// z is negative inside of screen
			Grid<double> grid(volume_size, voxel_size, grid_affine.matrix());

			//
			// Creating depth buffer for input clouds
			//
			std::pair<std::vector<double>, std::vector<double>> depth_buffer;
			timer.start();
			create_depth_buffer(depth_buffer.first, cloud.first, K, Eigen::Matrix4d::Identity(), far_plane);
			create_depth_buffer(depth_buffer.second, cloud.second, K, Eigen::Matrix4d::Identity(), far_plane);
			timer.print_interval("Create depth buffer : ");

			timer.start();
			update_volume(grid, depth_buffer.first, K, T.first.inverse());
			update_volume(grid, depth_buffer.second, K, T.second.inverse());
			timer.print_interval("Update volume       : ");

			timer.start();
			export_volume("../../data/grid_volume_monkey_test.obj", grid.data);
			timer.print_interval("Export volume       : ");

			//Assert::AreEqual(EXIT_SUCCESS, app_exit, L"\n<TestKinectColor appplication did not finish properly>\n", LINE_INFO());
		}



		TEST_METHOD(Test2)
		{
			int vol_size = 256;
			int vx_size = 32;
			int iterations = 2;
			int rot_interval = 2;

			Timer timer;

			std::pair<std::vector<double>, std::vector<double>> depth_buffer;

			//
			// Projection and Modelview Matrices
			//
			Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);
			std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());


			//
			// Creating volume
			//
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
			//
			Eigen::Affine3d grid_affine = Eigen::Affine3d::Identity();
			grid_affine.translate(Eigen::Vector3d(0, 0, -256));
			grid_affine.scale(Eigen::Vector3d(1, 1, -1));	// z is negative inside of screen
			Grid<double> grid(volume_size, voxel_size, grid_affine.matrix());


			//
			// Importing monkey obj
			//
			timer.start();
			std::vector<Eigen::Vector3d> points3DOrig, pointsTmp;
			import_obj("../../data/monkey.obj", points3DOrig);
			timer.print_interval("Importing monkey    : ");
			std::cout << "Monkey point count  : " << points3DOrig.size() << std::endl;

			// 
			// Translating and rotating monkey point cloud 
			std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> cloud;
			//
			Eigen::Affine3d rotate = Eigen::Affine3d::Identity();
			Eigen::Affine3d translate = Eigen::Affine3d::Identity();
			translate.translate(Eigen::Vector3d(0, 0, -256));

			// 
			// Compute first cloud
			//
			for (Eigen::Vector3d p3d : points3DOrig)
			{
				Eigen::Vector4d rot = translate.matrix() * rotate.matrix() * p3d.homogeneous();
				rot /= rot.w();
				cloud.first.push_back(rot.head<3>());
			}
			//
			// Update grid with first cloud
			//
			create_depth_buffer(depth_buffer.first, cloud.first, K, Eigen::Matrix4d::Identity(), far_plane);
			update_volume(grid, depth_buffer.first, K, T.first.inverse());

			//
			// Compute next clouds
			Eigen::Matrix4d cloud_mat = Eigen::Matrix4d::Identity();
			Timer iter_timer;
			for (int i = 1; i < iterations; ++i)
			{
				std::cout << std::endl << i << " : " << i * rot_interval << std::endl;
				iter_timer.start();

				// Rotation matrix
				rotate = Eigen::Affine3d::Identity();
				rotate.rotate(Eigen::AngleAxisd(DegToRad(i * rot_interval), Eigen::Vector3d::UnitY()));

				cloud.second.clear();
				for (Eigen::Vector3d p3d : points3DOrig)
				{
					Eigen::Vector4d rot = translate.matrix() * rotate.matrix() * p3d.homogeneous();
					rot /= rot.w();
					cloud.second.push_back(rot.head<3>());
				}

				timer.start();
				create_depth_buffer(depth_buffer.second, cloud.second, K, Eigen::Matrix4d::Identity(), far_plane);
				timer.print_interval("Compute depth buffer: ");

				timer.start();
				Eigen::Matrix4d icp_mat;
				ComputeRigidTransform(cloud.first, cloud.second, icp_mat);
				timer.print_interval("Compute rigid transf: ");

				std::cout << std::fixed << std::endl << "icp_mat " << std::endl << icp_mat << std::endl;

				// accumulate matrix
				cloud_mat = cloud_mat * icp_mat;

				std::cout << std::fixed << std::endl << "cloud_mat " << std::endl << cloud_mat << std::endl;

				timer.start();
				update_volume(grid, depth_buffer.second, K, cloud_mat.inverse());
				timer.print_interval("Update volume       : ");


				// copy second point cloud to first
				cloud.first = cloud.second;
				depth_buffer.first = depth_buffer.second;

				iter_timer.print_interval("Iteration time      : ");
			}


			//timer.start();
			//export_volume("../../data/grid_volume_monkey_test_2.obj", grid.data);
			//timer.print_interval("Exporting volume    : ");

			//Assert::IsTrue(info_are_equal, L"\n<Info captured from kinect and info loaded from file are not equal>\n", LINE_INFO());
			//Assert::IsTrue(buffer_are_equal, L"\n<Buffer captured from kinect and buffer loaded from file are not equal>\n", LINE_INFO());
		}



		TEST_METHOD(Test3)
		{
			int vol_size = 256;
			int vx_size = 32;
			int iterations = 2;

			Timer timer;

			//
			// Projection and Modelview Matrices
			//
			Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);
			std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());


			//
			// Creating volume
			//
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
			//
			Eigen::Affine3d grid_affine = Eigen::Affine3d::Identity();
			grid_affine.translate(Eigen::Vector3d(0, 0, -256));
			grid_affine.scale(Eigen::Vector3d(1, 1, -1));	// z is negative inside of screen
			Grid<double> grid(volume_size, voxel_size, grid_affine.matrix());


			timer.start();
			std::vector<Eigen::Vector3d> points3DOrig, points0, points1, points2;
			import_obj("../../data/monkey.obj", points3DOrig);
			timer.print_interval("Importing monkey    : ");
			std::cout << "Monkey point count  : " << points3DOrig.size() << std::endl;

			Eigen::Affine3d translate = Eigen::Affine3d::Identity();
			translate.translate(Eigen::Vector3d(0, 0, -256));
			//
			Eigen::Affine3d rotate0 = Eigen::Affine3d::Identity();
			Eigen::Affine3d rotate1 = Eigen::Affine3d::Identity();
			Eigen::Affine3d rotate2 = Eigen::Affine3d::Identity();
			rotate1.rotate(Eigen::AngleAxisd(DegToRad(45.0), Eigen::Vector3d::UnitY()));
			rotate2.rotate(Eigen::AngleAxisd(DegToRad(90.0), Eigen::Vector3d::UnitY()));
			//
			for (Eigen::Vector3d p3d : points3DOrig)
			{
				Eigen::Vector4d trans = translate.matrix() * p3d.homogeneous();
				trans /= trans.w();

				Eigen::Vector4d rot0 = translate.matrix() * rotate0.matrix() * p3d.homogeneous();
				rot0 /= rot0.w();

				Eigen::Vector4d rot1 = translate.matrix() * rotate1.matrix() * p3d.homogeneous();
				rot1 /= rot1.w();

				Eigen::Vector4d rot2 = translate.matrix() * rotate2.matrix() * p3d.homogeneous();
				rot2 /= rot2.w();

				points0.push_back(rot0.head<3>());
				points1.push_back(rot1.head<3>());
				points2.push_back(rot2.head<3>());
			}

			Eigen::Matrix4d cloud_mat_0, cloud_mat_1, cloud_mat_2;
			Eigen::Matrix4d icp_mat_01, icp_mat_12, icp_mat_02;
			ComputeRigidTransform(points0, points1, icp_mat_01);
			ComputeRigidTransform(points1, points2, icp_mat_12);
			ComputeRigidTransform(points0, points2, icp_mat_02);

			std::vector<double> depth_buffer_0, depth_buffer_1, depth_buffer_2;

			cloud_mat_0 = Eigen::Matrix4d::Identity();
			cloud_mat_1 = icp_mat_01;
			cloud_mat_2 = cloud_mat_1 * icp_mat_12;

			//
			// Update grid with cloud 0
			//
			create_depth_buffer(depth_buffer_0, points0, K, Eigen::Matrix4d::Identity(), far_plane);
			update_volume(grid, depth_buffer_0, K, cloud_mat_0.inverse());

			//
			// Update grid with cloud 1
			//
			create_depth_buffer(depth_buffer_1, points1, K, Eigen::Matrix4d::Identity(), far_plane);
			update_volume(grid, depth_buffer_1, K, cloud_mat_1.inverse());
			
			//
			// Update grid with cloud 2
			//
			create_depth_buffer(depth_buffer_2, points2, K, Eigen::Matrix4d::Identity(), far_plane);
			update_volume(grid, depth_buffer_2, K, cloud_mat_2.inverse());

			//timer.start();
			//export_volume("../../data/grid_volume_3_samples.obj", grid.data);
			//timer.print_interval("Exporting volume    : ");
		}


	};


	TEST_CLASS(UnitTestGrid)
	{
	public:

		TEST_METHOD(TestGridIntersection)
		{
			const int expected_intersections_count_case_1 = 17;
			const int expected_intersections_count_case_2 = 4;
			int vol_size = 16;
			int vx_size = 1;

			const Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			const Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);

			// Creating volume
			const Eigen::Vector3i voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
			const std::size_t slice_size = (voxel_count.x() + 1) * (voxel_count.y() + 1);
			std::vector<Voxeld> volume((voxel_count.x() + 1) * (voxel_count.y() + 1) * (voxel_count.z() + 1));

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
			Eigen::Vector3d half_voxel(vx_size * 0.5, vx_size * 0.5, vx_size * 0.5);

			const float t0 = 0.0f;
			const float t1 = 256.0f;

			Eigen::Vector3d origin(0, 0, -32);
			Eigen::Vector3d target(0, 0, -20);
			Eigen::Vector3d direction = (target - origin).normalized();

			int count = 0;
			for (const Voxeld v : volume)
			{
				Eigen::Vector3d corner_min = (volume_transformation * (v.point - half_voxel).homogeneous()).head<3>();
				Eigen::Vector3d corner_max = (volume_transformation * (v.point + half_voxel).homogeneous()).head<3>();

				Box<double> box(corner_min, corner_max);
				Ray<double> ray(origin, direction);

				if (box.intersect(ray, t0, t1))
				{
					std::cout << "Box Intersected: " << v.point.transpose() << std::endl;
					count++;
				}

			}
			std::cout << "Intersections Count: " << count << std::endl;
			Assert::IsTrue(count == expected_intersections_count_case_1);


			origin = Eigen::Vector3d(0, 0, -32);
			target = Eigen::Vector3d(4.0, 3.6, -20);
			direction = (target - origin).normalized();
			count = 0;
			for (const Voxeld v : volume)
			{
				Eigen::Vector3d corner_min = (volume_transformation * (v.point - half_voxel).homogeneous()).head<3>();
				Eigen::Vector3d corner_max = (volume_transformation * (v.point + half_voxel).homogeneous()).head<3>();

				Box<double> box(corner_min, corner_max);
				Ray<double> ray(origin, direction);

				if (box.intersect(ray, t0, t1))
				{
					std::cout << "Box Intersected: " << v.point.transpose() << std::endl;
					count++;
				}

			}
			std::cout << "Intersections Count: " << count << std::endl;

			Assert::IsTrue(count == expected_intersections_count_case_2);
		}


		TEST_METHOD(TestRaycastAll)
		{
			const int vol_size = 2;
			const int vx_size = 1;

			const Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			const Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);

			Grid<double> grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

			const float ray_near = 0;
			const float ray_far = 100;

			const Eigen::Vector3d origin(0, 2, -3);
			const Eigen::Vector3d target(0.72, -1.2, 2);
			const Eigen::Vector3d direction = (target - origin).normalized();

			const std::vector<int> expected_intersections = { 4, 7, 13, 14, 20, 23 };
			const std::vector<int>& intersections = grid.raycast_all(origin, direction, ray_near, ray_far);


			Assert::IsTrue((expected_intersections == intersections), L"\n<Unexpected intersections found>\n", LINE_INFO());
		}


		TEST_METHOD(TestRecursiveRaycast)
		{
			int vol_size = 2;
			int vx_size = 1;

			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);

			Grid<double> grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

			float ray_near = 0;
			float ray_far = 100;

			Eigen::Vector3d origin(0, 2, -3);
			Eigen::Vector3d target(0.72, -1.2, 2);
			Eigen::Vector3d direction = (target - origin).normalized();

			int voxel_index = 7;	// input vertex

			std::vector<int> expected_intersections = { 7, 4, 13, 14, 23, 20 };
			std::vector<int> intersections;
			grid.recursive_raycast(-1, voxel_index, origin, direction, ray_near, ray_far, intersections);


			Assert::IsTrue((expected_intersections == intersections), L"\n<Unexpected intersections found>\n", LINE_INFO());
		}


		TEST_METHOD(TestFind8Neighbour)
		{
			const int vol_size = 16;
			const int vx_size = 1;
			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			Grid<double> grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

			int voxel_index;
			std::vector<int> neighbours_expected, neighbours;

			voxel_index = 0;
			neighbours_expected = { 17, 18, 1 };
			neighbours = grid.neighbour_eight(voxel_index);
			Assert::IsTrue((neighbours_expected == neighbours), L"\n<Unexpected neighbours found>\n", LINE_INFO());

			voxel_index = 16;
			neighbours_expected = { 15, 32, 33 };
			neighbours = grid.neighbour_eight(voxel_index);
			Assert::IsTrue((neighbours_expected == neighbours), L"\n<Unexpected neighbours found>\n", LINE_INFO());

			voxel_index = 272;
			neighbours_expected = { 273, 256, 255 };
			neighbours = grid.neighbour_eight(voxel_index);
			Assert::IsTrue((neighbours_expected == neighbours), L"\n<Unexpected neighbours found>\n", LINE_INFO());

			voxel_index = 288;
			neighbours_expected = { 270, 287, 271 };
			neighbours = grid.neighbour_eight(voxel_index);
			Assert::IsTrue((neighbours_expected == neighbours), L"\n<Unexpected neighbours found>\n", LINE_INFO());


			voxel_index = 4640;
			neighbours_expected = { 4639, 4656, 4657 };
			neighbours = grid.neighbour_eight(voxel_index);
			Assert::IsTrue((neighbours_expected == neighbours), L"\n<Unexpected neighbours found>\n", LINE_INFO());

		}

		TEST_METHOD(TestIndex3dFromArrayIndex)
		{
			const int vol_size = 16;
			const int vx_size = 1;
			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			Grid<double> grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

			int voxel_index;
			Eigen::Vector3i index_3d, index_3d_expected;

			voxel_index = 0;
			index_3d_expected = Eigen::Vector3i(0, 0, 0);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

			voxel_index = 16;
			index_3d_expected = Eigen::Vector3i(16, 0, 0);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

			voxel_index = 272;
			index_3d_expected = Eigen::Vector3i(0, 16, 0);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

			voxel_index = 288;
			index_3d_expected = Eigen::Vector3i(16, 16, 0);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

			voxel_index = 4896;
			index_3d_expected = Eigen::Vector3i(0, 16, 16);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

			voxel_index = 4912;
			index_3d_expected = Eigen::Vector3i(16, 16, 16);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

			voxel_index = 4624;
			index_3d_expected = Eigen::Vector3i(0, 0, 16);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

			voxel_index = 4640;
			index_3d_expected = Eigen::Vector3i(16, 0, 16);
			index_3d = grid.index_3d_from_array_index(voxel_index);
			Assert::IsTrue((index_3d == index_3d_expected), L"\n<Unexpected index 3d found>\n", LINE_INFO());

		}

		TEST_METHOD(TestIndexArrayFrom3dIndex)
		{
			const int vol_size = 16;
			const int vx_size = 1;
			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			Grid<double> grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

			int voxel_index, voxel_index_expected;
			Eigen::Vector3i index_3d;

			voxel_index_expected = 0;
			index_3d = Eigen::Vector3i(0, 0, 0);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());

			voxel_index_expected = 16;
			index_3d = Eigen::Vector3i(16, 0, 0);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());

			voxel_index_expected = 272;
			index_3d = Eigen::Vector3i(0, 16, 0);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());

			voxel_index_expected = 288;
			index_3d = Eigen::Vector3i(16, 16, 0);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());

			voxel_index_expected = 4896;
			index_3d = Eigen::Vector3i(0, 16, 16);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());

			voxel_index_expected = 4912;
			index_3d = Eigen::Vector3i(16, 16, 16);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());

			voxel_index_expected = 4624;
			index_3d = Eigen::Vector3i(0, 0, 16);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());

			voxel_index_expected = 4640;
			index_3d = Eigen::Vector3i(16, 0, 16);
			voxel_index = grid.index_array_from_3d_index(index_3d);
			Assert::IsTrue((voxel_index == voxel_index_expected), L"\n<Unexpected index array found>\n", LINE_INFO());
		}

		TEST_METHOD(TestRayTriangleIntersection)
		{
			const Eigen::Vector3f target(0, 0, 3);
			Eigen::Vector3f origin;
			Eigen::Vector3f direction;

			struct Triangle
			{
				Eigen::Vector3f vertices[3];
				Eigen::Vector3f normal;
			};

			Triangle triangle[2];
			triangle[0].vertices[0] = Eigen::Vector3f(-0.5f, -0.5f, 0.0f);
			triangle[0].vertices[1] = Eigen::Vector3f(-0.5f, 0.5f, 0.0f);
			triangle[0].vertices[2] = Eigen::Vector3f(0.5f, -0.5f, 0.0f);

			triangle[1].vertices[0] = Eigen::Vector3f(0.5f, -0.5f, 0.0f);
			triangle[1].vertices[1] = Eigen::Vector3f(-0.5f, 0.5f, 0.0f);
			triangle[1].vertices[2] = Eigen::Vector3f(0.5f, 0.5f, 0.0f);

			Eigen::Vector3f hit[2];

			origin = Eigen::Vector3f(-0.5f, 0, -3);
			direction = (target - origin).normalized();
			const bool t0 = triangle_intersection(origin, direction, triangle[0].vertices[0], triangle[0].vertices[1], triangle[0].vertices[2], hit[0]);
			Assert::IsTrue(t0, L"\n<Intersection missed>\n", LINE_INFO());

			origin = Eigen::Vector3f(0.5f, 0, -3);
			direction = (target - origin).normalized();
			const bool t1 = triangle_intersection(origin, direction, triangle[1].vertices[0], triangle[1].vertices[1], triangle[1].vertices[2], hit[1]);
			Assert::IsTrue(t1, L"\n<Intersection missed>\n", LINE_INFO());

			//std::cout << std::fixed
			//	<< "t0: " << (t0 ? "hit  " : "fail  ") << hit[0].transpose() << std::endl
			//	<< "t1: " << (t1 ? "hit  " : "fail  ") << hit[1].transpose() << std::endl;
		}

		TEST_METHOD(TestRayPlaneIntersection)
		{
			const Eigen::Vector3f target(0, 0, 3);
			Eigen::Vector3f origin;
			Eigen::Vector3f direction;

			struct Triangle
			{
				Eigen::Vector3f vertices[3];
				Eigen::Vector3f normal;
			};

			Triangle triangle[2];
			triangle[0].vertices[0] = Eigen::Vector3f(-0.5f, -0.5f, 0.0f);
			triangle[0].vertices[1] = Eigen::Vector3f(-0.5f, 0.5f, 0.0f);
			triangle[0].vertices[2] = Eigen::Vector3f(0.5f, -0.5f, 0.0f);

			triangle[1].vertices[0] = Eigen::Vector3f(0.5f, -0.5f, 0.0f);
			triangle[1].vertices[1] = Eigen::Vector3f(-0.5f, 0.5f, 0.0f);
			triangle[1].vertices[2] = Eigen::Vector3f(0.5f, 0.5f, 0.0f);

			Eigen::Vector3f hit[3];

			origin = Eigen::Vector3f(-0.5f, 0, -3);
			direction = (target - origin).normalized();
			const bool p0 = plane_intersection(origin, direction, triangle[0].vertices[0], triangle[0].vertices[1], triangle[0].vertices[2], hit[0]);
			Assert::IsTrue(p0, L"\n<Intersection missed>\n", LINE_INFO());

			origin = Eigen::Vector3f(0.5f, 0, -3);
			direction = (target - origin).normalized();
			const bool p1 = plane_intersection(origin, direction, triangle[1].vertices[0], triangle[1].vertices[1], triangle[1].vertices[2], hit[1]);
			Assert::IsTrue(p1, L"\n<Intersection missed>\n", LINE_INFO());

			bool p2 = plane_intersection(origin, direction, Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f), hit[2]);
			Assert::IsTrue(p2, L"\n<Intersection missed>\n", LINE_INFO());

			//std::cout << std::fixed
			//	<< "p0: " << (p0 ? "hit  " : "fail  ") << hit[0].transpose() << std::endl
			//	<< "p1: " << (p1 ? "hit  " : "fail  ") << hit[1].transpose() << std::endl
			//	<< "p2: " << (p2 ? "hit  " : "fail  ") << hit[2].transpose() << std::endl;
		}
	};
}