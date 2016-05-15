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


static bool import_obj(const std::string& filename, std::vector<float>& points3d, int max_point_count = INT_MAX)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	points3d.clear();

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

			points3d.push_back(x);
			points3d.push_back(y);
			points3d.push_back(z);
		}

		if (i++ > max_point_count)
			break;
	}

	inFile.close();
	return true;
}


static bool import_obj(const std::string& filename, std::vector<Eigen::Vector3f>& points3d, int max_point_count = INT_MAX)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	points3d.clear();

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

		std::stringstream ss(str);
		std::vector <std::string> record;

		char c;
		float x, y, z;
		ss >> c >> x >> y >> z;

		Eigen::Vector3f p(x, y, z);
		points3d.push_back(p);

		if (i++ > max_point_count)
			break;
	}

	inFile.close();
	return true;
}




namespace TestQtKinect
{		
	TEST_CLASS(UnitTestKinect)
	{
	public:
		
		TEST_METHOD(TestKinectCapture)
		{
			int argc = 1;
			char* argv[] = { "TestKinectColor" };
			QApplication app(argc, argv);
			QTimer *timer = new QTimer();
			timer->start(5000);
			QApplication::connect(timer, SIGNAL(timeout()), &app, SLOT(quit()));

			QKinectGrabber k;
			k.start();

			Assert::IsTrue(k.isRunning(), L"\n<Kinect is not running>\n", LINE_INFO());

			QImageWidget colorWidget;
			colorWidget.setMinimumSize(720, 480);
			colorWidget.show();
			QApplication::connect(&k, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));

			QImageWidget depthWidget;
			depthWidget.setMinimumSize(512, 424);
			depthWidget.show();
			QApplication::connect(&k, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));

			int app_exit = app.exec();
			k.stop();
			 
			Assert::AreEqual(EXIT_SUCCESS, app_exit, L"\n<TestKinectColor appplication did not finish properly>\n", LINE_INFO());
		}

		TEST_METHOD(TestKinectIO)
		{
			QKinectGrabber kinectCapture;
			kinectCapture.start();

			Timer::sleep_ms(3000);

			Assert::IsTrue(kinectCapture.isRunning(), L"\n<Kinect is not running>\n", LINE_INFO());

			QString filename("TestKinectIO.knt");
						
			std::vector<unsigned short> info_captured; 
			std::vector<unsigned short> buffer_captured;

			kinectCapture.getDepthBuffer(info_captured, buffer_captured);

			QKinectIO::save(filename, info_captured, buffer_captured);

			std::vector<unsigned short> info_loaded;
			std::vector<unsigned short> buffer_loaded;

			QKinectIO::load(filename, info_loaded, buffer_loaded);

			Assert::AreEqual(info_captured.size(), info_loaded.size(), L"\n<The size of info vector captured from kinect and the info vector loaded from file are not equal>\n", LINE_INFO());
			Assert::AreEqual(buffer_captured.size(), buffer_loaded.size(), L"\n<The size of buffer captured from kinect and the buffer loaded from file are not equal>\n", LINE_INFO());

			bool info_are_equal = std::equal(info_captured.begin(), info_captured.end(), info_loaded.begin());
			bool buffer_are_equal = std::equal(buffer_captured.begin(), buffer_captured.end(), buffer_loaded.begin());

			Assert::IsTrue(info_are_equal, L"\n<Info captured from kinect and info loaded from file are not equal>\n", LINE_INFO());
			Assert::IsTrue(buffer_are_equal, L"\n<Buffer captured from kinect and buffer loaded from file are not equal>\n", LINE_INFO());
		}



		TEST_METHOD(TestKinectOBJ)
		{
			QString filenameKnt("TestKinectIO.knt");
			QString filenameObj("TestKinectIO.obj");

			std::vector<unsigned short> info_loaded;
			std::vector<unsigned short> buffer_loaded;

			QKinectIO::load(filenameKnt, info_loaded, buffer_loaded);
			QKinectIO::exportObj(filenameObj, info_loaded, buffer_loaded);
			
			Assert::IsTrue(true);
		}


	};


	TEST_CLASS(UnitTestProjection)
	{
	public:



		TEST_METHOD(TestProjectionPipelineKMatrix)
		{
			std::string filename("../../data/monkey/monkey.obj");

			std::vector<float> points3d;
			import_obj(filename, points3d);

			Eigen::Vector4f p3d(24.5292f, 21.9753f, 29.9848f, 1.0f);
			Eigen::Vector4f p2d;

			float fovy = 70.0f;
			float aspect_ratio = 512.0f / 424.0f;
			float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
			float x_scale = y_scale / aspect_ratio;

			Eigen::MatrixXf K = Eigen::MatrixXf(3, 4);
			K.setZero();
			K(0, 0) = x_scale;
			K(1, 1) = y_scale;
			K(2, 2) = 1;

			Eigen::MatrixXf K_inv = K;
			K_inv(0, 0) = 1.0f / K_inv(0, 0);
			K_inv(1, 1) = 1.0f / K_inv(1, 1);
			std::cout << K_inv << std::endl << std::endl;

			Eigen::Vector3f kp = K * p3d;
			kp /= kp.z();

			p2d = kp.homogeneous();
			p2d *= p3d.z();

			Eigen::Vector3f p3d_out = K_inv * p2d;

			std::cout << std::fixed
				<< "Input point 3d  : " << p3d.transpose() << std::endl
				<< "Projected point : " << kp.transpose() << std::endl
				<< "Input point 2d  : " << p2d.transpose() << std::endl
				<< "Output point 3d : " << p3d_out.transpose() << std::endl
				<< "Test Passed     : " << (p3d.isApprox(p3d_out.homogeneous()) ? "[ok]" : "[fail]") << std::endl;

			Assert::IsTrue(p3d.isApprox(p3d_out.homogeneous()));
		}


		TEST_METHOD(TestProjection)
		{
			const Eigen::Vector4f p3d(-0.5f, -0.5f, -0.88f, 1.0f);
			const Eigen::Vector3f pixel(285.71f, 5.71f, 88.73f);

			const float window_width = 1280.0f;
			const float window_height = 720.0f;
			const float near_plane = 0.1f;
			const float far_plane = 100.0f;
			const float fovy = 60.0f;
			const float aspect_ratio = window_width / window_height;
			const float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
			const float x_scale = y_scale / aspect_ratio;

			Eigen::Matrix4f Mdv = Eigen::Matrix4f::Identity();
			Mdv.col(3) << 0.f, 0.f, 0.0f, 1.f;

			const Eigen::Matrix4f Proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

			const Eigen::Vector4f p_clip = Proj * Mdv * p3d;

			const Eigen::Vector3f p_ndc = (p_clip / p_clip.w()).head<3>();

			Eigen::Vector3f p_window;
			p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
			p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;
			p_window.z() = (far_plane - near_plane) / 2.0f * p_ndc.z() + (far_plane + near_plane) / 2.0f;

			Assert::IsTrue(pixel.isApprox(p_window, 0.01f));
		}


		TEST_METHOD(TestWindowCoordTo3DWorld)
		{
			Eigen::Vector3f p3d(-0.5f, -0.5f, -0.88f);
			Eigen::Vector2f pixel(285.716888f, 5.716888f);
			float depth = p3d.z();

			float window_width = 1280.0f;
			float window_height = 720.0f;
			float near_plane = 0.1f;
			float far_plane = 100.0f;
			float fovy = 60.0f;
			float aspect_ratio = window_width / window_height;
			float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
			float x_scale = y_scale / aspect_ratio;

			Eigen::Vector3f ndc;
			ndc.x() = (pixel.x() - (window_width / 2.0f)) / (window_width / 2.0f);
			ndc.y() = (pixel.y() - (window_height / 2.0f)) / (window_height / 2.0f);
			ndc.z() = -1.0f;

			Eigen::Vector3f clip = ndc * depth;

			Eigen::Matrix4f proj_inv = perspective_matrix_inverse(fovy, aspect_ratio, near_plane, far_plane);
			Eigen::Vector4f vertex_proj_inv = proj_inv * clip.homogeneous();

			Eigen::Vector3f p3d_out = -vertex_proj_inv.head<3>();
			p3d_out.z() = depth;

			Assert::IsTrue(p3d_out.isApprox(p3d, 0.01f));
		}

		TEST_METHOD(TestProjectToImage)
		{
			float window_width = 512.0f;
			float window_height = 424.0f;
			float near_plane = 0.1f;
			float far_plane = 100.0f;
			float fovy = 70.0f;
			float aspect_ratio = window_width / window_height;
			float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
			float x_scale = y_scale / aspect_ratio;

			Eigen::Affine3f Mdv = Eigen::Affine3f::Identity();
			Mdv.translate(Eigen::Vector3f(0, 0, -105));
			Mdv.rotate(Eigen::AngleAxisf(DegToRad(90.0), Eigen::Vector3f::UnitY()));		// 90º

			Eigen::Matrix4f Proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

			QImage depth(window_width, window_height, QImage::Format_RGB888);
			depth.fill(Qt::white);
			QImage color(window_width, window_height, QImage::Format_RGB888);
			color.fill(Qt::white);

			std::string filename("../../data/monkey/monkey.obj");

			std::vector<Eigen::Vector3f> points3d;
			import_obj(filename, points3d);

			float max_z = -99999;
			float min_z = FLT_MAX;

			for (const Eigen::Vector3f p3d : points3d)
			{
				if (p3d.z() < min_z)
					min_z = p3d.z();
				if (p3d.z() > max_z)
					max_z = p3d.z();
			}

			for (const Eigen::Vector3f p3d : points3d)
			{
				const Eigen::Vector4f p_clip = Proj * Mdv.matrix() * p3d.homogeneous();
				const Eigen::Vector3f p_ndc = (p_clip / p_clip.w()).head<3>();

				Eigen::Vector3f p_window;
				p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
				p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;
				p_window.z() = (far_plane - near_plane) / 2.0f * p_ndc.z() + (far_plane + near_plane) / 2.0f;

				const int d = ((p3d.z() + min_z) / (max_z - min_z) * 255);

				if (p_window.x() > 0 && p_window.x() < window_width && p_window.y() > 0 && p_window.y() < window_height)
				{
					color.setPixel(QPoint(p_window.x(), window_height - p_window.y()), qRgb(255, 0, 0));
					depth.setPixel(QPoint(p_window.x(), window_height - p_window.y()), qRgb(0, 0, d));
				}
			}

			Assert::IsTrue(color.save("monkey_color.png") && depth.save("monkey_depth.png"));
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

			Grid grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

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

			Grid grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

			float ray_near = 0; 
			float ray_far = 100;

			Eigen::Vector3d origin(0, 2, -3);
			Eigen::Vector3d target(0.72, -1.2, 2);
			Eigen::Vector3d direction = (target - origin).normalized();

			int voxel_index = 7;	// input vertex

			std::vector<int> expected_intersections = {7, 4, 13, 14, 23, 20};
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
			Grid grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

			int voxel_index;
			std::vector<int> neighbours_expected, neighbours;

			voxel_index = 0;
			neighbours_expected = { 17, 18, 1 };
			neighbours = grid.neighbour_eight(voxel_index);
			Assert::IsTrue((neighbours_expected == neighbours), L"\n<Unexpected neighbours found>\n", LINE_INFO());

			voxel_index = 16;
			neighbours_expected = { 15, 32, 33};
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
			neighbours_expected = { 4639, 4656, 4657};
			neighbours = grid.neighbour_eight(voxel_index);
			Assert::IsTrue((neighbours_expected == neighbours), L"\n<Unexpected neighbours found>\n", LINE_INFO());

		}

		TEST_METHOD(TestIndex3dFromArrayIndex)
		{
			const int vol_size = 16;
			const int vx_size = 1;
			Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			Grid grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

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
			index_3d_expected = Eigen::Vector3i(16, 16,16);
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
			Grid grid(volume_size, voxel_size, Eigen::Matrix4d::Identity());

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