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
#include "Eigen/Eigen"

#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.

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


static Eigen::Matrix4f perspective_matrix(float fovy, float aspect_ratio, float near_plane, float far_plane)
{
	Eigen::Matrix4f out = Eigen::Matrix4f::Zero();

	const float	y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
	const float	x_scale = y_scale / aspect_ratio;
	const float	depth_length = far_plane - near_plane;

	out(0, 0) = x_scale;
	out(1, 1) = y_scale;
	out(2, 2) = -((far_plane + near_plane) / depth_length);
	out(3, 2) = -1.0;
	out(2, 3) = -((2 * near_plane * far_plane) / depth_length);

	return out;
}

static Eigen::Matrix4f perspective_matrix_inverse(float fovy, float aspect_ratio, float near_plane, float far_plane)
{
	Eigen::Matrix4f out = Eigen::Matrix4f::Zero();

	const float	y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
	const float	x_scale = y_scale / aspect_ratio;
	const float	depth_length = far_plane - near_plane;

	out(0, 0) = 1.0 / x_scale;
	out(1, 1) = 1.0 / y_scale;
	out(2, 3) = -1.0f;
	out(3, 2) = -1.0f / ((2 * near_plane * far_plane) / depth_length);
	out(3, 3) = ((far_plane + near_plane) / depth_length) / ((2 * near_plane * far_plane) / depth_length);

	return out;
}

Eigen::Vector3f vertex_to_window_coord(Eigen::Vector4f p3d, float fovy, float aspect_ratio, float near_plane, float far_plane, int window_width, int window_height)
{
	const Eigen::Matrix4f proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

	const Eigen::Vector4f p_clip = proj * p3d;

	const Eigen::Vector3f p_ndc = (p_clip / p_clip.w()).head<3>();

	Eigen::Vector3f p_window;
	p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
	p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;
	p_window.z() = (far_plane - near_plane) / 2.0f * p_ndc.z() + (far_plane + near_plane) / 2.0f;

	return p_window;
}


Eigen::Vector3f window_coord_to_3d(Eigen::Vector2f pixel, float depth, float fovy, float aspect_ratio, float near_plane, float far_plane, int window_width, int window_height)
{
	Eigen::Vector3f ndc;
	ndc.x() = (pixel.x() - (window_width / 2.0f)) / (window_width / 2.0f);
	ndc.y() = (pixel.y() - (window_height / 2.0f)) / (window_height / 2.0f);
	ndc.z() = -1.0f;

	const Eigen::Vector3f clip = ndc * depth;

	const Eigen::Matrix4f proj_inv = perspective_matrix_inverse(fovy, aspect_ratio, near_plane, far_plane);
	const Eigen::Vector4f vertex_proj_inv = proj_inv * clip.homogeneous();

	Eigen::Vector3f p3d_final;
	p3d_final.x() = -vertex_proj_inv.x();
	p3d_final.y() = -vertex_proj_inv.y();
	p3d_final.z() = depth;

	return p3d_final;
}


namespace TestQtKinect
{		
	TEST_CLASS(UnitTest1)
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
			Eigen::Vector4f p3d(-0.5f, -0.5f, -0.88f, 1.0f);
			Eigen::Vector3f pixel(285.71f, 5.71f, 88.73f);

			float window_width = 1280.0f;
			float window_height = 720.0f;
			float near_plane = 0.1f; 
			float far_plane = 100.0f;
			float fovy = 60.0f;
			float aspect_ratio = window_width / window_height;
			float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
			float x_scale = y_scale / aspect_ratio;
			float depth_length = far_plane - near_plane;

			Eigen::Matrix4f Mdv = Eigen::Matrix4f::Identity();
			Mdv.col(3) << 0.f, 0.f, 0.0f, 1.f;

			Eigen::Matrix4f Proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

			Eigen::Vector4f p_clip = Proj * Mdv * p3d;

			Eigen::Vector3f p_ndc = (p_clip / p_clip.w()).head<3>();

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
			float depth_length = far_plane - near_plane;

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
			float depth_length = far_plane - near_plane;

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
				Eigen::Vector4f p_clip = Proj * Mdv.matrix() * p3d.homogeneous();
				Eigen::Vector3f p_ndc = (p_clip / p_clip.w()).head<3>();

				Eigen::Vector3f p_window;
				p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
				p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;
				p_window.z() = (far_plane - near_plane) / 2.0f * p_ndc.z() + (far_plane + near_plane) / 2.0f;

				int d = ((p3d.z() + min_z) / (max_z - min_z) * 255);

				if (p_window.x() > 0 && p_window.x() < window_width && p_window.y() > 0 && p_window.y() < window_height)
				{
					color.setPixel(QPoint(p_window.x(), window_height - p_window.y()), qRgb(255, 0, 0));
					depth.setPixel(QPoint(p_window.x(), window_height - p_window.y()), qRgb(0, 0, d));
				}
			}

			Assert::IsTrue(color.save("monkey_color.png") && depth.save("monkey_depth.png"));
		}


		TEST_METHOD(TestGridIntersection)
		{

			const int expected_intersections_count_case_1 = 17;
			const int expected_intersections_count_case_2 = 4;
			int vol_size = 16;
			int vx_size = 1;

			const Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
			const Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
			
			
			// Creating volume
			Eigen::Vector3i voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
			std::size_t slice_size = (voxel_count.x() + 1) * (voxel_count.y() + 1);
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

			float t0 = 0.0f;
			float t1 = 256.0f;

			Eigen::Vector3d cube(0, 0, 0);

			Eigen::Vector3d origin(0, 0, -32);
			Eigen::Vector3d target(0, 0, -20);
			Eigen::Vector3d direction = (target - origin).normalized();

			int count = 0;
			for (const Voxeld v : volume)
			{
				Eigen::Vector3d corner_min = (volume_transformation * (v.point - half_voxel).homogeneous()).head<3>();
				Eigen::Vector3d corner_max = (volume_transformation * (v.point + half_voxel).homogeneous()).head<3>();

				Box box(corner_min, corner_max);
				Ray ray(origin, direction);
				
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

				Box box(corner_min, corner_max);
				Ray ray(origin, direction);

				if (box.intersect(ray, t0, t1))
				{
					std::cout << "Box Intersected: " << v.point.transpose() << std::endl;
					count++;
				}

			}
			std::cout << "Intersections Count: " << count << std::endl;

			Assert::IsTrue(count == expected_intersections_count_case_2);
		}

	};
}