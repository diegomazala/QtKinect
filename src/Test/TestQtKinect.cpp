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
#include "Eigen/Eigen"


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

static Eigen::Matrix4f createPerspectiveMatrix(float fy, float in_aspect_ratio, float in_near_plane, float in_far_plane)
{
	Eigen::Matrix4f out = Eigen::Matrix4f::Zero();

	const float
		y_scale = (float)1.0 / tan((fy / 2.0)*(M_PI / 180.0)),
		x_scale = y_scale / in_aspect_ratio,
		frustum_length = in_far_plane - in_near_plane;

	out(0, 0) = x_scale;
	out(1, 1) = y_scale;
	out(2, 2) = -((in_far_plane + in_near_plane) / frustum_length);
	out(3, 2) = -1.0;
	out(2, 3) = -((2 * in_near_plane * in_far_plane) / frustum_length);

	return out;
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


		TEST_METHOD(TestProjectionPipeline)
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

			Eigen::Matrix4f Proj = createPerspectiveMatrix(fovy, aspect_ratio, near_plane, far_plane);

			Eigen::Vector4f p_clip = Proj * Mdv * p3d;

			Eigen::Vector3f p_ndc = (p_clip / p_clip.w()).head<3>();

			Eigen::Vector3f p_window;
			p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
			p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;
			p_window.z() = (far_plane - near_plane) / 2.0f * p_ndc.z() + (far_plane + near_plane) / 2.0f;

			Assert::IsTrue(pixel.isApprox(p_window, 0.01f));
		}
	};
}