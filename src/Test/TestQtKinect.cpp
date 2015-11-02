// Including SDKDDKVer.h defines the highest available Windows platform.
// If you wish to build your application for a previous Windows platform, include WinSDKVer.h and
// set the _WIN32_WINNT macro to the platform you wish to support before including SDKDDKVer.h.
#include <SDKDDKVer.h>

#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include <algorithm>
#include <QApplication>
#include <QTimer>
#include "QImageWidget.h"
#include "QKinectReader.h"
#include "QKinectIO.h"
#include "Timer.h"


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

			QKinectReader k;
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
			QKinectReader kinectCapture;
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
}