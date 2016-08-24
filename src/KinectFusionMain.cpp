#include <QApplication>
#include <QDir>
#include "QImageWidget.h"
#include "QKinectFile.h"
#include "QKinectGrabberFromFile.h"
#include "QKinectIO.h"
#include "KinectFusionManager.h"
#include "KinectPlayerWidget.h"
#include "VolumeRenderWidget.h"
#include <iostream>

int volumetricRenderTest(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: KinectFusiion.exe ../../data/monkey_tsdf_float2_33.raw 33" << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = "../../data/monkey_tsdf_float2_33.raw"; // argv[1];

	size_t vol_width = 33;
	size_t vol_height = 33;
	size_t vol_depth = 33;

	//if (argc > 4)
	//{
	//	vol_width = atoi(argv[2]);
	//	vol_height = atoi(argv[3]);
	//	vol_depth = atoi(argv[4]);
	//}
	//else if (argc > 3)
	//{
	//	vol_width = atoi(argv[2]);
	//	vol_height = atoi(argv[3]);
	//	vol_height = 1;
	//}
	//else
	//{
	//	vol_width = vol_height = vol_depth = atoi(argv[2]);;
	//}


	QApplication app(argc, argv);
	app.setApplicationName("Qt Point Cloud OpenGL View");
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	//format.setMajorVersion(3);
	//format.setMinorVersion(3);
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setOption(QSurfaceFormat::DebugContext);
	QSurfaceFormat::setDefaultFormat(format);



	VolumeRenderWidget widget;
	widget.setup(filename, vol_width, vol_height, vol_depth);
	widget.setFixedSize(512, 512);
	widget.setWindowTitle("Volume Render Widget");
	widget.show();





	return app.exec();
}

// Usage: ./KinectLoadd.exe ../knt_frames/ 5
// Usage: ./KinectLoadd.exe ../knt_frames/frame_33.knt
int main(int argc, char **argv)
{
	//return volumetricRenderTest(argc, argv);

	QString input_path = QDir::currentPath();
	int fps = 30;

	if (argc > 1)
		input_path = argv[1];

	if (argc > 2)
		fps = atoi(argv[2]);


	//
	// check if the path is a valid folder
	// 
	if (!QFileInfo(input_path).isDir())
	{
		std::cerr << "Error: A valid folder is required" << std::endl;
		return EXIT_FAILURE;
	}

	QApplication app(argc, argv);

	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setOption(QSurfaceFormat::DebugContext);
	QSurfaceFormat::setDefaultFormat(format);

	std::string filename = "../../data/monkey_tsdf_float2_33.raw"; // argv[1];
	size_t vol_width = 33;
	size_t vol_height = 33;
	size_t vol_depth = 33;
	VolumeRenderWidget volumeWidget;
	volumeWidget.setup(filename, vol_width, vol_height, vol_depth);
	volumeWidget.setFixedSize(512, 512);
	volumeWidget.setWindowTitle("Volume Render Widget");
	volumeWidget.move(1280, 0);
	volumeWidget.show();



	QImageWidget colorWidget;
	colorWidget.setMinimumSize(640, 480);
	colorWidget.move(0, 0);
	colorWidget.show();

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(640, 480);
	depthWidget.move(640, 0);
	depthWidget.show();

	QKinectGrabberFromFile* kinectGrabber = new QKinectGrabberFromFile();
	kinectGrabber->setFolder(input_path);
	kinectGrabber->setFramesPerSecond(fps);
	
	QApplication::connect(kinectGrabber, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));
	QApplication::connect(kinectGrabber, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));
	QApplication::connect(kinectGrabber, SIGNAL(fileLoaded(QString)), &colorWidget, SLOT(setWindowTitle(QString)));
	QApplication::connect(kinectGrabber, SIGNAL(fileLoaded(QString)), &depthWidget, SLOT(setWindowTitle(QString)));
	
	kinectGrabber->stopAndGo(true);
	kinectGrabber->start();

	
	KinectPlayerWidget player(kinectGrabber);
	player.move(320, 500);
	player.show();
	
	KinectFusionManager kinectMngr(kinectGrabber);

	QApplication::connect(&player, SIGNAL(quit()), &depthWidget, SLOT(close()));
	QApplication::connect(&player, SIGNAL(quit()), &colorWidget, SLOT(close()));
	QApplication::connect(&player, SIGNAL(quit()), &volumeWidget, SLOT(close()));

	QApplication::connect(kinectGrabber, SIGNAL(frameUpdated()), &kinectMngr, SLOT(onNewFrame()));

	int app_exit = app.exec();
	return app_exit;

	
}

