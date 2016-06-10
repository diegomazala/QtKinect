#include <QApplication>
#include <QDir>
#include "QImageWidget.h"
#include "QKinectFile.h"
#include "QKinectGrabberFromFile.h"
#include "QKinectIO.h"
#include "KinectFusionManager.h"
#include "KinectPlayerWidget.h"
#include <iostream>

// Usage: ./KinectLoadd.exe ../knt_frames/ 5
// Usage: ./KinectLoadd.exe ../knt_frames/frame_33.knt
int main(int argc, char **argv)
{
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


	QImageWidget colorWidget;
	colorWidget.setMinimumSize(640, 480);
	colorWidget.move(0, 0);
	colorWidget.show();

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(640, 480);
	depthWidget.move(640, 0);
	depthWidget.show();

	QKinectGrabberFromFile* kinect_grabber = new QKinectGrabberFromFile();
	kinect_grabber->setFolder(input_path);
	kinect_grabber->setFramesPerSecond(fps);
	
	QApplication::connect(kinect_grabber, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));
	QApplication::connect(kinect_grabber, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));
	QApplication::connect(kinect_grabber, SIGNAL(fileLoaded(QString)), &colorWidget, SLOT(setWindowTitle(QString)));
	QApplication::connect(kinect_grabber, SIGNAL(fileLoaded(QString)), &depthWidget, SLOT(setWindowTitle(QString)));
	
	kinect_grabber->stopAndGo(true);
	kinect_grabber->start();


	KinectFusionManager mngr(input_path);

	KinectPlayerWidget player(kinect_grabber);
	player.move(320, 500);
	player.show();
	

	QApplication::connect(&player, SIGNAL(quit()), &depthWidget, SLOT(close()));
	QApplication::connect(&player, SIGNAL(quit()), &colorWidget, SLOT(close()));
	QApplication::connect(&player, SIGNAL(quit()), &mngr, SLOT(finish()));

	int app_exit = app.exec();
	return app_exit;

	
}

