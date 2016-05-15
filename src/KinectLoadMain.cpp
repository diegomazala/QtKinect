#include <QApplication>
#include <QDir>
#include "QImageWidget.h"
#include "QKinectFile.h"
#include "QKinectGrabberFromFile.h"
#include "QKinectIO.h"
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

	QKinectGrabberFromFile* kinect_grabber = nullptr;
	QKinectFrame* kinect_frame = nullptr;

	QApplication app(argc, argv);

	int width = 640;
	int height = 480;

	QImageWidget colorWidget;
	colorWidget.setMinimumSize(640, 480);
	colorWidget.move(0, 0);
	colorWidget.show();

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(640, 480);
	depthWidget.move(640, 0);
	depthWidget.show();

	if (QFileInfo(input_path).isDir())
	{
		kinect_grabber = new QKinectGrabberFromFile();
		kinect_grabber->setFolder(input_path);
		kinect_grabber->setFramesPerSecond(fps);
		kinect_grabber->start();

		QApplication::connect(kinect_grabber, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));
		QApplication::connect(kinect_grabber, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));
		QApplication::connect(kinect_grabber, SIGNAL(fileLoaded(QString)), &colorWidget, SLOT(setWindowTitle(QString)));
		QApplication::connect(kinect_grabber, SIGNAL(fileLoaded(QString)), &depthWidget, SLOT(setWindowTitle(QString)));
	}
	else
	{
		kinect_frame = new QKinectFrame(input_path);
		colorWidget.setImage(kinect_frame->convertColorToImage());
		depthWidget.setImage(kinect_frame->convertDepthToImage());
	}


	int app_exit = app.exec();
	
	if (kinect_grabber)
		kinect_grabber->stop();

	return app_exit;
}

