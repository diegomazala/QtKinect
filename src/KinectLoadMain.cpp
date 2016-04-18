#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QDateTime>
#include "QImageWidget.h"
#include "QKinectFile.h"
#include "QKinectGrabberFromFile.h"
#include "QKinectIO.h"
#include <iostream>


int main(int argc, char **argv)
{
	QDateTime now = QDateTime::currentDateTime();
	QString input_folder = ".";

	if (argc > 1)
		input_folder = argv[1];

	QApplication app(argc, argv);

	int width = 640;
	int height = 480;

	QKinectGrabberFromFile* kinect = new QKinectGrabberFromFile();
	kinect->setFolder(input_folder);
	kinect->start();


	QImageWidget colorWidget;
	colorWidget.setMinimumSize(640, 480);
	colorWidget.move(0, 0);
	colorWidget.show();
	QApplication::connect(kinect, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(640, 480);
	depthWidget.move(50, 50);
	depthWidget.show();
	QApplication::connect(kinect, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));


	int app_exit = app.exec();
	kinect->stop();

	return app_exit;
}

