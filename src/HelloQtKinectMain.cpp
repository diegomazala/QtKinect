
#include <QApplication>
#include "QImageWidget.h"
#include "QKinectGrabberV1.h"
#include "QKinectIO.h"
#include <iostream>


int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	QKinectGrabberV1 k;
	k.start();

	QImageWidget colorWidget;
	//colorWidget.setMinimumSize(720, 480);
	colorWidget.setMinimumSize(640, 480);
	colorWidget.show();
	QApplication::connect(&k, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));

	QImageWidget depthWidget;
	//depthWidget.setMinimumSize(512, 424);
	depthWidget.setMinimumSize(640, 480);
	depthWidget.show();
	QApplication::connect(&k, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));

	//QImageWidget infraredWidget;
	//infraredWidget.setMinimumSize(512, 424);
	//infraredWidget.show();
	//QApplication::connect(&k, SIGNAL(infraredImage(QImage)), &infraredWidget, SLOT(setImage(QImage)));

	int app_exit = app.exec();
	k.stop();
	return app_exit;

}

