#define Use_Kinect_V1 1


#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QDateTime>
#include "QImageWidget.h"
#include "QKinectFile.h"
#if Use_Kinect_V1 
#include "QKinectGrabberV1.h"
QKinectGrabberV1*	kinect;
#else
#include "QKinectGrabber.h"
QKinectGrabber*		kinect;
#endif
#include "QKinectIO.h"
#include <iostream>


int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	int width = 640;
	int height = 480;

#if Use_Kinect_V1 
	kinect = new QKinectGrabberV1();
#else
	kinect = new QKinectGrabber();
#endif
	kinect->start();

	QDateTime now = QDateTime::currentDateTime();
	QString output_folder = "../" + now.toString("yyyyMMdd_hhmmss");

	QDir dir(output_folder);
	if (!dir.exists())
		dir.mkdir(output_folder);
	
	//QString dir = QFileDialog::getExistingDirectory(nullptr, "Output Directory",
	//	"../",
	//	QFileDialog::ShowDirsOnly
	//	| QFileDialog::DontResolveSymlinks);


	QKinectFile kinectFile;
	kinectFile.setFolder(output_folder);
	kinectFile.setKinect(kinect);

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(640, 480);
	depthWidget.move(50, 50);
	depthWidget.show();
	QApplication::connect(kinect, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));
	QApplication::connect(kinect, SIGNAL(frameUpdated()), &kinectFile, SLOT(save()));



	int app_exit = app.exec();
	kinect->stop();


	QMessageBox msgBox;
	msgBox.setText("Kinect stream has been saved into " + output_folder);
	msgBox.exec();


	return app_exit;

}

