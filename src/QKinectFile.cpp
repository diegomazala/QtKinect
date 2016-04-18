
#include "QKinectFile.h"
#include "QKinectIO.h"
#include <QDateTime>
#include <iostream>


QKinectFile::QKinectFile(QObject* parent) : 
	QObject(parent)
	, kinect(nullptr)
	, folderPath(".")
{

}

QKinectFile::~QKinectFile()
{
	kinect = nullptr;
}


void QKinectFile::setFolder(QString folder_path)
{
	folderPath = folder_path;
}


void QKinectFile::setKinect(QKinectGrabberV1* kinect_ptr)
{
	kinect = kinect_ptr;
}



void QKinectFile::save()
{
	KinectFrame frame;
	kinect->copyFrameBuffer(frame);

	QString frame_name = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss-zzz");
	QString filename = folderPath + "/" + frame_name + ".knt";

	QKinectIO::saveFrame(filename, frame);
};

