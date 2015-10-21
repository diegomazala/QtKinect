#include "QKinectPlayerCtrl.h"
#include "MainWindow.h"


QKinectPlayerCtrl::QKinectPlayerCtrl(QObject *parent)
	: QObject(parent),
	view(nullptr)
{
}


QKinectPlayerCtrl::~QKinectPlayerCtrl()
{
}


void QKinectPlayerCtrl::setView(QWidget* viewUI)
{
	view = reinterpret_cast<MainWindow*>(viewUI);
}


void QKinectPlayerCtrl::setupConnections()
{
	if (view != nullptr)
	{
		connect(&kinectReader, SIGNAL(colorImage(QImage)), view, SLOT(setColorImage(QImage)));
		connect(&kinectReader, SIGNAL(depthImage(QImage)), view, SLOT(setDepthImage(QImage)));
	}
}


void QKinectPlayerCtrl::startKinect()
{
	kinectReader.enableImageSending(true);
	kinectReader.start();
}


void QKinectPlayerCtrl::stopKinect()
{
	kinectReader.stop();
}

