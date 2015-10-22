#include "QKinectPlayerCtrl.h"
#include "MainWindow.h"
#include <QFileDialog>
#include <iostream>


QKinectPlayerCtrl::QKinectPlayerCtrl(QObject *parent)
	: QObject(parent),
	kinectReader(this),
	kinectStream(),
	view(nullptr),
	recording(false)
{
	kinectStream.setKinecReader(&kinectReader);
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

		connect(view, SIGNAL(recordToggled(bool)), this, SLOT(record(bool)));
		connect(view, SIGNAL(play()), this, SLOT(playStream()));
		connect(view, SIGNAL(stop()), this, SLOT(stopStream()));
	}
	connect(&kinectReader, SIGNAL(frameUpdated()), this, SLOT(updateFrame()));
}

bool QKinectPlayerCtrl::isRecording() const
{
	return recording;
}

void QKinectPlayerCtrl::updateFrame()
{
	if (isRecording())
		kinectStream.appendFrame();
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


void QKinectPlayerCtrl::record(bool triggered)
{
	recording = triggered;

	if (recording == false)
	{
		QString fileName = QFileDialog::getSaveFileName(nullptr, tr("Save File"), "", tr("Kinect Stream (*.knt)"));
		if (!fileName.isEmpty())
		{
			kinectStream.save(fileName);
			kinectStream.clear();
		}
	}
}



void QKinectPlayerCtrl::playStream()
{
	//kinectStream.appendFrame();
}

void QKinectPlayerCtrl::stopStream()
{
	//kinectStream.save("C:/temp/test.knt");
}

