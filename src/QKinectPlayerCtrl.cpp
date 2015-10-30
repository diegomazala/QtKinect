#include "QKinectPlayerCtrl.h"
#include "MainWindow.h"
#include "GLDepthBufferRenderer.h"
#include <QFileDialog>
#include <QDateTime>
#include <iostream>


QKinectPlayerCtrl::QKinectPlayerCtrl(QObject *parent)
	: QObject(parent),
	kinectReader(this),
	kinectStream(),
	view(nullptr),
	depthRenderer(nullptr),
	recording(false),
	mDepthBuffer()
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


void QKinectPlayerCtrl::setDepthRenderer(GLDepthBufferRenderer* depthRenderer)
{
	depthRenderer->setController(*this);
}


void QKinectPlayerCtrl::setupConnections()
{
	if (view != nullptr)
	{
		connect(&kinectReader, SIGNAL(colorImage(QImage)), view, SLOT(setColorImage(QImage)));
		connect(&kinectReader, SIGNAL(depthImage(QImage)), view, SLOT(setDepthImage(QImage)));

		connect(view, SIGNAL(fileOpen(QString)), this, SLOT(fileOpen(QString)));
		connect(view, SIGNAL(fileSave(QString)), this, SLOT(fileSave(QString)));

		connect(view, SIGNAL(recordToggled(bool)), this, SLOT(record(bool)));
		connect(view, SIGNAL(captureToggled(bool)), this, SLOT(capture(bool)));
		connect(view, SIGNAL(takeShot()), this, SLOT(takeShot()));
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
	//kinectReader.getDepthData(mDepthBuffer.timespan, mDepthBuffer.width, mDepthBuffer.height, mDepthBuffer.minDistance, mDepthBuffer.maxDistance);
	kinectReader.getDepthData(mDepthBuffer.timespan, mDepthBuffer.info);
	mDepthBuffer.buffer.clear();
	kinectReader.copyDepthBuffer(mDepthBuffer.buffer, mDepthBuffer.buffer.begin());
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


void QKinectPlayerCtrl::capture(bool triggered)
{
	if (triggered)
		startKinect();
	else
		stopKinect();
}


void QKinectPlayerCtrl::takeShot()
{
	const QDateTime now = QDateTime::currentDateTime();
	const QString timestamp = now.toString(QLatin1String("yyyyMMdd-hhmmsszzz"));
	const QString filename = QDir::currentPath() + QString::fromLatin1("/KinectDepth-%1.knt").arg(timestamp);

	kinectStream.saveFrame(filename);
}



void QKinectPlayerCtrl::playStream()
{
	//kinectStream.appendFrame();
}

void QKinectPlayerCtrl::stopStream()
{
	//kinectStream.save("C:/temp/test.knt");
}


void QKinectPlayerCtrl::fileOpen(QString filename)
{
	kinectStream.load(filename);

	mDepthBuffer.buffer.clear();
	//kinectStream.copyDepthFrame(mDepthBuffer.info, mDepthBuffer.buffer, mDepthBuffer.buffer.begin(), 0);
	kinectStream.load(filename, mDepthBuffer.info, mDepthBuffer.buffer);

	std::cout << mDepthBuffer.info[0] << ", " << mDepthBuffer.info[1] << ", " << mDepthBuffer.size() << std::endl;

}


void QKinectPlayerCtrl::fileSave(QString filename)
{
	std::cout << "QKinectPlayerCtrl::fileSave: " << filename.toStdString() << std::endl;
}
