#include "QKinectPlayerCtrl.h"
#include "MainWindow.h"
#include "GLDepthBufferRenderer.h"
#include <QFileDialog>
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
	//depthRenderer = reinterpret_cast<GLDepthBufferRenderer*>(glDepthRenderer);
	depthRenderer->setController(*this);
}


void QKinectPlayerCtrl::setupConnections()
{
	if (view != nullptr)
	{
		connect(&kinectReader, SIGNAL(colorImage(QImage)), view, SLOT(setColorImage(QImage)));
		connect(&kinectReader, SIGNAL(depthImage(QImage)), view, SLOT(setDepthImage(QImage)));

		connect(view, SIGNAL(recordToggled(bool)), this, SLOT(record(bool)));
		connect(view, SIGNAL(captureToggled(bool)), this, SLOT(capture(bool)));
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
	kinectReader.getDepthData(mDepthBuffer.timespan, mDepthBuffer.width, mDepthBuffer.height, mDepthBuffer.minDistance, mDepthBuffer.maxDistance);
	mDepthBuffer.buffer.clear();
	kinectReader.copyDepthBuffer(mDepthBuffer.buffer, mDepthBuffer.buffer.begin());

//	if (depthRenderer != nullptr)
//		depthRenderer->setDepthBuffer(mDepthBuffer.buffer, mDepthBuffer.width, mDepthBuffer.height);


#if 0
	if (isRecording())
		kinectStream.appendFrame();

	if (depthRenderer != nullptr)
	{
		signed __int64 timespan;
		unsigned short width, height, minDistance, maxDistance;
		kinectReader.getDepthData(timespan, width, height, minDistance, maxDistance);

		std::vector<unsigned short> depthBuffer;
		kinectReader.copyDepthBuffer(depthRenderer->getDepthBufferCloud(), depthRenderer->getDepthBufferCloud().begin());
		//std::cout << depthBuffer.size() << std::endl;
		//depthRenderer->setDepthBuffer(depthBuffer, width, height);
	}
#endif
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


void QKinectPlayerCtrl::playStream()
{
	//kinectStream.appendFrame();
}

void QKinectPlayerCtrl::stopStream()
{
	//kinectStream.save("C:/temp/test.knt");
}

