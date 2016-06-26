
#include "KinectFusionManager.h"
#include "QKinectGrabberFromFile.h"
#include <iostream>


KinectFusionManager::KinectFusionManager(QKinectGrabberFromFile* kinect_grabber, QObject* parent) :
	QObject(parent)
	, kinectGrabber(kinect_grabber)
{
	
}

KinectFusionManager::~KinectFusionManager()
{
}






void KinectFusionManager::onNewFrame()
{
	if (kinectGrabber)
	{
		
		KinectFrame knt;
		kinectGrabber->getDepthBuffer(knt.info, knt.depth);
		std::cout << "onFileInfo: " << knt.info.size() << ", " << knt.depth.size() << std::endl;
	}
}