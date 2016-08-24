

#ifndef __KINECT_FUSION_MANAGER_H__
#define __KINECT_FUSION_MANAGER_H__

#include <QObject>
#include <QSharedPointer>

class QKinectGrabberFromFile;
class VolumeWidget;

class KinectFusionManager : public QObject
{
	Q_OBJECT

public:
	explicit KinectFusionManager(QKinectGrabberFromFile* kinect_grabber, QObject* parent = nullptr);
	~KinectFusionManager();


public slots :
	
	//void allocateGpu();
	//void releaseGpu();

	//void copyToGpu();
	//void copyFromGpu();

	void onNewFrame();

	


protected:
	
	QKinectGrabberFromFile* kinectGrabber;
	VolumeWidget*			volumeWidget;
};

#endif // __KINECT_FUSION_MANAGER_H__

