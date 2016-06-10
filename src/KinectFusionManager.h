

#ifndef __KINECT_FUSION_MANAGER_H__
#define __KINECT_FUSION_MANAGER_H__

#include <QObject>
#include <QSharedPointer>
#include "QKinectGrabberFromFile.h"
#include "QImageWidget.h"


class KinectFusionManager : public QObject
{
	Q_OBJECT

public:
	explicit KinectFusionManager(const QString& folder_path, QObject* parent = nullptr);
	~KinectFusionManager();

	//QKinectGrabberFromFile* grabber(){ return kinectGrabber.data(); }


public slots :
	void finish();


protected:
	
	//QSharedPointer<QKinectGrabberFromFile> kinectGrabber;
	//QSharedPointer<QImageWidget> depthWidget;
	//QSharedPointer<QImageWidget> colorWidget;
};

#endif // __KINECT_FUSION_MANAGER_H__

