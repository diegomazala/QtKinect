#ifndef __Q_KINECT_GPU_H__
#define __Q_KINECT_GPU_H__

#include <QObject>
#include "QKinectGrabberFromFile.h"
#include "KinectCuda.h"
#include "GLPointCloudViewer.h"

class QKinectGpu : public QObject
{
	Q_OBJECT

public:
	explicit QKinectGpu(QObject* parent = nullptr);
	~QKinectGpu();



public slots :
	void setKinect(QKinectGrabberFromFile* kinect_ptr);
	void setPointCloudViewer(GLPointCloudViewer* viewer_ptr);
	void setPointCloud(GLPointCloud* model_ptr);
	void onFrameUpdate();


signals:
	void kernelExecuted();


public://protected:
	
	QKinectGrabberFromFile*			kinect;
	KinectCuda						kinectCuda;
	GLPointCloudViewer*				viewer;
	GLPointCloud*					pointCloud;
	std::shared_ptr<GLPointCloud>	cloud;
};

#endif // __Q_KINECT_GPU_H__

