

#ifndef __Q_KINECT_FILE_H__
#define __Q_KINECT_FILE_H__

#include <QObject>
#include "QKinectGrabberV1.h"
#include "QKinectIO.h"
#include <iostream>


class QKinectFile : public QObject
{
	Q_OBJECT

public:
	explicit QKinectFile(QObject* parent = nullptr);
	~QKinectFile();


public slots :
	void save();
	void setFolder(QString folder_path);
	void setKinect(QKinectGrabberV1* kinect_ptr);

protected:
	
	QKinectGrabberV1*	kinect;
	QString				folderPath;

};

#endif // __Q_KINECT_FILE_H__

