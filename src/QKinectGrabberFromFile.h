

#ifndef __KINECT_GRABBER_FROM_FILE_H__
#define __KINECT_GRABBER_FROM_FILE_H__


// Windows Header Files
#include <windows.h>


// Kinect Header files
#include "NuiApi.h"



#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>
#include <QImage>
#include "QKinectFrame.h"

class QKinectGrabberFromFile : public QThread
{
	Q_OBJECT

	static const NUI_IMAGE_RESOLUTION   cDepthResolution = NUI_IMAGE_RESOLUTION_640x480;
	static const NUI_IMAGE_RESOLUTION   cColorResolution = NUI_IMAGE_RESOLUTION_640x480;

public:
	QKinectGrabberFromFile(QObject *parent = 0);
	~QKinectGrabberFromFile();

	void enableImageSending(bool value);

	void copyFrameBuffer(KinectFrame& frame);


	void copyColorBuffer(std::vector<unsigned char>& buffer,
						 std::vector<unsigned char>::iterator position);


	void getDepthBuffer(std::vector<unsigned short>& info,
						std::vector<unsigned short>& buffer);

	
	void copyDepthBuffer(std::vector<unsigned short>& buffer,
						 std::vector<unsigned short>::iterator position);




public slots:
	void stop();
	void setFolder(const QString& input_folder);
	void setFramesPerSecond(int fps = 30);

signals:
	void colorImage(const QImage &image);
	void depthImage(const QImage &image);
	void infraredImage(const QImage &image);
	void frameUpdated();

protected:
	
	int collectFilesInFolder();
	void run() Q_DECL_OVERRIDE;

private:

	QKinectFrame				kinectFrame;

	bool						emitImageEnabled;
	

	QMutex						mutex;
	bool						running;

	QString						folder;
	QStringList					frameFiles;
	int							framesPerSecond;
};




#endif	//__KINECT_GRABBER_FROM_FILE_H__
