

#ifndef __KINECT_GRABBER_V1_H__
#define __KINECT_GRABBER_V1_H__


//#ifndef WIN32_LEAN_AND_MEAN
//#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
//#endif


// Windows Header Files
#include <windows.h>


// Kinect Header files
#include "NuiApi.h"



//// Safe release for interfaces
//template<class Interface>
//inline void SafeRelease(Interface *& pInterfaceToRelease)
//{
//	if (pInterfaceToRelease != NULL)
//	{
//		pInterfaceToRelease->Release();
//		pInterfaceToRelease = NULL;
//	}
//}


#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>
#include <QImage>
#include "KinectFrame.h"

class QKinectGrabberV1 : public QThread
{
	Q_OBJECT

	static const NUI_IMAGE_RESOLUTION   cDepthResolution = NUI_IMAGE_RESOLUTION_640x480;
	static const NUI_IMAGE_RESOLUTION   cColorResolution = NUI_IMAGE_RESOLUTION_640x480;

public:
	QKinectGrabberV1(QObject *parent = 0);
	~QKinectGrabberV1();

	void enableImageSending(bool value);

	void copyFrameBuffer(KinectFrame& frame);

	void getColorData(	signed __int64& timespan, 
						unsigned short& width, 
						unsigned short& height, 
						unsigned short& channels);

	void copyColorBuffer(std::vector<unsigned char>& buffer,
						 std::vector<unsigned char>::iterator position);


	void getDepthData(	signed __int64& timespan,
						std::vector<unsigned short>& info);

	void getDepthData(	signed __int64& timespan, 
						unsigned short& width, 
						unsigned short& height, 
						unsigned short& minDistance, 
						unsigned short& maxDistance);

	void getDepthBuffer(std::vector<unsigned short>& info,
						std::vector<unsigned short>& buffer);

	
	void copyDepthBuffer(std::vector<unsigned short>& buffer,
						 std::vector<unsigned short>::iterator position);


public slots:
	void stop();

signals:
	void colorImage(const QImage &image);
	void depthImage(const QImage &image);
	void infraredImage(const QImage &image);
	void frameUpdated();

protected:
	void run() Q_DECL_OVERRIDE;

private:

	bool initializeSensor();
	bool updateColor();
	bool updateDepth();
	bool updateInfrared(QImage& infraredImage);
	void uninitializeSensor();


	INuiSensor*                 m_pNuiSensor;			// Current Kinect
	HANDLE                      m_hNextDepthFrameEvent;
	HANDLE                      m_pDepthStreamHandle;
	HANDLE                      m_hNextColorFrameEvent;
	HANDLE                      m_pColorStreamHandle;

	unsigned short*             m_depthD16;
	unsigned char*              m_colorRGBX;
	

	unsigned short				colorFrameWidth;		// = 1920;
	unsigned short				colorFrameHeight;		// = 1080;
	const unsigned short		colorFrameChannels;		// = 4;
	signed __int64				colorFrameTime;			// timestamp

	unsigned short				depthFrameWidth;		// = 512;
	unsigned short				depthFrameHeight;		// = 424;
	signed __int64				depthFrameTime;			// timestamp

	unsigned short				depthMinReliableDistance;
	unsigned short				depthMaxDistance;

	std::vector<unsigned char>	colorBuffer;
	std::vector<unsigned short>	depthBuffer;

	bool						emitImageEnabled;
	QVector<QRgb>				colorTable;

	QMutex						mutex;
	bool						running;
};




#endif	//__KINECT_GRABBER_V1_H__
