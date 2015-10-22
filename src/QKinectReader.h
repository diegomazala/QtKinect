

#ifndef __KINECT_READER_H__
#define __KINECT_READER_H__


#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#endif


// Windows Header Files
#include <windows.h>


// Kinect Header files
#include <Kinect.h>


#include <Shlobj.h>


#ifdef _UNICODE
#if defined _M_IX86
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
#elif defined _M_X64
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
#else
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif
#endif

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}


#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>
#include <QImage>



class QKinectReader : public QThread
{
	Q_OBJECT

public:
	QKinectReader(QObject *parent = 0);
	~QKinectReader();

	void enableImageSending(bool value);

	void getColorData(	signed __int64& timespan, 
						unsigned short& width, 
						unsigned short& height, 
						unsigned short& channels);

	void copyColorBuffer(std::vector<unsigned char>& buffer,
						 std::vector<unsigned char>::iterator position);

	void getDepthData(	signed __int64& timespan, 
						unsigned short& width, 
						unsigned short& height, 
						unsigned short& minDistance, 
						unsigned short& maxDistance);
	
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

	
	IKinectSensor*				m_pKinectSensor;		// Current Kinect

	
	IColorFrameReader*			m_pColorFrameReader;	// Color reader

	
	IDepthFrameReader*			m_pDepthFrameReader;	// Depth reader

	
	IInfraredFrameReader*		m_pInfraredFrameReader;	// Infrared reader


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




#endif	//__KINECT_READER_H__
