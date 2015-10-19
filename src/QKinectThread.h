

#ifndef __KINECT_THREAD_H__
#define __KINECT_THREAD_H__


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

class QImage;

class QKinectThread : public QThread
{

	static const int        cColorWidth = 1920;
	static const int        cColorHeight = 1080;
	static const int        cDepthWidth = 512;
	static const int        cDepthHeight = 424;

	Q_OBJECT

public:
	QKinectThread(QObject *parent = 0);
	~QKinectThread();

public slots:
	void stop();

signals:
	void colorImage(const QImage &image);
	void depthImage(const QImage &image);
	void infraredImage(const QImage &image);

protected:
	void run() Q_DECL_OVERRIDE;

private:

	bool initializeSensor();
	bool updateColor(QImage& colorImage);
	bool updateDepth(QImage& depthImage);
	bool updateInfrared(QImage& infraredImage);
	void uninitializeSensor();


	QMutex mutex;
	bool running;
	
	// Current Kinect
	IKinectSensor*          m_pKinectSensor;

	// Color reader
	IColorFrameReader*      m_pColorFrameReader;

	// Depth reader
	IDepthFrameReader*      m_pDepthFrameReader;

	// Infrared reader
	IInfraredFrameReader*   m_pInfraredFrameReader;

	// Direct2D
	RGBQUAD*                m_pColorRGBX;
	RGBQUAD*				m_pDepthRGBX;

};




#endif	//__KINECT_THREAD_H__
