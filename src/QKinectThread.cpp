

#include <strsafe.h>
#include <iostream>
#include "QKinectThread.h"

#include <QtWidgets>
#include <cmath>

QKinectThread::QKinectThread(QObject *parent)
	: QThread(parent),
	running(false),
	m_pKinectSensor(NULL),
	m_pColorFrameReader(NULL),
	m_pColorRGBX(NULL)
{
}


QKinectThread::~QKinectThread()
{
	stop();
}


void QKinectThread::stop()
{
	mutex.lock();
	running = false;
	mutex.unlock();

	wait();

	uninitializeSensor();
}



void QKinectThread::run()
{
	if (!initializeSensor())
	{
		std::cerr << "<Error> Kinect not started" << std::endl;
		return;
	}

	running = true;

	while (running)
	{
		//mutex.lock();
		//mutex.unlock();

		//QImage colorImg;
		//if (updateColor(colorImg))
		//{
		//	emit colorImage(colorImg);
		//}

		QImage depthImg;
		if (updateDepth(depthImg))
		{
			emit depthImage(depthImg);
		}

		//QImage infraredImg;
		//if (updateInfrared(infraredImg))
		//{
		//	emit infraredImage(infraredImg);
		//}

		msleep(10);
	}
}



/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
bool QKinectThread::initializeSensor()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return false;
	}

	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get the color reader
		IColorFrameSource* pColorFrameSource = NULL;
		IDepthFrameSource* pDepthFrameSource = NULL;
		IInfraredFrameSource* pInfraredFrameSource = NULL;

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		}
		
		// DepthFrame
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
		}

		// InfraredFrame
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_InfraredFrameSource(&pInfraredFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pInfraredFrameSource->OpenReader(&m_pInfraredFrameReader);
		}

		SafeRelease(pColorFrameSource);
		SafeRelease(pDepthFrameSource);
		SafeRelease(pInfraredFrameSource);
	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		std::cerr << "No ready Kinect found!" << std::endl;
		return false;
	}

	// create heap storage for color pixel data in RGBX format
	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}
	m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];


	// create heap storage for depth pixel data in RGBX format
	if (m_pDepthRGBX)
	{
		delete[] m_pDepthRGBX;
		m_pDepthRGBX = NULL;
	}
	m_pDepthRGBX = new RGBQUAD[cDepthWidth * cDepthHeight];


	return true;
}



void QKinectThread::uninitializeSensor()
{
	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}

	if (m_pDepthRGBX)
	{
		delete[] m_pDepthRGBX;
		m_pDepthRGBX = NULL;
	}

	// done with color frame reader
	SafeRelease(m_pColorFrameReader);

	// close the Kinect Sensor
	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}

	SafeRelease(m_pKinectSensor);
}



/// <summary>
/// Get color frame from kinect
/// </summary>
bool QKinectThread::updateColor(QImage& colorImg)
{
	if (!m_pColorFrameReader)
	{
		return false;
	}

	IColorFrame* pColorFrame = NULL;

	HRESULT hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int frameWidth = 0;
		int frameHeight = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		

		hr = pColorFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Width(&frameWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Height(&frameHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		if (SUCCEEDED(hr))
		{
			colorImg = QImage(frameWidth, frameHeight, QImage::Format_ARGB32);

			if (imageFormat == ColorImageFormat_Bgra)
			{
				UINT bufferSize;
				hr = pColorFrame->AccessRawUnderlyingBuffer(&bufferSize, reinterpret_cast<BYTE**>(colorImg.bits()));
			}
			else if (m_pColorRGBX)
			{
				hr = pColorFrame->CopyConvertedFrameDataToArray(colorImg.byteCount(), reinterpret_cast<BYTE*>(colorImg.bits()), ColorImageFormat_Bgra);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		//if (SUCCEEDED(hr))
		//{
		//	setPixmap(QPixmap::fromImage(imageColor).scaled(width(), height(), Qt::KeepAspectRatio));
		//}

		SafeRelease(pFrameDescription);
	}

	SafeRelease(pColorFrame);

	if (!SUCCEEDED(hr))
		return false;

	return true;
}


/// <summary>
/// Get depth frame from kinect
/// </summary>
bool QKinectThread::updateDepth(QImage& depthImg)
{
	if (!m_pDepthFrameReader)
	{
		return false;
	}

	IDepthFrame* pDepthFrame = NULL;

	HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		USHORT nDepthMinReliableDistance = 0;
		USHORT nDepthMaxDistance = 0;
		UINT nBufferSize = 0;
		UINT16 *pBuffer = NULL;
		
		hr = pDepthFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Width(&nWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Height(&nHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
		}

		if (SUCCEEDED(hr))
		{
			// In order to see the full range of depth (including the less reliable far field depth)
			// we are setting nDepthMaxDistance to the extreme potential depth threshold
			nDepthMaxDistance = USHRT_MAX;

			// Note:  If you wish to filter by reliable depth distance, uncomment the following line.
			hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
		}

		

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
		}

		if (SUCCEEDED(hr))
		{
			std::vector<uchar> depthBuffer;
			QVector<QRgb>  colorTable;
			for (int i = 0; i < 256; ++i)
				colorTable.push_back(qRgb(i, i, i));

			depthImg = QImage(nWidth, nHeight, QImage::Format::Format_Indexed8);
			depthImg.setColorTable(colorTable);

			int i = 0;
			for (int y = 0; y < depthImg.height(); ++y)
			{
				for (int x = 0; x < depthImg.width(); ++x)
				{
					//USHORT depth = pBuffer[i];
					//int intensity = static_cast<int>((depth >= nDepthMinReliableDistance) && (depth <= nDepthMaxDistance) ? (depth % 256) : 0);

					float depth = pBuffer[i];
					uchar intensity = static_cast<uchar>(depth / nDepthMaxDistance * 255.0f);
					depthImg.setPixel(x, y, qGray(intensity, intensity, intensity));

#if 0
					//int intensity = int(byte(pBuffer[i] & 0xFF00));// heigh byte
					int intensity = int(byte(pBuffer[i] & 0x00FF));	// low byte
					depthImg.setPixel(x, y, qGray(intensity, intensity, intensity));
#endif
					
					++i;
				}
			}
		}
		SafeRelease(pFrameDescription);
	}
	SafeRelease(pDepthFrame);

	if (!SUCCEEDED(hr))
		return false;

	return true;
}


/// <summary>
/// Get infrared frame from kinect
/// </summary>
bool QKinectThread::updateInfrared(QImage& infraredImg)
{
	if (!m_pInfraredFrameReader)
	{
		return false;
	}

	IInfraredFrame* pInfraredFrame = NULL;

	HRESULT hr = m_pInfraredFrameReader->AcquireLatestFrame(&pInfraredFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		UINT nBufferSize = 0;
		UINT16 *pBuffer = NULL;

		hr = pInfraredFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr))
		{
			hr = pInfraredFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Width(&nWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Height(&nHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pInfraredFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
		}

		if (SUCCEEDED(hr))
		{
			std::vector<uchar> infraredBuffer;
			QVector<QRgb>  colorTable;
			for (int i = 0; i < 256; ++i)
				colorTable.push_back(qRgb(i, i, i));

			infraredImg = QImage(nWidth, nHeight, QImage::Format::Format_Indexed8);
			infraredImg.setColorTable(colorTable);

			int i = 0;
			for (int y = 0; y < infraredImg.height(); ++y)
			{
				for (int x = 0; x < infraredImg.width(); ++x)
				{
					USHORT depth = pBuffer[i];
					int intensity = depth % 256;
					infraredImg.setPixel(x, y, qGray(intensity, intensity, intensity));
					++i;
				}
			}
		}

		SafeRelease(pFrameDescription);
	}

	SafeRelease(pInfraredFrame);

	if (!SUCCEEDED(hr))
		return false;

	return true;
}

