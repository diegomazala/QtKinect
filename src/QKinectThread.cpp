

#include <strsafe.h>
#include <iostream>
#include "QKinectThread.h"

#include <QtWidgets>
#include <cmath>

#include <time.h>

QKinectThread::QKinectThread(QObject *parent)
	: QThread(parent),
	m_pKinectSensor(NULL),
	m_pColorFrameReader(NULL),
	colorFrameWidth(1920),
	colorFrameHeight(1080),
	colorFrameChannels(4),
	depthFrameWidth(512),
	depthFrameHeight(424),
	colorBuffer(colorFrameWidth * colorFrameHeight * colorFrameChannels),
	depthBuffer(depthFrameWidth * depthFrameHeight),
	emitImageEnabled(true),
	running(false)
{
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));
}


QKinectThread::~QKinectThread()
{
	stop();
}


void QKinectThread::enableImageSending(bool value)
{
	emitImageEnabled = value;
}


void QKinectThread::getColorData(std::vector<unsigned char>& buffer, signed __int64& timespan)
{
	mutex.lock();
	{
		buffer = colorBuffer;
		timespan = colorFrameTime;
	}
	mutex.unlock();
}


void QKinectThread::getDepthData(std::vector<unsigned short>& buffer, signed __int64& timespan)
{
	mutex.lock();
	{
		buffer = depthBuffer;
		timespan = depthFrameTime;
	}
	mutex.unlock();
}


void QKinectThread::stop()
{
	mutex.lock();
	{
		running = false;
	}
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
		bool colorUpdated = updateColor();
		bool depthUpdated = updateDepth();

		// If send image is enabled, emit signal with the color image
		if (colorUpdated && emitImageEnabled)
		{
			mutex.lock();
			{
				emit colorImage(QImage(colorBuffer.data(), colorFrameWidth, colorFrameHeight, QImage::Format_ARGB32));
			}
			mutex.unlock();
		}

		// If send image is enabled, emit signal with the depth image
		if (depthUpdated && emitImageEnabled)
		{
			mutex.lock();
			{
				// create depth image
				QImage depthImg = QImage(depthFrameWidth, depthFrameHeight, QImage::Format::Format_Indexed8);
				depthImg.setColorTable(colorTable);

				std::vector<unsigned char> depthImgBuffer(depthBuffer.size());

				// casting from unsigned short (2 bytes precision) to unsigned char (1 byte precision)
				std::transform(
					depthBuffer.begin(),
					depthBuffer.end(),
					depthImgBuffer.begin(),
					[=](const unsigned short depth) { return static_cast<unsigned char>((float)depth / (float)depthMaxDistance * 255.f); });

				// set pixels to depth image
				for (int y = 0; y < depthImg.height(); y++)
					memcpy(depthImg.scanLine(y), depthImgBuffer.data() + y * depthImg.width(), depthImg.width());

				emit depthImage(depthImg);
			}
			mutex.unlock();
		}

		//QImage infraredImg;
		//if (updateInfrared(infraredImg))
		//{
		//	emit infraredImage(infraredImg);
		//}

		msleep(3);
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

	return true;
}



void QKinectThread::uninitializeSensor()
{

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
bool QKinectThread::updateColor()
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
		BYTE *pBuffer = NULL;

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
			if (imageFormat == ColorImageFormat_Bgra)
			{
				UINT bufferSize;
				hr = pColorFrame->AccessRawUnderlyingBuffer(&bufferSize, reinterpret_cast<BYTE**>(pBuffer));

				// copy data to color buffer
				if (SUCCEEDED(hr))
				{
					mutex.lock();
					{
						std::copy(reinterpret_cast<unsigned char*>(pBuffer), pBuffer + bufferSize, colorBuffer.begin());
						colorFrameTime = nTime;

						if (colorFrameWidth != frameWidth || colorFrameHeight != frameHeight)
						{
							std::cerr << "<Warning>	Unexpected size for depth buffer" << std::endl;
							colorFrameWidth = frameWidth;
							colorFrameHeight = frameHeight;
						}
					}
					mutex.unlock();
				}
			}
			else 
			{
				mutex.lock();
				{
					hr = pColorFrame->CopyConvertedFrameDataToArray(colorBuffer.size(), reinterpret_cast<BYTE*>(colorBuffer.data()), ColorImageFormat_Bgra);
					if (SUCCEEDED(hr))
					{
						colorFrameTime = nTime;
						if (colorFrameWidth != frameWidth || colorFrameHeight != frameHeight)
						{
							std::cerr << "<Warning>	Unexpected size for depth buffer" << std::endl;
							colorFrameWidth = frameWidth;
							colorFrameHeight = frameHeight;
						}
					}
					else
					{
						std::cerr << "<Error>	Could not convert data from color frame to color buffer" << std::endl;
					}
				}
				mutex.unlock();
			}
		}

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
bool QKinectThread::updateDepth()
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
		int frameWidth = 0;
		int frameHeight = 0;
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
			hr = pFrameDescription->get_Width(&frameWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Height(&frameHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
		}

		if (SUCCEEDED(hr))
		{
			// In order to see the full range of depth (including the less reliable far field depth)
			// we are setting nDepthMaxDistance to the extreme potential depth threshold
			//nDepthMaxDistance = USHRT_MAX;

			// Note:  If you wish to filter by reliable depth distance, uncomment the following line.
			hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
		}
		

		if (SUCCEEDED(hr))
		{			
			mutex.lock();
			{
				// copy data to depth buffer
				std::copy(reinterpret_cast<unsigned short*>(pBuffer), pBuffer + nBufferSize, depthBuffer.begin());
				depthMinReliableDistance = nDepthMinReliableDistance;
				depthMaxDistance = nDepthMaxDistance;
				depthFrameTime = nTime;

				if (depthFrameWidth != frameWidth || depthFrameHeight != frameHeight)
				{
					std::cerr << "<Warning>	Unexpected size for depth buffer" << std::endl;
					depthFrameWidth = frameWidth;
					depthFrameHeight = frameHeight;
				}
			}
			mutex.unlock();
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

