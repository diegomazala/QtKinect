

#include <strsafe.h>

#include "QKinectGrabberV1.h"
#include <QtWidgets>
#include <iostream>

QKinectGrabberV1::QKinectGrabberV1(QObject *parent)
	: QThread(parent),
	m_pNuiSensor(nullptr),
	colorFrameWidth(640),
	colorFrameHeight(480),
	colorFrameChannels(4),
	depthFrameWidth(640),
	depthFrameHeight(480),
	emitImageEnabled(true),
	running(false)
{
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));

	// get resolution as DWORDS, but store as LONGs to avoid casts later
	DWORD width = 0;
	DWORD height = 0;

	NuiImageResolutionToSize(cDepthResolution, width, height);
	depthFrameWidth = static_cast<unsigned short>(width);
	depthFrameHeight = static_cast<unsigned short>(height);

	NuiImageResolutionToSize(cColorResolution, width, height);
	colorFrameWidth = static_cast<unsigned short>(width);
	colorFrameHeight = static_cast<unsigned short>(height);

	colorBuffer.resize(colorFrameWidth * colorFrameHeight * colorFrameChannels);
	depthBuffer.resize(depthFrameWidth * depthFrameHeight);
}


QKinectGrabberV1::~QKinectGrabberV1()
{
	stop();
}


void QKinectGrabberV1::enableImageSending(bool value)
{
	emitImageEnabled = value;
}


void QKinectGrabberV1::copyFrameBuffer(KinectFrame& frame)
{
	mutex.lock();
	{
		frame.clear();
		frame.info.push_back(colorFrameWidth);
		frame.info.push_back(colorFrameHeight);
		frame.info.push_back(colorFrameChannels);
		frame.info.push_back(depthFrameWidth);
		frame.info.push_back(depthFrameHeight);
		frame.info.push_back(depthMinReliableDistance);
		frame.info.push_back(depthMaxDistance);
		frame.color = colorBuffer;
		frame.depth = depthBuffer;
	}
	mutex.unlock();
}


void QKinectGrabberV1::getColorData(	signed __int64& timespan,
									unsigned short& width,
									unsigned short& height,
									unsigned short& channels)
{
	mutex.lock();
	{
		timespan	= colorFrameTime;
		width		= colorFrameWidth;
		height		= colorFrameHeight;
		channels	= colorFrameChannels;
	}
	mutex.unlock();
}



void QKinectGrabberV1::copyColorBuffer(std::vector<unsigned char>& buffer, std::vector<unsigned char>::iterator position)
{
	mutex.lock();
	{
		buffer.insert(position, colorBuffer.begin(), colorBuffer.end());
	}
	mutex.unlock();
}



void QKinectGrabberV1::getDepthData(signed __int64& timespan,
								std::vector<unsigned short>& info)
{
	info.clear();
	info.resize(4, 0);
	mutex.lock();
	{
		timespan = depthFrameTime;
		info[0] = depthFrameWidth;
		info[1] = depthFrameHeight;
		info[2] = depthMinReliableDistance;
		info[3] = depthMaxDistance;
	}
	mutex.unlock();
}

void QKinectGrabberV1::getDepthData(	signed __int64& timespan,
									unsigned short& width,
									unsigned short& height,
									unsigned short& minDistance,
									unsigned short& maxDistance)
{
	mutex.lock();
	{
		timespan	= depthFrameTime;
		width		= depthFrameWidth;
		height		= depthFrameHeight;
		minDistance = depthMinReliableDistance;
		maxDistance = depthMaxDistance;
	}
	mutex.unlock();
}


void QKinectGrabberV1::getDepthBuffer(	std::vector<unsigned short>& info,
									std::vector<unsigned short>& buffer)
{
	info.clear();
	info.resize(4, 0);
	mutex.lock();
	{
		info[0] = depthFrameWidth;
		info[1] = depthFrameHeight;
		info[2] = depthMinReliableDistance;
		info[3] = depthMaxDistance;
		buffer = depthBuffer;
	}
	mutex.unlock();
}



void QKinectGrabberV1::copyDepthBuffer(std::vector<unsigned short>& buffer, std::vector<unsigned short>::iterator position)
{
	mutex.lock();
	{
		buffer.insert(position, depthBuffer.begin(), depthBuffer.end());
	}
	mutex.unlock();
}



void QKinectGrabberV1::stop()
{
	mutex.lock();
	{
		running = false;
	}
	mutex.unlock();

	wait();

	uninitializeSensor();
}



void QKinectGrabberV1::run()
{
	if (!initializeSensor())
	{
		std::cerr << "<Error> Kinect not started" << std::endl;
		return;
	}

	running = true;


	while (running)
	{
		bool colorUpdated = false;
		bool depthUpdated = false;

		if (WAIT_OBJECT_0 == WaitForSingleObject(m_hNextDepthFrameEvent, 0))
		{
			// if we have received any valid new depth data we may need to draw
			depthUpdated = updateDepth();
		}

		if (WAIT_OBJECT_0 == WaitForSingleObject(m_hNextColorFrameEvent, 0))
		{
			// if we have received any valid new color data we may need to draw
			colorUpdated = updateColor();
		}


		if (colorUpdated || depthUpdated)
			emit frameUpdated();

		// If send image is enabled, emit signal with the color image
		if (colorUpdated && emitImageEnabled)
		{
			mutex.lock();
			{
				emit colorImage(QImage(colorBuffer.data(), colorFrameWidth, colorFrameHeight, QImage::Format_RGB32));
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

		msleep(3);
	}

}



/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
bool QKinectGrabberV1::initializeSensor()
{
	INuiSensor * pNuiSensor = NULL;
	HRESULT hr;

	int iSensorCount = 0;
	hr = NuiGetSensorCount(&iSensorCount);
	if (FAILED(hr))
	{
		std::cerr << "Kinect not found!" << std::endl;
		return false;
	}


	// Look at each Kinect sensor
	for (int i = 0; i < iSensorCount; ++i)
	{
		// Create the sensor so we can check status, if we can't create it, move on to the next
		hr = NuiCreateSensorByIndex(i, &pNuiSensor);
		if (FAILED(hr))
		{
			continue;
		}

		// Get the status of the sensor, and if connected, then we can initialize it
		hr = pNuiSensor->NuiStatus();
		if (S_OK == hr)
		{
			m_pNuiSensor = pNuiSensor;
			break;
		}

		// This sensor wasn't OK, so release it since we're not using it
		pNuiSensor->Release();
	}

	if (NULL == m_pNuiSensor)
	{
		return false;
	}

	// Initialize the Kinect and specify that we'll be using depth
	hr = m_pNuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX);
	if (FAILED(hr))
	{
		std::cerr << "Could not initialize Kinect!" << std::endl;
		return false;
	}

	// Create an event that will be signaled when depth data is available
	m_hNextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	// Open a depth image stream to receive depth frames
	hr = m_pNuiSensor->NuiImageStreamOpen(
		NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
		cDepthResolution,
		0,
		2,
		m_hNextDepthFrameEvent,
		&m_pDepthStreamHandle);
	if (FAILED(hr))
	{
		std::cerr << "Could not open depth image stream!" << std::endl;
		return false;
	}

	// Create an event that will be signaled when color data is available
	m_hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	// Open a color image stream to receive color frames
	hr = m_pNuiSensor->NuiImageStreamOpen(
		NUI_IMAGE_TYPE_COLOR,
		cColorResolution,
		0,
		2,
		m_hNextColorFrameEvent,
		&m_pColorStreamHandle);
	if (FAILED(hr))
	{
		std::cerr << "Could not open color image stream!" << std::endl;
		return false;
	}
	

	return true;
}



void QKinectGrabberV1::uninitializeSensor()
{
	// close the Kinect Sensor
	if (m_pNuiSensor)
	{
		m_pNuiSensor->Release();
		m_pNuiSensor = nullptr;
	}
}



/// <summary>
/// Get color frame from kinect
/// </summary>
bool QKinectGrabberV1::updateColor()
{
	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pColorStreamHandle, 0, &imageFrame);
	if (FAILED(hr)) { return false; }

	NUI_LOCKED_RECT LockedRect;
	hr = imageFrame.pFrameTexture->LockRect(0, &LockedRect, NULL, 0);
	if (FAILED(hr)) { return false; }
		
	mutex.lock();
	{
		memcpy(colorBuffer.data(), LockedRect.pBits, LockedRect.size);
	//	UINT bufferSize = sizeof(byte) * colorFrameWidth * colorFrameHeight * colorFrameChannels;
	//	std::copy(reinterpret_cast<unsigned char*>(LockedRect.pBits), LockedRect.pBits + bufferSize, colorBuffer.begin());
	}
	mutex.unlock();

	hr = imageFrame.pFrameTexture->UnlockRect(0);
	if ( FAILED(hr) ) { return false; };

	hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pColorStreamHandle, &imageFrame);
	if ( FAILED(hr) ) { return false; };

	return true;
}


/// <summary>
/// Get depth frame from kinect
/// </summary>
bool QKinectGrabberV1::updateDepth()
{
	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pDepthStreamHandle, 0, &imageFrame);
	if (FAILED(hr)) { return false; }

	NUI_LOCKED_RECT LockedRect;
	hr = imageFrame.pFrameTexture->LockRect(0, &LockedRect, NULL, 0);
	if (FAILED(hr)) { return false; }

	mutex.lock();
	{
		memcpy(depthBuffer.data(), LockedRect.pBits, LockedRect.size);
	}
	mutex.unlock();
	

	hr = imageFrame.pFrameTexture->UnlockRect(0);
	if (FAILED(hr)) { return false; };

	hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pDepthStreamHandle, &imageFrame);
	if (FAILED(hr)) { return false; };

	return true;
}


/// <summary>
/// Get infrared frame from kinect
/// </summary>
bool QKinectGrabberV1::updateInfrared(QImage& infraredImg)
{
	return false;
}

