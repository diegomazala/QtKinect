

#include <strsafe.h>

#include "QKinectGrabberFromFile.h"
#include <QtWidgets>
#include <iostream>

QKinectGrabberFromFile::QKinectGrabberFromFile(QObject *parent)
	: QThread(parent),
	emitImageEnabled(true),
	running(false)
{
	
}


QKinectGrabberFromFile::~QKinectGrabberFromFile()
{
	stop();
}

void QKinectGrabberFromFile::setFramesPerSecond(int fps)
{
	framesPerSecond = fps;
}

void QKinectGrabberFromFile::setFolder(const QString& input_folder)
{
	folder = input_folder;
	collectFilesInFolder();
}

void QKinectGrabberFromFile::enableImageSending(bool value)
{
	emitImageEnabled = value;
}


void QKinectGrabberFromFile::copyFrameBuffer(KinectFrame& frame)
{
	mutex.lock();
	{
		frame.clear();
		frame.info = kinectFrame.info;
		frame.color = kinectFrame.color;
		frame.depth = kinectFrame.depth;
	}
	mutex.unlock();
}



void QKinectGrabberFromFile::copyColorBuffer(
	std::vector<unsigned char>& buffer, 
	std::vector<unsigned char>::iterator position)
{
	mutex.lock();
	{
		buffer.insert(position, kinectFrame.color.begin(), kinectFrame.color.end());
	}
	mutex.unlock();
}




void QKinectGrabberFromFile::getDepthBuffer(	
	std::vector<unsigned short>& info,
	std::vector<unsigned short>& buffer)
{
	info.clear();
	info.resize(4, 0);
	mutex.lock();
	{
		info[0] = kinectFrame.depth_width();
		info[1] = kinectFrame.depth_height();
		info[2] = kinectFrame.depth_min_distance();
		info[3] = kinectFrame.depth_max_distance();
		buffer = kinectFrame.depth;
	}
	mutex.unlock();
}





void QKinectGrabberFromFile::copyDepthBuffer(
	std::vector<unsigned short>& buffer, 
	std::vector<unsigned short>::iterator position)
{
	mutex.lock();
	{
		buffer.insert(position, kinectFrame.depth.begin(), kinectFrame.depth.end());
	}
	mutex.unlock();
}


void QKinectGrabberFromFile::getKinectFrame(KinectFrame& frame)
{
	mutex.lock();
	{
		frame.info = kinectFrame.info;
		frame.color = kinectFrame.color;
		frame.depth = kinectFrame.depth;
	}
	mutex.unlock();
}

void QKinectGrabberFromFile::stop()
{
	mutex.lock();
	{
		running = false;
	}
	mutex.unlock();

	wait();
}

int QKinectGrabberFromFile::collectFilesInFolder()
{
	QStringList filters;
	filters << "*.knt";
	QDir dir(folder);
	dir.setNameFilters(filters);
	frameFiles = dir.entryList();

	return frameFiles.size();
}

void QKinectGrabberFromFile::run()
{
	// color table for depth images
	QVector<QRgb> colorTable;
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));


	if (frameFiles.size() < 1)
	{
		std::cerr << "<Error> Could not find any kinect frame file in folder '" << folder.toStdString() << "'" << std::endl;
		return;
	}

	running = true;

	QStringListIterator frameFile(frameFiles);

	while (running)
	{
		const QString frame_file_name = folder + "/" + frameFile.next();
		QKinectFrame kinect_frame(frame_file_name);
		
		emit fileLoaded(frameFile.next());

		mutex.lock();
		{
			kinectFrame.info = kinect_frame.info;
			kinectFrame.color = kinect_frame.color;
			kinectFrame.depth = kinect_frame.depth;
		}
		mutex.unlock();
		
		emit frameUpdated();

		// If send image is enabled, emit signal with the color image
		if (emitImageEnabled)
		{
			mutex.lock();
			{
				emit colorImage(kinectFrame.convertColorToImage());
			}
			mutex.unlock();
		}


		// If send image is enabled, emit signal with the depth image
		if (emitImageEnabled)
		{
			mutex.lock();
			{
				emit depthImage(kinectFrame.convertDepthToImage());
			}
			mutex.unlock();
		}


		if (!frameFile.hasNext())
		{
			mutex.lock();
			{
				running = false;
			}
			mutex.unlock();
		}

		msleep(1000 / framesPerSecond);
	}


}






