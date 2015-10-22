#include "QKinectIO.h"
#include <QThread>
#include <QtConcurrent>
#include <iostream>
#include <fstream>

template<typename T>
static void vector_write(std::ostream& out_file, const std::vector<T>& data)
{
	const std::size_t count = data.size();
	out_file.write(reinterpret_cast<const char*>(&count), sizeof(std::size_t));
	out_file.write(reinterpret_cast<const char*>(&data[0]), count * sizeof(T));
}

template<typename T>
static void vector_read(std::istream& in_file, std::vector<T>& data)
{
	std::size_t count;
	in_file.read(reinterpret_cast<char*>(&count), sizeof(std::size_t));
	data.resize(count);
	in_file.read(reinterpret_cast<char*>(&data[0]), count * sizeof(T));
}


QKinectIO::QKinectIO():
	kinectReader(nullptr),
	maxSizeInMegaBytes(512)
{
}


QKinectIO::~QKinectIO()
{
}


void QKinectIO::setKinecReader(QKinectReader* kinect)
{
	kinectReader = kinect;
}


void QKinectIO::setMaxSizeInMegaBytes(unsigned int size_in_Mbytes)
{
	maxSizeInMegaBytes = size_in_Mbytes;
}


void QKinectIO::clear()
{
	depthBufferStream.clear();
}


unsigned int QKinectIO::size() const
{
	return depthBufferStream.size() * sizeof(unsigned short);
}


void QKinectIO::appendFrame()
{
	if (kinectReader != nullptr)
	{
		signed __int64 timespan;
		unsigned short width, height, minDistance, maxDistance;
		kinectReader->getDepthData(timespan, width, height, minDistance, maxDistance);
				
		const unsigned int frame_size_bytes = width * height * sizeof(unsigned short);	// width * height * ushort
		const unsigned int new_size_bytes = size() + frame_size_bytes;
		const unsigned int max_size_bytes = maxSizeInMegaBytes * 1024 * 1024;

		if (new_size_bytes < max_size_bytes)
			kinectReader->copyDepthBuffer(depthBufferStream, depthBufferStream.end());
	}
}


void QKinectIO::save(const QString& filename)
{
	signed __int64 timespan;
	std::vector<unsigned short> info(4);
	kinectReader->getDepthData(timespan, info[0], info[1], info[2], info[3]);

	//QFuture<void> f = 
		QtConcurrent::run(saveFile, filename, info, depthBufferStream);
	//f.waitForFinished();
}





void QKinectIO::saveFile(QString filename, std::vector<unsigned short> info, std::vector<unsigned short> depthBuffer)
{
	std::ofstream out;
	out.open(filename.toStdString(), std::ofstream::out | std::ofstream::binary);

	const unsigned int sample_size = info[0] * info[1] * sizeof(unsigned short);	// width * height * ushort
	const unsigned int total_size = depthBuffer.size() * sizeof(unsigned short);
	const unsigned int frame_count = total_size / sample_size;

	vector_write(out, info);
	vector_write(out, depthBuffer);

	out.close();
}


void QKinectIO::load(const QString& filename)
{
	std::vector<unsigned short> info;
	depthBufferStream.clear();
	loadFile(filename, info, depthBufferStream);
}


void QKinectIO::loadFile(QString filename, std::vector<unsigned short>& info, std::vector<unsigned short>& depthBuffer)
{
	std::ifstream in_file;
	in_file.open(filename.toStdString(), std::ios::in | std::ios::binary);
	vector_read(in_file, info);
	vector_read(in_file, depthBuffer);
	in_file.close();
}

