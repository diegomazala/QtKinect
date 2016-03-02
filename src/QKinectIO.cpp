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


void QKinectIO::setKinecReader(QKinectGrabber* kinect)
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


void QKinectIO::copyDepthFrame(	std::vector<unsigned short>& info, 
								std::vector<unsigned short>& buffer, 
								std::vector<unsigned short>::iterator position, 
								unsigned int frame_index)
{
	info.clear();
	info.resize(4);
		
	signed __int64 timespan;
	kinectReader->getDepthData(timespan, info[0], info[1], info[2], info[3]);

	const unsigned int frame_size = info[0] * info[1];	// width * height
	std::vector<unsigned short>::iterator begin = depthBufferStream.begin() + frame_index * frame_size;
	std::vector<unsigned short>::iterator end = begin + frame_size;

	buffer.insert(position, begin, end);
}


void QKinectIO::appendFrame()
{
	if (kinectReader != nullptr && kinectReader->isRunning())
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


void QKinectIO::saveFrame(const QString& filename)
{
	if (kinectReader == nullptr || !kinectReader->isRunning())	// kinect is not running
		return;
	
	std::vector<unsigned short> buffer;
	signed __int64 timespan;
	std::vector<unsigned short> info(4);
	kinectReader->getDepthData(timespan, info[0], info[1], info[2], info[3]);
	kinectReader->copyDepthBuffer(buffer, buffer.begin());
	
	QtConcurrent::run(save, filename, info, buffer);
}


void QKinectIO::save(const QString& filename)
{
	if (depthBufferStream.size() < 1)
		return;

	signed __int64 timespan;
	std::vector<unsigned short> info(4);
	kinectReader->getDepthData(timespan, info[0], info[1], info[2], info[3]);

	//QFuture<void> f = 
	QtConcurrent::run(save, filename, info, depthBufferStream);
	//f.waitForFinished();
}




void QKinectIO::save(QString filename, std::vector<unsigned short> info, std::vector<unsigned short> depthBuffer)
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
	load(filename, info, depthBufferStream);
}



void QKinectIO::load(QString filename, std::vector<unsigned short>& info, std::vector<unsigned short>& depthBuffer)
{
	std::ifstream in_file;
	in_file.open(filename.toStdString(), std::ios::in | std::ios::binary);
	vector_read(in_file, info);
	vector_read(in_file, depthBuffer);
	in_file.close();
}


void QKinectIO::exportObj(QString filename, std::vector<unsigned short> info, std::vector<unsigned short> depthBuffer)
{
	std::ofstream out;
	out.open(filename.toStdString(), std::ofstream::out);

	const unsigned int sample_size = info[0] * info[1] * sizeof(unsigned short);	// width * height * ushort
	const unsigned int total_size = depthBuffer.size() * sizeof(unsigned short);
	const unsigned int frame_count = total_size / sample_size;

	out << "# w " << info[0] << " h " << info[1] << " minDist " << info[2] << " maxDist " << info[3] << std::endl;

	for (int i = 0; i < depthBuffer.size(); ++i)
	{
		const unsigned short z = depthBuffer[i];

		if (z < info[2] || z > info[3])
			continue;

		const unsigned short x = i % info[0];
		const unsigned short y = i / info[0];

		const float fx = (static_cast<float>(x) - static_cast<float>(info[0]) * 0.5f) / static_cast<float>(info[0]);
		const float fy = (static_cast<float>(y) - static_cast<float>(info[1]) * 0.5f) / static_cast<float>(info[1]);
		//const float fz = static_cast<float>(z) / static_cast<float>(info[3] - );
		const float fz = static_cast<float>(z - info[2]) / static_cast<float>(info[3] - info[2]);
		
		out << "v " << fx << ' ' << fy << ' ' << fz << std::endl;
	}

	out.close();
}



void QKinectIO::saveFrame(const std::string& filename, const KinectFrame& frame)
{
	QtConcurrent::run(save, QString::fromStdString(filename), frame.info, frame.color, frame.depth);
}


void QKinectIO::save(QString filename, 
					std::vector<unsigned short> info, 
					std::vector<unsigned char> color_buffer, 
					std::vector<unsigned short> depth_buffer)
{
	std::ofstream out;
	out.open(filename.toStdString(), std::ofstream::binary);

	vector_write(out, info);
	vector_write(out, color_buffer);
	vector_write(out, depth_buffer);

	out.close();
}



void QKinectIO::loadFrame(const std::string& filename, KinectFrame& frame)
{
	std::ifstream in_file;
	in_file.open(filename, std::ifstream::binary);
	vector_read(in_file, frame.info);
	vector_read(in_file, frame.color);
	vector_read(in_file, frame.depth);
	in_file.close();
}