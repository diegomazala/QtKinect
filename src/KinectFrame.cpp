
#include <fstream>
#include "KinectFrame.h"


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


void KinectFrame::load(
	const std::string& filename, 
	KinectFrame& frame)
{
	std::ifstream in_file;
	in_file.open(filename, std::ifstream::binary);
	if (in_file.is_open())
	{
		vector_read(in_file, frame.info);
		vector_read(in_file, frame.color);
		vector_read(in_file, frame.depth);
		in_file.close();
	}
	else
	{
		throw(const std::exception("could not open file"));
	}
}


void KinectFrame::load(const std::string& filename)
{
	KinectFrame::load(filename, info, color, depth);
}

void KinectFrame::load(
	const std::string& filename, 
	std::vector<unsigned short>& info,
	std::vector<unsigned char>& color,
	std::vector<unsigned short>& depth)
{
	std::ifstream in_file;
	in_file.open(filename, std::ifstream::binary);
	vector_read(in_file, info);
	vector_read(in_file, color);
	vector_read(in_file, depth);
	in_file.close();
}


void KinectFrame::loadDepth(
	const std::string& filename,
	std::vector<unsigned short>& depth)
{
	std::ifstream in_file;
	in_file.open(filename, std::ifstream::binary);
	std::size_t count_info, count_color;
	in_file.read(reinterpret_cast<char*>(&count_info), sizeof(std::size_t));
	in_file.seekg(0, count_info);
	in_file.read(reinterpret_cast<char*>(&count_color), sizeof(std::size_t));
	in_file.seekg(0, count_color);
	vector_read(in_file, depth);
	in_file.close();
}


void KinectFrame::save(
	const std::string& filename,
	const KinectFrame& frame)
{
	std::ofstream out;
	out.open(filename, std::ofstream::binary);
	vector_write(out, frame.info);
	vector_write(out, frame.color);
	vector_write(out, frame.depth);
	out.close();
}


void KinectFrame::save(
	const std::string& filename,
	const std::vector<unsigned short>& info,
	const std::vector<unsigned char>& color_buffer,
	const std::vector<unsigned short>& depth_buffer)
{
	std::ofstream out;
	out.open(filename, std::ofstream::binary);
	vector_write(out, info);
	vector_write(out, color_buffer);
	vector_write(out, depth_buffer);
	out.close();
}
