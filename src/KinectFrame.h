

#ifndef __KINECT_FRAME_H__
#define __KINECT_FRAME_H__


#include <vector>


class KinectFrame
{
public:

	static const unsigned short BytesPerPixel = 4;

	// info : [0] color_width
	// info : [1] color_height
	// info : [2] color_channels
	// info : [3] depth_width
	// info : [4] depth_width
	// info : [5] depth_min_distance
	// info : [6] depth_max_distance
	enum
	{
		ColorWidth = 0, 
		ColorHeight,
		ColorChannels,
		DepthWidth,
		DepthHeight,
		DepthMinDistance,
		DepthMaxDistance
	};
	
	KinectFrame()
	{
		reset();
	}
	KinectFrame(
		const std::vector<unsigned short>& _info,
		const std::vector<unsigned char>& _color,
		const std::vector<unsigned short>& _depth) :
		info(_info), color(_color), depth(_depth){}

	KinectFrame(const std::string& filename)
	{
		load(filename);
	}

	void clear()
	{
		info.clear();
		color.clear();
		depth.clear();
	}

	void reset()
	{
		clear();
		info.resize(7, 0);
	}

	unsigned short color_width() const { return info[ColorWidth]; }
	unsigned short color_height() const { return info[ColorHeight]; }
	unsigned short depth_width() const { return info[DepthWidth]; }
	unsigned short depth_height() const { return info[DepthHeight]; }
	unsigned short depth_min_distance() const { return info[DepthMinDistance]; }
	unsigned short depth_max_distance() const { return info[DepthMaxDistance]; }

	void load(const std::string& filename);

	static void KinectFrame::load(
		const std::string& filename,
		KinectFrame& frame);


	static void KinectFrame::load(
		const std::string& filename,
		std::vector<unsigned short>& info,
		std::vector<unsigned char>& color,
		std::vector<unsigned short>& depth);


	static void KinectFrame::loadDepth(
		const std::string& filename,
		std::vector<unsigned short>& depth);


	static void KinectFrame::save(
		const std::string& filename,
		const KinectFrame& frame);


	static void KinectFrame::save(
		const std::string& filename,
		const std::vector<unsigned short>& info,
		const std::vector<unsigned char>& color_buffer,
		const std::vector<unsigned short>& depth_buffer);


	KinectFrame& KinectFrame::operator=(KinectFrame other) // copy/move constructor is called to construct arg
	{
		// resources are exchanged between *this and other
		info.swap(other.info); 
		color.swap(other.color);
		depth.swap(other.depth);
		return *this;
	} // destructor of other is called to release the resources formerly held by *this

	KinectFrame& operator=(const KinectFrame& other) // copy assignment
	{
		info = other.info;
		color = other.color;
		depth = other.depth;
		return *this;
	}

	std::vector<unsigned short> info;
	std::vector<unsigned char> color;
	std::vector<unsigned short> depth;
			
};

#endif	//__KINECT_FRAME_H__
