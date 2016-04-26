

#ifndef __KINECT_SPECS_H__
#define __KINECT_SPECS_H__

#ifndef ushort
typedef unsigned short ushort;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif


const float		KINECT_V1_FOVY = 43.f;
const float		KINECT_V1_FOVX = 57.f;
const float		KINECT_V1_ASPECT_RATIO = 1.33f;	// 640 / 480
const ushort	KINECT_V1_COLOR_WIDTH = 640;
const ushort	KINECT_V1_COLOR_HEIGHT = 480;
const ushort	KINECT_V1_DEPTH_WIDTH = 320;
const ushort	KINECT_V1_DEPTH_HEIGHT = 240;
const ushort	KINECT_V1_DEPTH_MIN = 40;	// 40 cm
const ushort	KINECT_V1_DEPTH_MAX = 4500;	// 4.5 m

const float		KINECT_V2_FOVY = 60.f;
const float		KINECT_V2_FOVX = 70.f;
const float		KINECT_V2_COLOR_ASPECT_RATIO = 1.77f;	// 1920 / 1080
const float		KINECT_V2_DEPTH_ASPECT_RATIO = 1.20f;	// 512 / 424
const ushort	KINECT_V2_COLOR_WIDTH = 1920;
const ushort	KINECT_V2_COLOR_HEIGHT = 1080;
const ushort	KINECT_V2_DEPTH_WIDTH = 512;
const ushort	KINECT_V2_DEPTH_HEIGHT = 424;
const ushort	KINECT_V2_DEPTH_MIN = 50;	// 50 cm
const ushort	KINECT_V2_DEPTH_MAX= 4500;	// 4.5 m




#endif	//__KINECT_SPECS_H__
