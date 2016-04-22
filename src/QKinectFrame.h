

#ifndef __QKINECT_FRAME_H__
#define __QKINECT_FRAME_H__

#include "KinectFrame.h"
#include <QImage>

class QKinectFrame : public KinectFrame
{
public:

	QKinectFrame();
	
	QKinectFrame(
		const std::vector<unsigned short>& _info,
		const std::vector<unsigned char>& _color,
		const std::vector<unsigned short>& _depth);

	QKinectFrame(const QString& filename);

	QImage convertColorToImage() const;
	QImage convertDepthToImage() const;

	static void convertColorToImage(const std::vector<unsigned char>& buffer, const int width, const int height, QImage& image);
	static void convertDepthToImage(const std::vector<unsigned short>& buffer, const int width, const int height, const int max_distance, QImage& image);

};

#endif	//__QKINECT_FRAME_H__
