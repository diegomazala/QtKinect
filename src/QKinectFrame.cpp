
#include "QKinectFrame.h"


QKinectFrame::QKinectFrame() : KinectFrame()
{
}


QKinectFrame::QKinectFrame(
	const std::vector<unsigned short>& _info,
	const std::vector<unsigned char>& _color,
	const std::vector<unsigned short>& _depth) 
	: KinectFrame(_info, _color, _depth)
{
}


QKinectFrame::QKinectFrame(const QString& filename)
	: KinectFrame(filename.toStdString())
{
}



QImage QKinectFrame::convertColorToImage() const
{
	return QImage(color.data(), color_width(), color_height(), QImage::Format_RGB32);
}



QImage QKinectFrame::convertDepthToImage() const
{
	QVector<QRgb> colorTable;
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));

	// create depth image
	QImage depthImg = QImage(depth_width(), depth_height(), QImage::Format::Format_Indexed8);
	depthImg.setColorTable(colorTable);

	std::vector<unsigned char> depthImgBuffer(depth.size());

	// casting from unsigned short (2 bytes precision) to unsigned char (1 byte precision)
	std::transform(
		depth.begin(),
		depth.end(),
		depthImgBuffer.begin(),
		[=](const unsigned short depth) { return static_cast<unsigned char>((float)depth / (float)depth_max_distance() * 255.f); });

	// set pixels to depth image
	for (int y = 0; y < depthImg.height(); y++)
		memcpy(depthImg.scanLine(y), depthImgBuffer.data() + y * depthImg.width(), depthImg.width());

	return depthImg;
}



void QKinectFrame::convertColorToImage(const std::vector<unsigned char>& buffer, const int width, const int height, QImage& image)
{
	image = QImage(buffer.data(), width, height, QImage::Format_RGB32);
}



void QKinectFrame::convertDepthToImage(const std::vector<unsigned short>& buffer, const int width, const int height, const int max_distance, QImage& image)
{
	QVector<QRgb> colorTable;
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));

	// create depth image
	image = QImage(width, height, QImage::Format::Format_Indexed8);
	image.setColorTable(colorTable);

	std::vector<unsigned char> depthImgBuffer(buffer.size());

	// casting from unsigned short (2 bytes precision) to unsigned char (1 byte precision)
	std::transform(
		buffer.begin(),
		buffer.end(),
		depthImgBuffer.begin(),
		[=](const unsigned short depth) { return static_cast<unsigned char>((float)depth / (float)max_distance * 255.f); });

	// set pixels to depth image
	for (int y = 0; y < image.height(); y++)
		memcpy(image.scanLine(y), depthImgBuffer.data() + y * image.width(), image.width());
}