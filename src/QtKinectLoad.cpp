
#include <QApplication>
#include <QKeyEvent>
#include "QImageWidget.h"
#include "QKinectIO.h"
#include "GLKinectWidget.h"
#include <iostream>




int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: QtKinectLoad.exe scene.knt" << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = argv[1];

	QVector<QRgb>		colorTable;
	QImage				colorImage;
	QImage				depthImage;

	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));

	QApplication app(argc, argv);

	KinectFrame frame;
	QKinectIO::loadFrame(filename, frame);

	/////////////////////////////////////
	//
	// create depth image
	depthImage = QImage(frame.depth_width(), frame.depth_height(), QImage::Format::Format_Indexed8);
	depthImage.setColorTable(colorTable);

	std::vector<unsigned char> depthImgBuffer(frame.depth.size());

	// casting from unsigned short (2 bytes precision) to unsigned char (1 byte precision)
	std::transform(
		frame.depth.begin(),
		frame.depth.end(),
		depthImgBuffer.begin(),
		[=](const unsigned short depth) { return static_cast<unsigned char>((float)depth / (float)frame.depth_max_distance() * 255.f); });

	// set pixels to depth image
	for (int y = 0; y < depthImage.height(); y++)
		memcpy(depthImage.scanLine(y), depthImgBuffer.data() + y * depthImage.width(), depthImage.width());

	bool is_kinect_v2 = (frame.color_width() == 1920);

	if (is_kinect_v2)	// Kinect v2
		colorImage = QImage(frame.color.data(), frame.color_width(), frame.color_height(), QImage::Format_ARGB32);
	else     // kinect v1
		colorImage = QImage(frame.color.data(), frame.color_width(), frame.color_height(), QImage::Format_RGB32);



	QImageWidget colorWidget;
	colorWidget.setMinimumSize(720, 480);
	colorWidget.move(0, 0);
	colorWidget.setImage(colorImage);
	colorWidget.setWindowTitle("Color Map");
	colorWidget.show();
	

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(512, 424);
	depthWidget.move(720, 0);
	depthWidget.setImage(depthImage);
	depthWidget.setWindowTitle("Depth Map");
	depthWidget.show();

	if (is_kinect_v2)
	{
		GLKinectWidget glwidget;
		glwidget.setMinimumSize(512, 424);
		glwidget.move(720, 512);
		glwidget.setWindowTitle("Vertex Map");
		glwidget.show();
		glwidget.setFrame(&frame);
	}

	return app.exec();
}
