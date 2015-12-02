#if 1
#include <QApplication>
#include <QKeyEvent>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include <iostream>
#include <chrono>
#include <sstream>

class QKinectWidgetIO;
QVector<QRgb>		colorTable;
QKinectGrabber*		kinect;
QKinectWidgetIO*	colorWidgetLoad; 
QKinectWidgetIO*	depthWidgetLoad;
std::string			last_file_saved;

class QKinectWidgetIO : public QImageWidget
{
public:
	QKinectWidgetIO(QWidget* parent = nullptr) : QImageWidget(parent){}

	void save()
	{
		KinectFrameBuffer frame;
		kinect->copyFrameBuffer(frame);
		
		std::stringstream filename;
		filename << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() << ".kntf";
		
		std::cout << "saving kinect frame " << filename.str() << std::endl;

		QKinectIO::saveFrame(filename.str(), frame);

		last_file_saved = filename.str();
	};

	void load()
	{
		if (last_file_saved.empty())
			return;

		std::cout << "loading kinect frame " << last_file_saved << std::endl;

		KinectFrameBuffer frame;
		QKinectIO::loadFrame(last_file_saved, frame);
		
		/////////////////////////////////////
		//
		// create depth image
		QImage depthImg = QImage(frame.depth_width(), frame.depth_height(), QImage::Format::Format_Indexed8);
		depthImg.setColorTable(colorTable);

		std::vector<unsigned char> depthImgBuffer(frame.depth.size());

		// casting from unsigned short (2 bytes precision) to unsigned char (1 byte precision)
		std::transform(
			frame.depth.begin(),
			frame.depth.end(),
			depthImgBuffer.begin(),
			[=](const unsigned short depth) { return static_cast<unsigned char>((float)depth / (float)frame.depth_max_distance() * 255.f); });

		// set pixels to depth image
		for (int y = 0; y < depthImg.height(); y++)
			memcpy(depthImg.scanLine(y), depthImgBuffer.data() + y * depthImg.width(), depthImg.width());

		depthWidgetLoad->setImage(depthImg);
		colorWidgetLoad->setImage(QImage(frame.color.data(), frame.color_width(), frame.color_height(), QImage::Format_ARGB32));

		std::cout << "loaded kinect frame " << last_file_saved << std::endl;
	};


protected:
	void keyReleaseEvent(QKeyEvent* e)
	{
		
		if (e->key() == Qt::Key_S)
			this->save();
		else if (e->key() == Qt::Key_L)
			this->load();
		else if (e->key() == Qt::Key_Q || e->key() == Qt::Key_Escape)
			this->close();
	}
};


int main(int argc, char **argv)
{
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));

	std::cout << "Press 's' to save a frame" << std::endl;
	std::cout << "Press 'l' to load last frame saved" << std::endl;

	QApplication app(argc, argv);
	
	kinect = new QKinectGrabber();
	kinect->start();

	QKinectWidgetIO* colorWidget = new QKinectWidgetIO();
	colorWidget->setMinimumSize(720, 480);
	colorWidget->move(0, 0);
	colorWidget->show();
	QApplication::connect(kinect, SIGNAL(colorImage(QImage)), colorWidget, SLOT(setImage(QImage)));

	QKinectWidgetIO* depthWidget = new QKinectWidgetIO();
	depthWidget->setMinimumSize(512, 424);
	depthWidget->move(720, 0);
	depthWidget->show();
	QApplication::connect(kinect, SIGNAL(depthImage(QImage)), depthWidget, SLOT(setImage(QImage)));


	colorWidgetLoad = new QKinectWidgetIO();
	colorWidgetLoad->setMinimumSize(720, 480);
	colorWidgetLoad->move(0, 480);
	colorWidgetLoad->show();
	//QApplication::connect(&kinect, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));

	depthWidgetLoad = new QKinectWidgetIO();
	depthWidgetLoad->setMinimumSize(512, 424);
	depthWidgetLoad->move(720, 480);
	depthWidgetLoad->show();
	//QApplication::connect(&kinect, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));

	int app_exit = app.exec();
	kinect->stop();
	return app_exit;


	return 0;
}

#else



#include <QApplication>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include <iostream>


int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	QKinectGrabber k;
	k.start();

	QImageWidget colorWidget;
	colorWidget.setMinimumSize(720, 480);
	colorWidget.show();
	QApplication::connect(&k, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(512, 424);
	depthWidget.show();
	QApplication::connect(&k, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));

	//QImageWidget infraredWidget;
	//infraredWidget.setMinimumSize(512, 424);
	//infraredWidget.show();
	//QApplication::connect(&k, SIGNAL(infraredImage(QImage)), &infraredWidget, SLOT(setImage(QImage)));

	int app_exit = app.exec();
	k.stop();
	return app_exit;

}

#endif