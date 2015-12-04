
#include <QApplication>
#include <QKeyEvent>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include <iostream>
#include <chrono>
#include <sstream>

QKinectGrabber*		kinect;

class QKinectWidgetIO : public QImageWidget
{
public:
	QKinectWidgetIO(QWidget* parent = nullptr) : QImageWidget(parent){}

	void save()
	{
		KinectFrameBuffer frame;
		kinect->copyFrameBuffer(frame);
		
		std::stringstream filename;
		filename << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() << ".knt";
		
		std::cout << "saving kinect frame " << filename.str() << std::endl;

		QKinectIO::saveFrame(filename.str(), frame);
	};

protected:
	void keyReleaseEvent(QKeyEvent* e)
	{
		
		if (e->key() == Qt::Key_S)
			this->save();
		else if (e->key() == Qt::Key_Q || e->key() == Qt::Key_Escape)
			this->close();
	}
};


int main(int argc, char **argv)
{
	std::cout << "Press 's' to save a frame" << std::endl;

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

	int app_exit = app.exec();
	kinect->stop();
	return app_exit;


	return 0;
}
