
#include <QApplication>
#include <QKeyEvent>
#include <QPushButton>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include <iostream>
#include <chrono>
#include <sstream>

QKinectGrabber*		kinect;

class QKeyCtrl : public QPushButton
{
public:
	QKeyCtrl(QWidget* parent = nullptr) 
		: QPushButton(parent), colorWidget(nullptr), depthWidget(nullptr) {}

	void setWidgets(QImageWidget* color, QImageWidget* depth)
	{
		colorWidget = color;
		depthWidget = depth;
	}

	void save()
	{
		KinectFrame frame;
		kinect->copyFrameBuffer(frame);

		std::stringstream filename, filename_knt, filename_color, filename_depth;
		filename << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		filename_knt << filename.str() << ".knt";
		filename_color << filename.str() << "_color.png";
		filename_depth << filename.str() << "_depth.png";

		std::cout << "saving kinect frame " << filename_knt.str() << std::endl;
		QKinectIO::saveFrame(filename_knt.str(), frame);

		if (colorWidget != nullptr)
			colorWidget->pixmap()->save(QString::fromStdString(filename_color.str()));
		if (depthWidget != nullptr)
			depthWidget->pixmap()->save(QString::fromStdString(filename_depth.str()));
		
	};

protected:

	void click()
	{
		std::cout << "click" << std::endl;
	}

	void keyReleaseEvent(QKeyEvent* e)
	{
		if (e->key() == Qt::Key_S)
			this->save();
		else if (e->key() == Qt::Key_Q || e->key() == Qt::Key_Escape)
		{
			colorWidget->close();
			depthWidget->close();
			QApplication::quit();
		}
	}

	QImageWidget* colorWidget;
	QImageWidget* depthWidget;
};

class QKinectWidgetIO : public QImageWidget
{
public:
	QKinectWidgetIO(QWidget* parent = nullptr) : QImageWidget(parent){}

	void save()
	{
		KinectFrame frame;
		kinect->copyFrameBuffer(frame);
		
		std::stringstream filename, filename_knt, filename_png;
		filename << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		filename_knt << filename.str() << ".knt";
		filename_png << filename.str() << ".png";
		
		std::cout << "saving kinect frame " << filename_knt.str() << std::endl;
		QKinectIO::saveFrame(filename_knt.str(), frame);
		this->pixmap()->save(QString::fromStdString(filename_png.str()));

	};

protected:
	//void keyReleaseEvent(QKeyEvent* e)
	//{
	//	
	//	if (e->key() == Qt::Key_S)
	//		this->save();
	//	else if (e->key() == Qt::Key_Q || e->key() == Qt::Key_Escape)
	//		this->close();
	//}
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

	QKeyCtrl ctrl;
	ctrl.setMinimumSize(512, 50);
	ctrl.setText("Press 's' to save");
	ctrl.move(720, 430);
	ctrl.setWidgets(colorWidget, depthWidget);
	ctrl.setFocus();
	ctrl.show();

	int app_exit = app.exec();
	kinect->stop();
	return app_exit;


	return 0;
}
