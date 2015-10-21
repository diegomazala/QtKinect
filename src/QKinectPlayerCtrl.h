

#ifndef __Q_KINECT_PLAYER_CTRL_H__
#define __Q_KINECT_PLAYER_CTRL_H__

#include <QObject>
#include "QKinectReader.h"

class MainWindow;
class QLabel;

class QKinectPlayerCtrl : public QObject
{
	Q_OBJECT

public:
	QKinectPlayerCtrl(QObject *parent = 0);
	virtual ~QKinectPlayerCtrl();

	void startKinect();
	void stopKinect();
	
	void setView(QWidget* viewUI);
	//void setColorImageUI(QLabel* colorImageLabel);
	//void setDepthImageUI(QLabel* colorImageLabel);

	void setupConnections();

public slots:

signals:
	
protected:


private:
	QKinectReader	kinectReader;
	MainWindow*		view;
};




#endif	//__Q_KINECT_PLAYER_CTRL_H__
