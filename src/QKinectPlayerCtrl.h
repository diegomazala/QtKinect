

#ifndef __Q_KINECT_PLAYER_CTRL_H__
#define __Q_KINECT_PLAYER_CTRL_H__

#include <QObject>
#include "QKinectReader.h"
#include "QKinectIO.h"


class MainWindow;
class GLDepthBufferRenderer;

struct DepthBuffer
{
	signed __int64 timespan;
	unsigned short width, height, minDistance, maxDistance;
	std::vector<unsigned short> buffer;
};

class QKinectPlayerCtrl : public QObject
{
	Q_OBJECT

public:
	QKinectPlayerCtrl(QObject *parent = 0);
	virtual ~QKinectPlayerCtrl();

	void startKinect();
	void stopKinect();
	
	void setView(QWidget* viewUI);
	void setDepthRenderer(GLDepthBufferRenderer* depthRenderer);
	void setupConnections();

	bool isRecording() const;

public slots:

	void updateFrame();
	void record(bool triggered);
	void capture(bool triggered);
	void playStream();
	void stopStream();


private:
	QKinectReader			kinectReader;
	QKinectIO				kinectStream;
	MainWindow*				view;
	GLDepthBufferRenderer*	depthRenderer;
	bool					recording;
	bool					frameUpdated;

public:
	DepthBuffer				mDepthBuffer;
};




#endif	//__Q_KINECT_PLAYER_CTRL_H__
