

#ifndef __Q_KINECT_PLAYER_CTRL_H__
#define __Q_KINECT_PLAYER_CTRL_H__

#include <QObject>
#include "QKinectGrabber.h"
#include "QKinectIO.h"


class MainWindow;
class GLDepthBufferRenderer;

struct DepthBuffer
{
	signed __int64 timespan;
	std::vector<unsigned short> info;
	std::vector<unsigned short> buffer;

	DepthBuffer() :info(4){}

	unsigned short width() const { return info[0]; }
	unsigned short height() const { return info[1]; }
	unsigned short minDistance() const { return info[2]; }
	unsigned short maxDistance() const { return info[3]; }
	std::size_t size() const { return buffer.size(); }

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

	void fileOpen(QString);
	void fileSave(QString);

	void updateFrame();
	void record(bool triggered);
	void capture(bool triggered);
	void takeShot();
	void playStream();
	void stopStream();


private:
	QKinectGrabber			kinectReader;
	QKinectIO				kinectStream;
	MainWindow*				view;
	GLDepthBufferRenderer*	depthRenderer;
	bool					recording;
	bool					frameUpdated;

public:
	DepthBuffer				mDepthBuffer;
};




#endif	//__Q_KINECT_PLAYER_CTRL_H__
