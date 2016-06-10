
#ifndef _KINECT_PLAYER_WIDGET_H_
#define _KINECT_PLAYER_WIDGET_H_

#include <QWidget>

QT_BEGIN_NAMESPACE
class QCheckBox;
class QGridLayout;
class QHBoxLayout;
class QLabel;
class QMovie;
class QSlider;
class QSpinBox;
class QToolButton;
class QVBoxLayout;
QT_END_NAMESPACE

class QKinectGrabberFromFile;

class KinectPlayerWidget : public QWidget
{
    Q_OBJECT

public:
	KinectPlayerWidget(QKinectGrabberFromFile* grabber, QWidget *parent = 0);
    void openFile(const QString &fileName);

	void setKinectGrabber(QKinectGrabberFromFile* grabber);

private slots:
    void open();
    void updateButtons();
	virtual void close();

Q_SIGNALS:
	void quit();

private:
    void createButtons();

    QString currentMovieDirectory;
    QToolButton *openButton;
    QToolButton *playButton;
	QToolButton *backwardButton;
	QToolButton *forwardButton;
    QToolButton *pauseButton;
    QToolButton *stopButton;
    QToolButton *quitButton;

    QHBoxLayout *buttonsLayout;
    QVBoxLayout *mainLayout;

	QKinectGrabberFromFile*	kinectGrabber;
};

#endif	// _KINECT_PLAYER_WIDGET_H_
