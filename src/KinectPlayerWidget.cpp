
#include <QtWidgets>
#include "KinectPlayerWidget.h"
#include "QKinectGrabberFromFile.h"


KinectPlayerWidget::KinectPlayerWidget(QKinectGrabberFromFile* grabber, QWidget *parent)
    : QWidget(parent)
	, kinectGrabber(grabber)
{
    currentMovieDirectory = ".";

    createButtons();

    mainLayout = new QVBoxLayout;
    mainLayout->addLayout(buttonsLayout);
    setLayout(mainLayout);

    updateButtons();

    setWindowTitle(tr("Kinect Player Controls"));
    resize(400, 120);
}

void KinectPlayerWidget::setKinectGrabber(QKinectGrabberFromFile* grabber)
{
	kinectGrabber = grabber;
}

void KinectPlayerWidget::open()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open a Movie"),
                               currentMovieDirectory);
    if (!fileName.isEmpty())
        openFile(fileName);
}

void KinectPlayerWidget::openFile(const QString &fileName)
{
    currentMovieDirectory = QFileInfo(fileName).path();

    updateButtons();
}



void KinectPlayerWidget::updateButtons()
{
    playButton->setEnabled(false);
    pauseButton->setEnabled(false);
    stopButton->setEnabled(false);
	backwardButton->setEnabled(false);
}



void KinectPlayerWidget::createButtons()
{
    QSize iconSize(36, 36);

    openButton = new QToolButton;
    openButton->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
    openButton->setIconSize(iconSize);
    openButton->setToolTip(tr("Open File"));
    connect(openButton, SIGNAL(clicked()), this, SLOT(open()));

	backwardButton = new QToolButton;
	backwardButton->setIcon(style()->standardIcon(QStyle::SP_MediaSeekBackward));
	backwardButton->setIconSize(iconSize);
	backwardButton->setToolTip(tr("Forward"));
	//connect(backwardButton, SIGNAL(clicked()), movie, SLOT(backwardButton()));

	forwardButton = new QToolButton;
	forwardButton->setIcon(style()->standardIcon(QStyle::SP_MediaSeekForward));
	forwardButton->setIconSize(iconSize);
	forwardButton->setToolTip(tr("Forward"));
	//connect(forwardButton, SIGNAL(clicked()), movie, SLOT(forward()));
	connect(forwardButton, SIGNAL(clicked()), kinectGrabber, SLOT(resume()));

    playButton = new QToolButton;
	playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    playButton->setIconSize(iconSize);
    playButton->setToolTip(tr("Play"));
    //connect(playButton, SIGNAL(clicked()), movie, SLOT(start()));
		
    pauseButton = new QToolButton;
    pauseButton->setCheckable(true);
    pauseButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    pauseButton->setIconSize(iconSize);
    pauseButton->setToolTip(tr("Pause"));
    //connect(pauseButton, SIGNAL(clicked(bool)), movie, SLOT(setPaused(bool)));

    stopButton = new QToolButton;
    stopButton->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    stopButton->setIconSize(iconSize);
    stopButton->setToolTip(tr("Stop"));
    //connect(stopButton, SIGNAL(clicked()), movie, SLOT(stop()));

    quitButton = new QToolButton;
    quitButton->setIcon(style()->standardIcon(QStyle::SP_DialogCloseButton));
    quitButton->setIconSize(iconSize);
    quitButton->setToolTip(tr("Quit"));
    connect(quitButton, SIGNAL(clicked()), this, SLOT(close()));

    buttonsLayout = new QHBoxLayout;
    buttonsLayout->addStretch();
    buttonsLayout->addWidget(openButton);
	buttonsLayout->addWidget(backwardButton);
	buttonsLayout->addWidget(forwardButton);
	buttonsLayout->addWidget(playButton);
    buttonsLayout->addWidget(pauseButton);
    buttonsLayout->addWidget(stopButton);
    buttonsLayout->addWidget(quitButton);
    buttonsLayout->addStretch();
}


void KinectPlayerWidget::close()
{
	emit quit();
	QWidget::close();
}