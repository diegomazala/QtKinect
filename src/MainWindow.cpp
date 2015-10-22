#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "QKinectPlayerCtrl.h"
#include <QKeyEvent>
#include <QFileDialog>
#include <QDebug>
#include <QMessageBox>
#include <QStandardItemModel>
#include <iostream>


MainWindow::MainWindow(QWidget *parent) : 
			QMainWindow(parent),
			ui(new Ui::MainWindow),
			currentFileName(QString())
{
	ui->setupUi(this);
}



MainWindow::~MainWindow()
{
	delete ui;
}


void MainWindow::setController(QKinectPlayerCtrl* ctrl)
{
	controller = ctrl;
	controller->setView(this);
}


void MainWindow::setColorImage(const QImage& image)
{
	if (!image.isNull())
	{
		ui->colorImageLabel->setPixmap(QPixmap::fromImage(image).scaled(ui->colorImageLabel->width(), ui->colorImageLabel->height(), Qt::KeepAspectRatio));
	}
}


void MainWindow::setDepthImage(const QImage& image)
{
	ui->depthImageLabel->setPixmap(QPixmap::fromImage(image).scaled(ui->depthImageLabel->width(), ui->depthImageLabel->height(), Qt::KeepAspectRatio));
}



void MainWindow::fileNew()
{
	currentFileName.clear();
}




void MainWindow::fileOpen()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "", tr("Kinect Stream (*.knt)"));

	if (!fileName.isEmpty())
	{
		currentFileName = fileName;
	}
}




void MainWindow::fileSave()
{
	if (!currentFileName.isEmpty())
	{
	}
	else
	{
		fileSaveAs();
	}
}




void MainWindow::fileSaveAs()
{
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("Kinect Stream (*.knt)"));
	if (!fileName.isEmpty())
	{
		currentFileName = fileName;
		fileSave();
	}
}




void MainWindow::playerPlay()
{
	emit play();
}


void MainWindow::playerStop()
{
	emit stop();
}


void MainWindow::playerRecord(bool triggered)
{
	emit recordToggled(triggered);
#if 0
	if (triggered)
	{
		std::cout << "++++ START ++++" << std::endl;
		controller->startRecord();
	}
	else
	{
		std::cout << "++++ STOP ++++" << std::endl;
		controller->stopRecord();
	}
#endif
}


void MainWindow::aboutDialogShow()
{
	QString message
		("<p>Alpha Matting algorithm using Qt and Opengl" \
		"<p><p>" \
		"<br>   [1] K. He, C. Rhemann, C. Rother, X. Tang, J. Sun, A Global Sampling Method for Alpha Matting, CVPR, 2011. <br>" \
		"<br>   [2] C. Tomasi and R. Manduchi, Bilateral Filtering for Gray and Color Images, Proc.IEEE Intel Computer Vision Conference, 1998. <br>" \
		"<br>   [3] K.He, J.Sun, and X.Tang, Guided Image Filtering,  Proc. European Conf.Computer Vision, pp. 1 - 14, 2010. <br>" \
		"<p><p><p>" \
		"<p>Developed by: Diego Mazala, June-2015" \
		"<p>");

	QMessageBox::about(this, tr("Alpha Matting"), message);
}

