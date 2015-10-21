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

}


void MainWindow::playerStop()
{

}


void MainWindow::playerRecord(bool triggered)
{
	if (triggered)
		std::cout << "start recording" << std::endl;
	else
		std::cout << "stop recording" << std::endl;
}


void MainWindow::aboutDialogShow()
{
}

