
#include <QApplication>
#include "MainWindow.h"
#include "QKinectPlayerCtrl.h"
#include <iostream>



int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	MainWindow w;
	w.show();

	QKinectPlayerCtrl controller;
	controller.setView(&w);
	controller.setupConnections();
	controller.startKinect();

	return app.exec();
}

