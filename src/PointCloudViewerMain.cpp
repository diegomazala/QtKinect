
#include <QApplication>
#include <QKeyEvent>
#include "GLPointCloudViewer.h"
#include <iostream>
#include "Timer.h"
#include "ObjFile.h"




int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: PointCloudViewer.exe monkey.obj" << std::endl;
		return EXIT_FAILURE;
	}

	Timer timer;
	std::string filename = argv[1];

	QApplication app(argc, argv);

	//
	// Importing .obj
	//
	timer.start();
	PointCloudXYZW points;
	import_obj(filename, points);
	timer.print_interval("Importing monkey    : ");
	std::cout << "Monkey point count  : " << points.size() << std::endl;



	GLPointCloudViewer glwidget;
	glwidget.setMinimumSize(512, 424);
	glwidget.move(0, 0);
	glwidget.setWindowTitle("Vertex Map");
	glwidget.show();
	glwidget.addPointCloud(&points);
	

	return app.exec();
}
