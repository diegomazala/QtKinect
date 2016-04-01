
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
		std::cerr << "Usage: PointCloudViewer.exe monkey_low.obj" << std::endl;
		return EXIT_FAILURE;
	}

	Timer timer;
	std::string filename = argv[1];

	QApplication app(argc, argv);

	//
	// Importing .obj
	//
	timer.start();
	std::pair<PointCloudXYZW, PointCloudXYZW> points;
	import_obj(filename, points.first);
	timer.print_interval("Importing monkey    : ");
	std::cout << "Monkey point count  : " << points.first.size() << std::endl;


	// Rotating to generate second point cloud
	Eigen::Affine3f rotate = Eigen::Affine3f::Identity();
	rotate.rotate(Eigen::AngleAxisf(90.f * M_PI / 180.f, Eigen::Vector3f::UnitY()));
	for (const Eigen::Vector4f& p : points.first)
		points.second.push_back(rotate.matrix() * p);


	GLPointCloudViewer glwidget[2];
	glwidget[0].resize(512, 424);
	glwidget[0].move(0, 0);
	glwidget[0].setWindowTitle("Point Cloud Viewer");
	glwidget[0].show();
	glwidget[0].addPointCloud(&points.first);
	glwidget[0].setWeelSpeed(0.01);
	glwidget[0].setDistance(-5);
	
	glwidget[1].resize(512, 424);
	glwidget[1].move(512, 0);
	glwidget[1].setWindowTitle("Point Cloud Viewer");
	glwidget[1].show();
	glwidget[1].addPointCloud(&points.second);
	glwidget[1].setWeelSpeed(0.01);
	glwidget[1].setDistance(-5);


	return app.exec();
}
