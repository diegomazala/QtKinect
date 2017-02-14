
#include <QApplication>
#include <QKeyEvent>
#include "GLPointCloudViewer.h"
#include <iostream>
#include "Timer.h"
#include "ObjFile.h"


struct Cloud
{
	std::string filename;
	std::vector<Eigen::Vector4f> vertices;
};


int main(int argc, char **argv)
{

	if (argc < 2)
	{
		std::cerr << "Usage: PointCloudViewer.exe obj_file_point_cloud " << std::endl;
		std::cerr << "Usage: PointCloudViewer.exe monkey_low.obj cube.obj" << std::endl;
		return EXIT_FAILURE;
	}


	Timer timer;
	std::vector<std::string> filenames;
	std::vector<std::vector<Eigen::Vector4f>> clouds;
	for (int s = 1; s < argc; ++s)
	{
		// Importing .obj
		//
		timer.start();
		std::vector<Eigen::Vector4f> vertices;
		if (import_obj(argv[s], vertices))
		{
			filenames.push_back(argv[s]);
			clouds.push_back(vertices);
		}

		timer.print_interval("Importing file    : ");
		timer.print_interval_msec("Importing file    : ");
		std::cout << "Vertices count  : " << vertices.size() << std::endl;
	}

	std::cout << "Number of clouds: " << clouds.size() << std::endl;
	
	//
	// Viewer
	//
	QApplication app(argc, argv);

	GLPointCloudViewer glwidget;
	glwidget.resize(1024, 848);
	glwidget.move(0, 0);
	glwidget.setWindowTitle("Point Cloud");
	glwidget.show();

	for (const auto c : clouds)
	{
		std::shared_ptr<GLPointCloud> cloud(new GLPointCloud);
		cloud->initGL();
		cloud->setVertices((float*)&c.data()[0], c.size(), 4);
		cloud->setColor(QVector3D(1, 1, 0));
		glwidget.addPointCloud(cloud);
	}

	glwidget.setWeelSpeed(0.01);
	glwidget.setPosition(0, 0, -5);

	return app.exec();
}
