#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QDateTime>
#include "QImageWidget.h"
#include "QKinectFile.h"
#include "QKinectGrabberFromFile.h"
#include "QKinectIO.h"
#include "QKinectFrame.h"
#include <iostream>
#include <string>
#include "Eigen/Dense"
#include "Projection.h"
#include "ObjFile.h"




void exportKinectFrame2Obj(const std::string& kinect_file_name, const std::string& obj_file_name)
{
	KinectFrame frame(kinect_file_name);
	
	const float window_width = frame.depth_width();
	const float window_height = frame.depth_height();
	const float near_plane = frame.depth_min_distance();
	const float far_plane = frame.depth_max_distance();
	const float fovy = 43.0f;
	const float aspect_ratio = window_width / window_height;
	const float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
	const float x_scale = y_scale / aspect_ratio;
	const float depth_length = far_plane - near_plane;
	const  Eigen::Matrix4f projection_inverse = perspective_matrix_inverse(fovy, aspect_ratio, near_plane, far_plane);

	std::ofstream file;
	file.open(obj_file_name);

	for (int y = 0; y < window_height; ++y)
	{
		for (int x = 0; x < window_width; ++x)
		{
			const int d = y * window_width + x;
			const float depth = (float)frame.depth.at(d);
			
			if (depth < near_plane)
				continue;

			const Eigen::Vector3f v = window_coord_to_3d(Eigen::Vector2f(x, y), depth, projection_inverse, (int)window_width, (int)window_height);
			file << std::fixed << "v " << v.transpose() << std::endl;
		}
	}
	
	file.close();
}


int main(int argc, char **argv)
{
	QDateTime now = QDateTime::currentDateTime();
	QString input_folder;
	QString input_file;

	if (argc < 2)
	{
		std::cerr 
			<< "Missing parameters. Abort."	<< std::endl
			<< "Usage:  ./KinectFrame2Obj.exe input_file.knt" << std::endl
			<< std::endl;
		return EXIT_FAILURE;
	}

	
	QFileInfo fileinfo(argv[1]);
	if (fileinfo.isFile())
	{
		input_file = argv[1];
		QString obj_file_name = fileinfo.absoluteFilePath().remove(".knt") + ".obj";
		exportKinectFrame2Obj(argv[1], obj_file_name.toStdString());
	}
	else
	{
		std::cerr
			<< "Error: Could not recognize file . Abort." << std::endl
			<< std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

