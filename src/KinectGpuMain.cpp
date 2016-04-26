#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QDateTime>
#include "QImageWidget.h"
#include "QKinectFile.h"
#include "QKinectGrabberFromFile.h"
#include "QKinectIO.h"
#include "QKinectGpu.h"

#include <iostream>


#include "cuda_kernels\cuda_kernels.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "GLPointCloud.h"
#include "helper_cuda.h"
#include "helper_image.h"
#include "Projection.h"

template<typename T>
static void vector_read(std::istream& in_file, std::vector<T>& data)
{
	std::size_t count;
	in_file.read(reinterpret_cast<char*>(&count), sizeof(std::size_t));
	data.resize(count);
	in_file.read(reinterpret_cast<char*>(&data[0]), count * sizeof(T));
}


template<typename T>
bool load_buffer(const std::string& filename, std::vector<T>& data)
{
	std::ifstream in_file;
	in_file.open(filename, std::ifstream::binary);
	if (in_file.is_open())
	{
		vector_read(in_file, data);
		in_file.close();
		return true;
	}
	return false;
}



static void export_obj_float4(const std::string& filename, const std::vector<float4>& vertices, const std::vector<float4>& normals)
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < vertices.size(); ++i)
	{
		const float4& v = vertices[i];
		const float4& n = normals[i];
		file << std::fixed << "v "
			<< v.x << ' ' << v.y << ' ' << v.z << ' '
			<< ((n.x * 0.5) + 0.5) * 255 << ' ' << ((n.y * 0.5) + 0.5) * 255 << ' ' << ((n.z * 0.5) + 0.5) * 255
			<< std::endl;
	}
	file.close();
}







void test_kinect_gpu()
{
	KinectFrame frame;
	QKinectIO::loadFrame("../knt_frames/frame_11.knt", frame);

	KinectCuda kcuda;
	kcuda.set_depth_buffer(frame.depth.data(), frame.depth_width(), frame.depth_height(), frame.depth_min_distance(), frame.depth_max_distance());
	kcuda.allocate();
	kcuda.copyHostToDevice();
	kcuda.runKernel();
	kcuda.copyDeviceToHost();
	export_obj_float4("../../data/room_normals_kcuda.obj", kcuda.vertices, kcuda.normals);
}


int main(int argc, char **argv)
{
//	test_kinect_gpu();
//	return 0;

	QDateTime now = QDateTime::currentDateTime();
	QString input_folder = ".";

	if (argc > 1)
		input_folder = argv[1];

	int fps = 30;
	if (argc > 2)
		fps = atoi(argv[2]);

	QApplication app(argc, argv);

	int width = 640;
	int height = 480;

	QKinectGrabberFromFile* kinect = new QKinectGrabberFromFile();
	kinect->setFolder(input_folder);
	kinect->setFramesPerSecond(fps);
	kinect->start();




	QImageWidget colorWidget;
	colorWidget.setMinimumSize(640, 480);
	colorWidget.move(0, 0);
	colorWidget.show();
	QApplication::connect(kinect, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(640, 480);
	depthWidget.move(0, 480);
	depthWidget.show();
	QApplication::connect(kinect, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));

	

	GLPointCloudViewer glwidget;
	glwidget.resize(640, 480);
	glwidget.move(640, 0);
	glwidget.setWindowTitle("Point Cloud");
	glwidget.setWeelSpeed(0.1f);
	glwidget.setDistance(-0.5f);
	glwidget.show();
	QApplication::connect(kinect, SIGNAL(fileLoaded(QString)), &glwidget, SLOT(setWindowTitle(QString)));

	std::shared_ptr<GLPointCloud> cloud(new GLPointCloud);
	cloud->initGL();
	float4 v = make_float4(1, 1, 1, 1);
	cloud->setVertices(&v.x, 1, 4);
	glwidget.addPointCloud(cloud);

	QKinectGpu kinectGpu;
	kinectGpu.setKinect(kinect);
	kinectGpu.setPointCloudViewer(&glwidget);
	QApplication::connect(kinect, SIGNAL(frameUpdated()), &kinectGpu, SLOT(onFrameUpdate()));


	//int app_exit = app.exec();
	int app_exit = 0;
	int i = 0;
	while (1)
	{
		i++;
		QCoreApplication::processEvents();  // ???

		
	}
	QCoreApplication::exit(app_exit);
	kinect->stop();


	

	return app_exit;
}

