#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QDateTime>
#include "QImageWidget.h"
#include "GLModelViewer.h"
#include "QKinectFile.h"
#include "KinectPlayerWidget.h"
#include "QKinectGrabberFromFile.h"
#include "QKinectIO.h"
#include "QKinectGpu.h"
#include "KinectSpecs.h"

#include <iostream>


#include "cuda_kernels\cuda_kernels.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "GLPointCloud.h"
#include "GLShaderProgram.h"
#include "helper_cuda.h"
#include "helper_image.h"
#include "Projection.h"



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
	try
	{
		KinectFrame frame;
		QKinectIO::loadFrame("../../data/knt_frames/frame_27.knt", frame);

		KinectCuda kcuda;
		kcuda.set_depth_buffer(frame.depth.data(), frame.depth_width(), frame.depth_height(), frame.depth_min_distance(), frame.depth_max_distance());
		kcuda.allocate();
		kcuda.copyHostToDevice();
		kcuda.runKernel();
		kcuda.copyDeviceToHost();
		export_obj_float4("../../data/room_normals_kcuda.obj", kcuda.vertices, kcuda.normals);
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Exception: " << ex.what() << std::endl;
	}

}


int main(int argc, char **argv)
{
	//test_kinect_gpu();
	//return 0;

	QDateTime now = QDateTime::currentDateTime();
	QString input_folder = ".";

	if (argc > 1)
		input_folder = argv[1];

	int fps = 30;
	if (argc > 2)
		fps = atoi(argv[2]);

	//
	// check if the path is a valid folder
	// 
	if (!QFileInfo(input_folder).isDir())
	{
		std::cerr << "Error: A valid folder is required" << std::endl;
		return EXIT_FAILURE;
	}


	QApplication app(argc, argv);

	int width = 640;
	int height = 480;

	QKinectGrabberFromFile* kinect = new QKinectGrabberFromFile();
	kinect->setFolder(input_folder);
	kinect->setFramesPerSecond(fps);

	QImageWidget colorWidget;
	colorWidget.setMinimumSize(320, 240);
	colorWidget.move(0, 0);
	colorWidget.show();
	QApplication::connect(kinect, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));
	QApplication::connect(kinect, SIGNAL(fileLoaded(QString)), &colorWidget, SLOT(setWindowTitle(QString)));

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(320, 240);
	depthWidget.move(0, 240);
	depthWidget.show();
	QApplication::connect(kinect, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));
	QApplication::connect(kinect, SIGNAL(fileLoaded(QString)), &depthWidget, SLOT(setWindowTitle(QString)));

	KinectPlayerWidget player(kinect);
	player.move(320, 500);
	player.show();

	//kinect->stopAndGo(true);
	kinect->start();
	
	//
	// setup opengl viewer
	// 
	GLPointCloudViewer glwidget;
	glwidget.resize(640, 480);
	glwidget.setPerspective(KINECT_V1_FOVY, KINECT_V1_DEPTH_MIN * 0.1f, KINECT_V1_DEPTH_MAX * 2);
	glwidget.move(320, 0);
	glwidget.setWindowTitle("Point Cloud");
	glwidget.setWeelSpeed(0.1f);
	glwidget.setDistance(-0.5f);
	glwidget.show();
	QApplication::connect(kinect, SIGNAL(fileLoaded(QString)), &glwidget, SLOT(setWindowTitle(QString)));


	//
	// setup model
	// 
	std::shared_ptr<GLPointCloud> cloud(new GLPointCloud);
	cloud->initGL();
	int vertex_count = 640 * 480;
	int vertex_tuple_size = 4;
	float* dumb_vertices = new float[vertex_count * vertex_tuple_size];
	cloud->setVertices(dumb_vertices, vertex_count, vertex_tuple_size);
	cloud->setNormals(dumb_vertices, vertex_count, vertex_tuple_size);
	delete dumb_vertices;
	cloud->transform().rotate(180, 0, 1, 0);
	glwidget.addPointCloud(cloud);


	//
	// setup kinect shader program
	// 
	std::shared_ptr<GLShaderProgram> kinectShaderProgram(new GLShaderProgram);
	if (kinectShaderProgram->build("normal2rgb.vert", "normal2rgb.frag"))
		cloud->setShaderProgram(kinectShaderProgram);


	QKinectGpu kinectGpu;
	kinectGpu.setKinect(kinect);
	kinectGpu.setPointCloudViewer(&glwidget);
	QApplication::connect(kinect, SIGNAL(frameUpdated()), &kinectGpu, SLOT(onFrameUpdate()));



	QApplication::connect(&player, SIGNAL(quit()), &depthWidget, SLOT(close()));
	QApplication::connect(&player, SIGNAL(quit()), &colorWidget, SLOT(close()));
	QApplication::connect(&player, SIGNAL(quit()), &glwidget, SLOT(close()));

#if 1
	int app_exit = app.exec();
#else
	int app_exit = 0;
	int i = 0;
	while (1)
	{
		i++;
		QCoreApplication::processEvents();  // ???
	}
	QCoreApplication::exit(app_exit);
#endif
	


	

	return app_exit;
}

