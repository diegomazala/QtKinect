#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QDateTime>
#include "QImageWidget.h"
#include "GLModelViewer.h"
#include "QKinectFile.h"
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
#include "KinectShaderProgram.h"
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
	colorWidget.setMinimumSize(320, 240);
	colorWidget.move(0, 0);
	colorWidget.show();
	QApplication::connect(kinect, SIGNAL(colorImage(QImage)), &colorWidget, SLOT(setImage(QImage)));

	QImageWidget depthWidget;
	depthWidget.setMinimumSize(320, 240);
	depthWidget.move(0, 240);
	depthWidget.show();
	QApplication::connect(kinect, SIGNAL(depthImage(QImage)), &depthWidget, SLOT(setImage(QImage)));

#if 0
	GLModelViewer glViewer;
	glViewer.setMinimumSize(640, 480);
	glViewer.move(640, 0);
	glViewer.setWindowTitle("Point Cloud");
	glViewer.setWeelSpeed(0.1f);
	glViewer.setDistance(-0.5f);
	glViewer.show();

	GLPointCloud pointCloud;
	pointCloud.initGL();
	float4 vv = make_float4(1, 1, 1, 1);
	pointCloud.setVertices(&vv.x, 1, 4);
	//pointCloud.setVertices(&vertices[0].x, static_cast<uint>(vertices.size()), static_cast<uint>(4));
	//pointCloud.setColors(&rgb[0].x, static_cast<uint>(rgb.size()), static_cast<uint>(3));
	glViewer.setModel(&pointCloud);
#endif


	GLPointCloudViewer glwidget;
	glwidget.resize(640, 480);
	glwidget.setPerspective(KINECT_V1_FOVY, KINECT_V1_DEPTH_MIN * 0.1f, KINECT_V1_DEPTH_MAX * 2);
	glwidget.move(320, 0);
	glwidget.setWindowTitle("Point Cloud");
	glwidget.setWeelSpeed(0.1f);
	glwidget.setDistance(-0.5f);
	glwidget.show();
	QApplication::connect(kinect, SIGNAL(fileLoaded(QString)), &glwidget, SLOT(setWindowTitle(QString)));

	std::shared_ptr<GLPointCloud> cloud(new GLPointCloud);
	cloud->initGL();
	int vertex_count = 640 * 480;
	int vertex_tuple_size = 4;
	float* dumb_vertices = new float[vertex_count * vertex_tuple_size];
	cloud->setVertices(dumb_vertices, vertex_count, vertex_tuple_size);
	cloud->setNormals(dumb_vertices, vertex_count, vertex_tuple_size);
	delete dumb_vertices;
	glwidget.addPointCloud(cloud);


	//
	// setup kinect shader program
	// 
	std::shared_ptr<KinectShaderProgram> kinectShaderProgram(new KinectShaderProgram);
	if (kinectShaderProgram->build("normal2rgb.vert", "normal2rgb.frag"))
		glwidget.setShaderProgram(kinectShaderProgram);


	QKinectGpu kinectGpu;
	kinectGpu.setKinect(kinect);
	kinectGpu.setPointCloudViewer(&glwidget);
	QApplication::connect(kinect, SIGNAL(frameUpdated()), &kinectGpu, SLOT(onFrameUpdate()));


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
	kinect->stop();


	

	return app_exit;
}

