
#include "QKinectGpu.h"
#include <iostream>


QKinectGpu::QKinectGpu(QObject* parent) : 
	QObject(parent)
	, kinect(nullptr)
	, cloud(new GLPointCloud)
	, viewer(nullptr)
	, pointCloud(nullptr)
{

}

QKinectGpu::~QKinectGpu()
{
	if (kinect != nullptr)
	{
		kinectCuda.free();
		kinect = nullptr;
	}
}

void QKinectGpu::setPointCloudViewer(GLPointCloudViewer* viewer_ptr)
{
	viewer = viewer_ptr;
}


void QKinectGpu::setPointCloud(GLPointCloud* model_ptr)
{
	pointCloud = model_ptr;
}

void QKinectGpu::setKinect(QKinectGrabberFromFile* kinect_ptr)
{
	if (kinect_ptr == nullptr)
		return;

	kinect = kinect_ptr;

	std::vector<ushort> info;
	std::vector<ushort> depth;

	kinect->getDepthBuffer(info, depth);
	kinectCuda.set_depth_buffer(depth.data(), info[0], info[1], info[2], info[3]);
	kinectCuda.allocate();
}




void QKinectGpu::onFrameUpdate()
{	
	if (kinect == nullptr)
		return;
	
	//
	// copy frame from kinect device
	// 
	KinectFrame frame;
	kinect->getKinectFrame(frame);

	// 
	// run cuda kernel
	// 
	kinectCuda.set_depth_buffer(frame.depth.data(), frame.depth_width(), frame.depth_height(), frame.depth_min_distance(), frame.depth_max_distance());
	kinectCuda.copyHostToDevice();
	kinectCuda.runKernel();
	kinectCuda.copyDeviceToHost();

	//
	// emit signal saying that cuda kernel has been executed
	emit kernelExecuted();


	float* vertices = nullptr;
	size_t vertex_count = 0;
	size_t vertex_tuple_size = 0;
	float* normals = nullptr;
	size_t normal_count = 0;
	size_t normal_tuple_size = 0;
	kinectCuda.get_vertex_data(&vertices, vertex_count, vertex_tuple_size);
	kinectCuda.get_normal_data(&normals, normal_count, normal_tuple_size);
	
	if (viewer)
	{
		viewer->updateCloud(vertices, normals, vertex_count, vertex_tuple_size);
	}

	//if (pointCloud)
	//	pointCloud->updateVertices(vertices, vertex_count, vertex_tuple_size);
}