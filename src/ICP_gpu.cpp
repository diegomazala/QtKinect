
#include <QApplication>
#include <QFileInfo>
#include <QDir>
#include "GLPointCloudViewer.h"
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include <iostream>
#include <iterator>
#include <array>
#include <memory>
#include <time.h>

#include "Volumetric_helper.h"
#include "Timer.h"
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_kernels/cuda_kernels.h"

#include "GLPointCloud.h"

#include "helper_cuda.h"
#include "helper_image.h"

static QVector3D colors[7] = {
	QVector3D(1, 0, 0), QVector3D(0, 1, 0), QVector3D(0, 0, 1),
	QVector3D(1, 1, 0), QVector3D(1, 0, 1), QVector3D(0, 1, 1),
	QVector3D(1, 1, 1) };



static void export_obj_float4(const std::string& filename, const std::vector<float4>& vertices, const std::vector<float4>& normals)
{
	std::ofstream file;
	file.open(filename);
	//for (const float4 v : vertices)
	//	file << std::fixed << "v " << v.x << ' ' << v.y << ' ' << v.z << std::endl;
	//for (const float4 n : normals)
	//	file << std::fixed << "vn " << n.x << ' ' << n.y << ' ' << n.z << std::endl;

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

void convert_normal_to_rgba(std::vector<float4>& normals)
{
	for (float4& n : normals)
	{
		n.x = (n.x * 0.5) + 0.5;
		n.y = (n.y * 0.5) + 0.5;
		n.z = (n.z * 0.5) + 0.5;
		n.w = (n.w * 0.5) + 0.5;
	}
}



void run_back_projection_with_normal_estimate(
	std::vector<float4>& vertices,
	std::vector<float4>& normals,
	const std::vector<ushort>& depth_buffer,
	uint width,
	uint height,
	ushort max_depth)
{
	StopWatchInterface *kernel_timer = nullptr;

	ushort* h_depth_buffer = (ushort*)depth_buffer.data();

	size_t in_pitch, out_pitch;

	ushort* d_depth_buffer = nullptr;
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&d_depth_buffer, &in_pitch, sizeof(ushort) * width, height));
	checkCudaErrors(cudaMemcpy2D(
		d_depth_buffer,
		in_pitch,
		h_depth_buffer,
		sizeof(ushort) * width,
		sizeof(ushort) * width,
		height,
		cudaMemcpyHostToDevice));



	float4* d_vertex_buffer;
	checkCudaErrors(cudaMallocPitch(
		&d_vertex_buffer,
		&out_pitch,
		width * sizeof(float4),
		height));

	float4* d_normal_buffer;
	checkCudaErrors(cudaMallocPitch(
		&d_normal_buffer,
		&out_pitch,
		width * sizeof(float4),
		height));


	sdkCreateTimer(&kernel_timer);
	sdkStartTimer(&kernel_timer);

	Eigen::Matrix4f h_inverse_projection = perspective_matrix_inverse<float>(fov_y, aspect_ratio, near_plane, far_plane);
	//bilateralFilter_normal_estimate_float4((OutputPixelType*)dOutputImage, (InputPixelType*)dInputImage, width, height, in_pitch, out_pitch, max_depth, euclidean_delta, filter_radius, iterations, kernel_timer);
	back_projection_with_normal_estimation(d_vertex_buffer, d_normal_buffer, d_depth_buffer, width, height, max_depth, in_pitch, out_pitch, h_inverse_projection.data());

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&kernel_timer);
	std::cout << "Kernel Timer                              : " << kernel_timer->getTime() << " msec" << std::endl;
	sdkDeleteTimer(&kernel_timer);

	vertices.resize(depth_buffer.size());
	normals.resize(depth_buffer.size());

	cudaMemcpy2D(
		vertices.data(),
		sizeof(float4) * width,
		d_vertex_buffer,
		out_pitch,
		sizeof(float4) * width,
		height,
		cudaMemcpyDeviceToHost);

	cudaMemcpy2D(
		normals.data(),
		sizeof(float4) * width,
		d_normal_buffer,
		out_pitch,
		sizeof(float4) * width,
		height,
		cudaMemcpyDeviceToHost);


	checkCudaErrors(cudaFree(d_depth_buffer));
	checkCudaErrors(cudaFree(d_vertex_buffer));
	checkCudaErrors(cudaFree(d_normal_buffer));
}



void run_icp_with_normal_estimate(
	std::vector<float4>& vertices_0,
	std::vector<float4>& vertices_1,
	std::vector<float4>& normals,
	std::vector<ushort2>& indices,
	std::vector<float>& distances,
	const std::vector<ushort>& depth_buffer_0,
	const std::vector<ushort>& depth_buffer_1,
	uint width,
	uint height,
	ushort max_depth,
	const ushort half_window_search_size)
{
	StopWatchInterface *kernel_timer = nullptr;

	ushort* h_depth_buffer_0 = (ushort*)depth_buffer_0.data();
	ushort* h_depth_buffer_1 = (ushort*)depth_buffer_1.data();

	size_t in_pitch, out_pitch, index_pitch, distance_pitch;

	
	//
	// copy depth buffer data to array
	// 
	ushort* d_depth_buffer_0 = nullptr;
	checkCudaErrors(cudaMallocPitch(&d_depth_buffer_0, &in_pitch, sizeof(ushort) * width, height));
	checkCudaErrors(cudaMemcpy2D(
		d_depth_buffer_0,
		in_pitch,
		h_depth_buffer_0,
		sizeof(ushort) * width,
		sizeof(ushort) * width,
		height,
		cudaMemcpyHostToDevice));

	ushort* d_depth_buffer_1 = nullptr;
	checkCudaErrors(cudaMallocPitch(&d_depth_buffer_1, &in_pitch, sizeof(ushort) * width, height));
	checkCudaErrors(cudaMemcpy2D(
		d_depth_buffer_1,
		in_pitch,
		h_depth_buffer_1,
		sizeof(ushort) * width,
		sizeof(ushort) * width,
		height,
		cudaMemcpyHostToDevice));


	//
	// allocate vertex buffer in gpu
	// 
	float4* d_vertex_buffer_0;
	checkCudaErrors(cudaMallocPitch(
		&d_vertex_buffer_0,
		&out_pitch,
		width * sizeof(float4),
		height));

	float4* d_vertex_buffer_1;
	checkCudaErrors(cudaMallocPitch(
		&d_vertex_buffer_1,
		&out_pitch,
		width * sizeof(float4),
		height));

	//
	//  allocate normal buffer in gpu
	//  
	float4* d_normal_buffer;
	checkCudaErrors(cudaMallocPitch(
		&d_normal_buffer,
		&out_pitch,
		width * sizeof(float4),
		height));


	sdkCreateTimer(&kernel_timer);
	sdkStartTimer(&kernel_timer);

	Eigen::Matrix4f h_inverse_projection = perspective_matrix_inverse<float>(fov_y, aspect_ratio, near_plane, far_plane);
	back_projection_with_normal_estimation(d_vertex_buffer_0, d_normal_buffer, d_depth_buffer_0, width, height, max_depth, in_pitch, out_pitch, h_inverse_projection.data());
	back_projection_with_normal_estimation(d_vertex_buffer_1, d_normal_buffer, d_depth_buffer_1, width, height, max_depth, in_pitch, out_pitch, h_inverse_projection.data());


	//
	// allocate index buffer in gpu
	// 
	ushort2* d_index_buffer;
	checkCudaErrors(cudaMallocPitch(
		&d_index_buffer,
		&index_pitch,
		width * sizeof(ushort2),
		height));

	//
	// allocate index buffer in gpu
	// 
	float* d_distances_buffer;
	checkCudaErrors(cudaMallocPitch(
		&d_distances_buffer,
		&distance_pitch,
		width * sizeof(float),
		height));

	icp_matching_vertices(d_index_buffer, d_distances_buffer, d_vertex_buffer_0, d_vertex_buffer_1, width, height, out_pitch, index_pitch, half_window_search_size);
	

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&kernel_timer);
	std::cout << "Kernel Timer                              : " << kernel_timer->getTime() << " msec" << std::endl;
	sdkDeleteTimer(&kernel_timer);

	vertices_0.resize(depth_buffer_0.size());
	vertices_1.resize(depth_buffer_1.size());
	normals.resize(depth_buffer_0.size());
	indices.resize(depth_buffer_0.size());
	distances.resize(depth_buffer_0.size());


	cudaMemcpy2D(
		vertices_0.data(),
		sizeof(float4) * width,
		d_vertex_buffer_0,
		out_pitch,
		sizeof(float4) * width,
		height,
		cudaMemcpyDeviceToHost);

	cudaMemcpy2D(
		vertices_1.data(),
		sizeof(float4) * width,
		d_vertex_buffer_1,
		out_pitch,
		sizeof(float4) * width,
		height,
		cudaMemcpyDeviceToHost);

	cudaMemcpy2D(
		normals.data(),
		sizeof(float4) * width,
		d_normal_buffer,
		out_pitch,
		sizeof(float4) * width,
		height,
		cudaMemcpyDeviceToHost);


	cudaMemcpy2D(
		indices.data(),
		sizeof(ushort2) * width,
		d_index_buffer,
		index_pitch,
		sizeof(ushort2) * width,
		height,
		cudaMemcpyDeviceToHost);

	cudaMemcpy2D(
		distances.data(),
		sizeof(float) * width,
		d_distances_buffer,
		distance_pitch,
		sizeof(float) * width,
		height,
		cudaMemcpyDeviceToHost);
	

	for (int i = 0; i < width * height; ++i)
	{
		ushort2 index = indices[i];
		int ii = index.y * width + index.x;
		//std::cout << index.x << ", " << index.y << std::endl;

		if (ii > width * height)
		{
			std::cout << ii << " ---> " << width * height << std::endl;
		}
	}


	checkCudaErrors(cudaFree(d_depth_buffer_0));
	checkCudaErrors(cudaFree(d_depth_buffer_1));
	checkCudaErrors(cudaFree(d_vertex_buffer_0));
	checkCudaErrors(cudaFree(d_vertex_buffer_1));
	checkCudaErrors(cudaFree(d_normal_buffer));
	checkCudaErrors(cudaFree(d_index_buffer));
	checkCudaErrors(cudaFree(d_distances_buffer));
}


void convertKinectFrame2QImage(const KinectFrame& frame, QImage& depthImage)
{
	QVector<QRgb> colorTable;
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));
	depthImage = QImage(frame.depth_width(), frame.depth_height(), QImage::Format::Format_Indexed8);
	depthImage.setColorTable(colorTable);

	// set pixels to depth image
	for (int y = 0; y < depthImage.height(); y++)
	{
		for (int x = 0; x < depthImage.width(); x++)
		{
			const unsigned short depth = frame.depth[y * frame.depth_width() + x];
			depthImage.scanLine(y)[x] = static_cast<uchar>((float)depth / (float)frame.depth_max_distance() * 255.f);;
		}
	}
}

template<typename Type>
void cpu_test(const KinectFrame frame_0, const KinectFrame& frame_1)//, std::vector<Eigen::Vector3f>& vertices_0, std::vector<Eigen::Vector3f>& vertices_1)
{
	Type window_width = 640.0f;
	Type window_height = 480.0f;
	Type near_plane = 0.1f;
	Type far_plane = 10240.0f;
	Type fovy = 60.0f;
	Type aspect_ratio = window_width / window_height;

	Eigen::Matrix<Type, 4, 4> proj_inv = perspective_matrix_inverse(fovy, aspect_ratio, near_plane, far_plane);

	std::vector<Eigen::Matrix<Type, 3, 1>> vertices_0;
	std::vector<Eigen::Matrix<Type, 3, 1>> vertices_1;
	vertices_0.clear();
	vertices_1.clear();

	int d = 0;
	for (int y = 0; y < frame_0.depth_height(); ++y)
	{
		for (int x = 0; x < frame_0.depth_width(); ++x)
		{
			Eigen::Matrix<Type, 3, 1> v = window_coord_to_3d(Eigen::Matrix<Type, 2, 1>(x, y), (Type)frame_0.depth.at(d), proj_inv, (int)window_width, (int)window_height);
			++d;

			vertices_0.push_back(v);
		}
	}

	d = 0;
	for (int y = 0; y < frame_1.depth_height(); ++y)
	{
		for (int x = 0; x < frame_1.depth_width(); ++x)
		{
			Eigen::Matrix<Type, 3, 1> v = window_coord_to_3d(Eigen::Matrix<Type, 2, 1>(x, y), (Type)frame_1.depth.at(d), proj_inv, (int)window_width, (int)window_height);
			++d;

			vertices_1.push_back(v);
		}
	}

	Eigen::Matrix<Type, 3, 3> R = Eigen::Matrix<Type, 3, 3>::Zero();
	Eigen::Matrix<Type, 3, 1> t = Eigen::Matrix<Type, 3, 1>::Zero();

	ComputeRigidTransform(vertices_0, vertices_1, R, t);

	std::cout << std::fixed
		<< "ComputeRigidTransform: " << std::endl
		<< "Rotate" << std::endl << R << std::endl
		<< "Translate" << std::endl << t.transpose() << std::endl
		<< std::endl;
}

int main(int argc, char **argv)
{
	srand(time(NULL));

	std::string filename_0 = "../../data/knt_frames/frame_27.knt";
	std::string filename_1 = "../../data/knt_frames/frame_27.knt";
	ushort half_window_search_size = 3;
	ushort max_distance = 10;

	if (argc < 3)
	{
		std::cerr << "Usage: ICP_gpu.exe ../../data/knt_frames/frame_27.knt ../../data/knt_frames/frame_27.knt 3 10"
		<< std::endl
		<< "The app will continue with default parameters."
		<< std::endl;
	}
	else
	{
		filename_0 = argv[1];
		filename_1 = argv[2];
	}
	
	if (argc > 3)
		half_window_search_size = atoi(argv[3]);

	if (argc > 4)
		max_distance = atoi(argv[4]);

	Timer timer;

	timer.start();
	KinectFrame frame_0, frame_1;
	QKinectIO::loadFrame(QString::fromStdString(filename_0), frame_0);
	QKinectIO::loadFrame(QString::fromStdString(filename_1), frame_1);
	timer.print_interval("Importing kinect frames (.knt)            : ");

	//std::vector<Eigen::Vector3f> verts_0, verts_1;
	//cpu_test<float>(frame_0, frame_1);
	//return 0;


	
	std::vector<float4> vertices_0, vertices_1, normals;
	std::vector<ushort2> indices;
	std::vector<float> distances;
	std::vector<Eigen::Vector3f> vertices_00, vertices_11;

	timer.start();
	run_icp_with_normal_estimate(
		vertices_0, 
		vertices_1, 
		normals, 
		indices, 
		distances,
		frame_0.depth, 
		frame_1.depth,
		frame_0.depth_width(), 
		frame_0.depth_height(), 
		frame_0.depth_max_distance(), 
		half_window_search_size);
	timer.print_interval("Running normal estimate in GPU            : ");


	vertices_1.clear();
	for (const ushort2 index : indices)
	{
		float4 v = vertices_0.at(index.y * frame_0.depth_width() + index.x);
		vertices_1.push_back(v);
	}


	//for (int i = 0; i < vertices_0.size(); ++i)
	//{
	//	const float4& v0 = vertices_0.at(i);
	//	const float4& v1 = vertices_1.at(i);

	//	if (v0.z > 0.1 && v1.z > 0.1)
	//	{
	//		vertices_00.push_back(Eigen::Vector3f((Eigen::Vector4f(v0.x, v0.y, v0.z, v0.w) / v0.w).head<3>()));
	//		vertices_11.push_back(Eigen::Vector3f((Eigen::Vector4f(v1.x, v1.y, v1.z, v1.w) / v1.w).head<3>()));
	//	}
	//}


	for (int y = 0; y < frame_0.depth_height(); ++y)
	{
		for (int x = 0; x < frame_0.depth_width(); ++x)
		{
			int i = y * frame_0.depth_width() + x;

			float distance = distances[i];

			if (distance > max_distance)
				continue;

			ushort2 index = indices[i];
			int ii = index.y * frame_0.depth_width() + index.x;

			const float4& v0 = vertices_0.at(ii);
			const float4& v1 = vertices_1.at(ii);

			vertices_00.push_back(Eigen::Vector3f((Eigen::Vector4f(v0.x, v0.y, v0.z, v0.w) / v0.w).head<3>()));
			vertices_11.push_back(Eigen::Vector3f((Eigen::Vector4f(v1.x, v1.y, v1.z, v1.w) / v1.w).head<3>()));

		}

	}


	std::cout << "Vertices size " << vertices_0.size() << ", " << vertices_1.size() << std::endl;
	std::cout << "Vertices size " << vertices_00.size() << ", " << vertices_11.size() << std::endl;


	Eigen::Matrix<float, 3, 3> R = Eigen::Matrix<float, 3, 3>::Zero();
	Eigen::Matrix<float, 3, 1> t = Eigen::Matrix<float, 3, 1>::Zero();

	ComputeRigidTransform(vertices_00, vertices_11, R, t);

	std::cout << std::fixed
		<< "ComputeRigidTransform: " << std::endl
		<< "Rotate" << std::endl << R << std::endl
		<< "Translate" << std::endl << t.transpose() << std::endl
		<< std::endl;


	return 0;

	//
	// Viewer
	//
	QApplication app(argc, argv);

	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	QSurfaceFormat::setDefaultFormat(format);

	GLPointCloudViewer glwidget;
	glwidget.resize(1024, 848);
	glwidget.move(0, 0);
	glwidget.setWindowTitle("Point Cloud");
	glwidget.show();

	

	
	std::shared_ptr<GLPointCloud> cloud_0(new GLPointCloud);
	cloud_0->initGL();
	//cloud_0->setVertices((float*)&vertices_0.data()[0], (uint)vertices_0.size(), 4);
	cloud_0->setVertices((float*)&vertices_00.data()[0], (uint)vertices_00.size(), 4);
	cloud_0->setColor(QVector3D(1, 0, 0));
	
	std::shared_ptr<GLPointCloud> cloud_1(new GLPointCloud);
	cloud_1->initGL();
	//cloud_1->setVertices((float*)&vertices_1.data()[0], (uint)vertices_1.size(), 4);
	cloud_1->setVertices((float*)&vertices_11.data()[0], (uint)vertices_11.size(), 4);
	cloud_1->setColor(QVector3D(0, 0, 1));

	glwidget.addPointCloud(cloud_0);
	glwidget.addPointCloud(cloud_1);

	glwidget.setWeelSpeed(0.1f);
	glwidget.setPosition(0, 0, -0.5f);

	
	

	return app.exec();
}
