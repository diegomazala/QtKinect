
#include <QApplication>
#include <QKeyEvent>
#include <QBuffer>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include "GLKinectWidget.h"
#include <iostream>
#include <iterator>
#include <array>

#include "Volumetric_helper.h"
#include "Timer.h"
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_kernels/cuda_kernels.h"

#include "GLBaseWidget.h"
#include "GLPointCloud.h"

#include "helper_cuda.h"
#include "helper_image.h"





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



static void export_obj_eigen3(const std::string& filename, const std::vector<Eigen::Vector3f>& vertices)
{
	std::ofstream file;
	file.open(filename);
	for (const Eigen::Vector3f v : vertices)
	{

		int z = v.z();
		if ((v.x() + v.y() + v.z()) > 1 && z != 0)
			file << std::fixed << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}
	file.close();
}




template <typename InputPixelType, typename OutputPixelType>
void run_bilateral_filter_depth_buffer(
	std::vector<OutputPixelType>& output_buffer,
	const std::vector<InputPixelType>& input_buffer,
	uint width,
	uint height,
	ushort max_depth,
	float gaussian_delta,
	float euclidean_delta,
	int filter_radius,
	int iterations)
{
	updateGaussian(gaussian_delta, filter_radius);

	StopWatchInterface *kernel_timer = nullptr;

	InputPixelType* hImage = (InputPixelType*)input_buffer.data();


	size_t in_pitch, out_pitch;

	InputPixelType* dInputImage = nullptr;
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&dInputImage, &in_pitch, sizeof(InputPixelType) * width, height));
	checkCudaErrors(cudaMemcpy2D(
		dInputImage,
		in_pitch,
		hImage,
		sizeof(InputPixelType) * width,
		sizeof(InputPixelType) * width,
		height,
		cudaMemcpyHostToDevice));



	OutputPixelType* dOutputImage;
	checkCudaErrors(cudaMallocPitch(
		&dOutputImage,
		&out_pitch,
		width * sizeof(OutputPixelType),
		height));


	sdkCreateTimer(&kernel_timer);
	sdkStartTimer(&kernel_timer);


	bilateralFilter_normal_estimate_float4((OutputPixelType*)dOutputImage, (InputPixelType*)dInputImage, width, height, in_pitch, out_pitch, max_depth, euclidean_delta, filter_radius, iterations, kernel_timer);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&kernel_timer);
	std::cout << "Kernel Timer                              : " << kernel_timer->getTime() << " msec" << std::endl;
	sdkDeleteTimer(&kernel_timer);

	output_buffer.resize(input_buffer.size());

	cudaMemcpy2D(
		output_buffer.data(),
		sizeof(OutputPixelType) * width,
		dOutputImage,
		out_pitch,
		sizeof(OutputPixelType) * width,
		height,
		cudaMemcpyDeviceToHost);


	checkCudaErrors(cudaFree(dInputImage));
	checkCudaErrors(cudaFree(dOutputImage));
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
	std::cout << "\nproj:\n" << h_inverse_projection << std::endl;

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

template <typename PixelType>
void convertDepthBuffer2QImage(const std::vector<PixelType>& depth_buffer, uint width, uint height, ushort depth_max_distance, QImage& depthImage)
{
	assert(depth_buffer.size() == width * height);

	QVector<QRgb> colorTable;
	for (int i = 0; i < 256; ++i)
		colorTable.push_back(qRgb(i, i, i));
	depthImage = QImage(width, height, QImage::Format::Format_Indexed8);
	depthImage.setColorTable(colorTable);

	// set pixels to depth image
	for (int y = 0; y < depthImage.height(); y++)
	{
		for (int x = 0; x < depthImage.width(); x++)
		{
			const float depth = static_cast<float>(depth_buffer[y * width + x]);
			depthImage.scanLine(y)[x] = static_cast<uchar>(depth / static_cast<float>(depth_max_distance) * 255.f);;
		}
	}
}


void convertDepthBuffer2QImageFloat4(const std::vector<float4>& depth_buffer, uint width, uint height, ushort depth_max_distance, QImage& depthImage)
{
	assert(depth_buffer.size() == width * height);

	depthImage = QImage(width, height, QImage::Format::Format_ARGB32);

	// set pixels to depth image
	for (int y = 0; y < depthImage.height(); y++)
	{
		for (int x = 0; x < depthImage.width(); x++)
		{
			const float4 depth = static_cast<float4>(depth_buffer[y * width + x]);
			depthImage.setPixel(x, y, qRgb(
				static_cast<uchar>(depth.x / static_cast<float>(depth_max_distance)* 255.f),
				static_cast<uchar>(depth.y / static_cast<float>(depth_max_distance)* 255.f),
				static_cast<uchar>(depth.z / static_cast<float>(depth_max_distance)* 255.f)	));
		}
	}
}



int main(int argc, char **argv)
{
	if (argc < 6)
	{
		std::cerr << "Usage: BilateralFilter_gpu.exe ../../data/room.knt number_of_iterations gaussian_delta euclidean_delta filter_radius" << std::endl;
		std::cerr << "Usage: BilateralFilter_gpu.exe ../../data/room.knt 10 4.0 0.1 5" << std::endl;
		return EXIT_FAILURE;
	}

	QApplication app(argc, argv);
	app.setApplicationName("Bilateral Filter");


	int iterations = atoi(argv[2]);
	float gaussian_delta = atof(argv[3]);
	float euclidean_delta = atof(argv[4]);
	int filter_radius = atoi(argv[5]);

	std::string filename = argv[1];
	Timer timer;



	timer.start();
	KinectFrame frame;
	QKinectIO::loadFrame(filename, frame);


	//std::vector<Eigen::Vector3f> verts;
	//Eigen::Matrix4f h_inverse_projection = perspective_matrix_inverse<float>(fov_y, aspect_ratio, near_plane, far_plane);
	//for (int y = 0; y < frame.depth_height(); y++)
	//{
	//	for (int x = 0; x < frame.depth_width(); x++)
	//	{
	//		const float depth = static_cast<float>(frame.depth[y * frame.depth_width() + x]);
	//		Eigen::Vector3f v = window_coord_to_3d(Eigen::Vector2f(x, y), depth, h_inverse_projection, frame.depth_width(), frame.depth_height());
	//		verts.push_back(v);
	//	}
	//}
	//export_obj_eigen3("../../data/eigen_vertices.obj", verts);

	/////////////////////////////////////
	//
	// create depth image
	QVector<QRgb>		colorTable;
	QImage				depthImage;
	QImage				depthImageFiltered;

	Timer t;
	t.start();
	convertKinectFrame2QImage(frame, depthImage);
	t.print_interval("Converting original depth buffer to image : ");


	QImageWidget inputWidget;
	inputWidget.setImage(depthImage);
	inputWidget.move(0, 0);
	inputWidget.setWindowTitle("Original Depth");
	inputWidget.show();

	t.start();
	std::vector<float4> vertices, normals;
	//std::vector<float4> depth_buffer_filtered;
	//run_bilateral_filter_depth_buffer<ushort, float4>(depth_buffer_filtered, frame.depth, frame.depth_width(), frame.depth_height(), frame.depth_max_distance(), gaussian_delta, euclidean_delta, filter_radius, iterations);
	run_back_projection_with_normal_estimate(vertices, normals, frame.depth, frame.depth_width(), frame.depth_height(), frame.depth_max_distance());
	t.print_interval("Running bilateral filter in GPU           : ");


	t.start();
	export_obj_float4("../../data/float4_vertices.obj", vertices, normals);
	t.print_interval("Exporting obj                             : ");

	//for (int i = 0; i < depth_buffer_filtered.size(); ++i)
	//	depth_buffer_filtered[i] = 8192;


	//t.start();
	//////convertDepthBuffer2QImage(depth_buffer_filtered, frame.depth_width(), frame.depth_height(), frame.depth_max_distance(), depthImageFiltered);
	//convertDepthBuffer2QImageFloat4(vertices, frame.depth_width(), frame.depth_height(), frame.depth_max_distance(), depthImageFiltered);
	//t.print_interval("Converting filtered depth buffer to image : ");

	//QString str = "Result_Bilateral_" + QString::number(iterations) + "_" + QString::number(gaussian_delta) + "_" + QString::number(euclidean_delta) + "_" + QString::number(filter_radius);

	//QImageWidget outputWidget;
	//outputWidget.setImage(depthImageFiltered);
	//outputWidget.move(inputWidget.width(), 0);
	//outputWidget.setWindowTitle(str);
	//outputWidget.show();


	//depthImageFiltered.save(str + ".png");


	return app.exec();
}
