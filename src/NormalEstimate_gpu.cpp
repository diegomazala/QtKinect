
#include <QApplication>
#include <QKeyEvent>
#include <QBuffer>
#include <QFileInfo>
#include "QImageWidget.h"
#include "QKinectGrabber.h"
#include "QKinectIO.h"
#include <iostream>
#include <iterator>
#include <array>

#include "Volumetric_helper.h"
#include "Timer.h"
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_kernels/cuda_kernels.h"

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




int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: NormalEstimate_gpu.exe ../../data/room.knt " << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = argv[1];
	Timer timer;

	timer.start();
	KinectFrame frame;
	QKinectIO::loadFrame(QString::fromStdString(filename), frame);
	timer.print_interval("Importing kinect frame (.knt)             : ");

	timer.start();
	std::vector<float4> vertices, normals;
	run_back_projection_with_normal_estimate(vertices, normals, frame.depth, frame.depth_width(), frame.depth_height(), frame.depth_max_distance());
	timer.print_interval("Running normal estimate in GPU            : ");



	timer.start();
	QString frame_filename = QFileInfo(filename.c_str()).absolutePath() + '/' + QFileInfo(filename.c_str()).fileName().remove(".knt") + "_normal_estimate.obj";
	export_obj_float4(frame_filename.toStdString(), vertices, normals);
	timer.print_interval("Exporting obj                             : ");

}
