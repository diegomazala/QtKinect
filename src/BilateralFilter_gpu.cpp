
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



StopWatchInterface *timer = NULL;
StopWatchInterface *kernel_timer = NULL;




template<typename PixelType>
void run_bilateral_filter(
	QImage& outputImage, 
	const QImage& inputImage, 
	float gaussian_delta, 
	float euclidean_delta, 
	int filter_radius,
	int iterations)
{
	updateGaussian(gaussian_delta, filter_radius);

	StopWatchInterface *kernel_timer = nullptr;

	PixelType* hImage = (PixelType*)(inputImage.bits());
	unsigned int width = inputImage.width();
	unsigned int height = inputImage.height();

	size_t pitch;

	PixelType* dInputImage = nullptr;
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&dInputImage, &pitch, sizeof(PixelType) * width, height));
	checkCudaErrors(cudaMemcpy2D(
		dInputImage,
		pitch,
		hImage,
		sizeof(PixelType) * width,
		sizeof(PixelType) * width,
		height,
		cudaMemcpyHostToDevice));



	PixelType* dOutputImage;
	checkCudaErrors(cudaMallocPitch(
		&dOutputImage,
		&pitch,
		width * sizeof(PixelType),
		height));


	sdkCreateTimer(&kernel_timer);

	if (sizeof(PixelType) == 1)
		bilateralFilterGray((uchar*)dOutputImage, (uchar*)dInputImage, width, height, pitch, euclidean_delta, filter_radius, iterations, kernel_timer);
	else
		bilateralFilterRGBA((uint*)dOutputImage, width, height, euclidean_delta, filter_radius, iterations, kernel_timer);

	sdkDeleteTimer(&kernel_timer);

	cudaMemcpy2D(
		outputImage.bits(),
		sizeof(PixelType) * width,
		dOutputImage,
		pitch,
		sizeof(PixelType) * width,
		height,
		cudaMemcpyDeviceToHost);


	checkCudaErrors(cudaFree(dInputImage));
	checkCudaErrors(cudaFree(dOutputImage));
}



template<typename PixelType>
void run_bilateral_filter_depth_buffer(
	std::vector<PixelType>& output_buffer,
	const std::vector<PixelType>& input_buffer,
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

	PixelType* hImage = (PixelType*)input_buffer.data();


	size_t pitch;

	PixelType* dInputImage = nullptr;
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&dInputImage, &pitch, sizeof(PixelType) * width, height));
	checkCudaErrors(cudaMemcpy2D(
		dInputImage,
		pitch,
		hImage,
		sizeof(PixelType) * width,
		sizeof(PixelType) * width,
		height,
		cudaMemcpyHostToDevice));



	PixelType* dOutputImage;
	checkCudaErrors(cudaMallocPitch(
		&dOutputImage,
		&pitch,
		width * sizeof(PixelType),
		height));


	sdkCreateTimer(&kernel_timer);
	sdkStartTimer(&kernel_timer);

	
	bilateralFilter_ushort((ushort*)dOutputImage, (ushort*)dInputImage, width, height, pitch, max_depth, euclidean_delta, filter_radius, iterations, kernel_timer);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&kernel_timer);
	std::cout << "Kernel Timer                              : " << kernel_timer->getTime() << " msec" << std::endl;
	sdkDeleteTimer(&kernel_timer);

	output_buffer.resize(input_buffer.size());

	cudaMemcpy2D(
		output_buffer.data(),
		sizeof(PixelType) * width,
		dOutputImage,
		pitch,
		sizeof(PixelType) * width,
		height,
		cudaMemcpyDeviceToHost);


	checkCudaErrors(cudaFree(dInputImage));
	checkCudaErrors(cudaFree(dOutputImage));
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

void convertDepthBuffer2QImage(const std::vector<ushort>& depth_buffer, uint width, uint height, ushort depth_max_distance, QImage& depthImage)
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
			const unsigned short depth = depth_buffer[y * width + x];
			depthImage.scanLine(y)[x] = static_cast<uchar>((float)depth / (float)depth_max_distance * 255.f);;
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


#if 0
	QImage inputImage, outputImage;
	if (!inputImage.load(filename.c_str()))
	{
		std::cout << "Error: Could not load file image: " << filename << std::endl;
		return EXIT_FAILURE;
	}

	outputImage = inputImage;

	run_bilateral_filter<unsigned char>(outputImage, inputImage, gaussian_delta, euclidean_delta, filter_radius, iterations);


	QImageWidget inputWidget;
	inputWidget.setImage(inputImage);
	inputWidget.move(0, 0);
	inputWidget.setWindowTitle("Input");
	inputWidget.show();


	QImageWidget outputWidget;
	outputWidget.setImage(outputImage);
	outputWidget.move(inputWidget.width(), 0);
	outputWidget.setWindowTitle("Output");
	outputWidget.show();
#else

	timer.start();
	KinectFrame frame;
	QKinectIO::loadFrame(QString::fromStdString(filename), frame);

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
	std::vector<ushort> depth_buffer_filtered;
	run_bilateral_filter_depth_buffer<ushort>(depth_buffer_filtered, frame.depth, frame.depth_width(), frame.depth_height(), frame.depth_max_distance(), gaussian_delta, euclidean_delta, filter_radius, iterations);
	t.print_interval("Running bilateral filter in GPU           : ");

	t.start();
	convertDepthBuffer2QImage(depth_buffer_filtered, frame.depth_width(), frame.depth_height(), frame.depth_max_distance(), depthImageFiltered);
	t.print_interval("Converting filtered depth buffer to image : ");

	t.start();
	frame.depth = depth_buffer_filtered;
	t.print_interval("Copying depth buffer filtered             : ");

	QString str = "_bilateral_filter_" + QString::number(iterations) + "_" + QString::number(gaussian_delta) + "_" + QString::number(euclidean_delta) + "_" + QString::number(filter_radius);
	
	QImageWidget outputWidget;
	outputWidget.setImage(depthImageFiltered);
	outputWidget.move(inputWidget.width(), 0);
	outputWidget.setWindowTitle(str);
	outputWidget.show();
#endif

	t.start();
	depthImageFiltered.save(QFileInfo(filename.c_str()).absolutePath() + '/' + QFileInfo(filename.c_str()).baseName() + str + ".png");
	t.print_interval("Saving image result                       : ");

	t.start();
	QString frame_filename = QFileInfo(filename.c_str()).absolutePath() + '/' + QFileInfo(filename.c_str()).baseName() + str + ".knt";
	QKinectIO::saveFrame(frame_filename, frame);
	t.print_interval("Saving frame result                       : ");
	
	return app.exec();
}
