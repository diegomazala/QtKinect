#include <QApplication>
#include "GLModelViewer.h"
#include "GLModel.h"
#include "GLShaderProgram.h"

#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "Interpolator.hpp"
#include "RayBox.h"
#include "Grid.h"
#include "Timer.h"
#include "TsdfGridIO.h"
#include "ObjFile.h"

#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_kernels/cuda_kernels.h"
#include "helper_cuda.h"
#include "helper_image.h"

typedef float Decimal;

StopWatchInterface *cuda_timer = 0;
float invViewMatrix[12];

dim3 blockSize(16, 16);
dim3 gridSize;
uint cuda_width = 512, cuda_height = 512;

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;


const Decimal window_width = 512.0f;
const Decimal window_height = 424.0f;

template<typename Type>
static std::vector<int> raycast_all(
	const Eigen::Matrix<Type, 3, 1>& origin,
	const Eigen::Matrix<Type, 3, 1>& direction,
	const Type ray_near,
	const Type ray_far,
	Eigen::Matrix<Type, 3, 1> voxel_size,
	const std::vector<Eigen::Matrix<Type, 4, 1>>& vertices)
{
	Eigen::Matrix<Type, 4, 1> half_voxel(voxel_size.x() * 0.5, voxel_size.y() * 0.5, voxel_size.z() * 0.5, 1.0);

	std::vector<int> intersections;

	int voxel_index = 0;
	for (const auto v : vertices)
	{
		//Eigen::Vector3d corner_min = (transformation * (v.point - half_voxel).homogeneous()).head<3>();
		//Eigen::Vector3d corner_max = (transformation * (v.point + half_voxel).homogeneous()).head<3>();
		Eigen::Matrix<Type, 3, 1> corner_min = (v - half_voxel).head<3>();
		Eigen::Matrix<Type, 3, 1> corner_max = (v + half_voxel).head<3>();

		Box<Type> box(corner_min, corner_max);
		Ray<Type> ray(origin, direction);

		if (box.intersect(ray, ray_near, ray_far))
			intersections.push_back(voxel_index);

		++voxel_index;
	}

	return intersections;
}


template<typename Type>
void create_grid(const Eigen::Matrix<Type, 3, 1>& volume_size, const Eigen::Matrix<Type, 3, 1>& voxel_size, const Eigen::Matrix<Type, 4, 4>& transformation, std::vector<Eigen::Matrix<Type, 4, 1>>& vertices)
{
	Eigen::Vector3i voxel_count = Eigen::Vector3i(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	vertices.resize((voxel_count.x() + 1) * (voxel_count.y() + 1) * (voxel_count.z() + 1));

	Eigen::Matrix<Type, 4, 4> to_origin = Eigen::Matrix<Type, 4, 4>::Identity();
	to_origin.col(3) << -(volume_size.x() / 2.0), -(volume_size.y() / 2.0), -(volume_size.z() / 2.0), 1.0;	// set translate

	int i = 0;
	for (int z = 0; z <= volume_size.z(); z += voxel_size.z())
	{
		for (int y = 0; y <= volume_size.y(); y += voxel_size.y())
		{
			for (int x = 0; x <= volume_size.x(); x += voxel_size.x(), i++)
			{
				Eigen::Matrix<Type, 4, 1> p = transformation * to_origin * Eigen::Matrix<Type, 4, 1>(x, y, z, 1);
				p /= p.w();
				vertices[i] = p;
			}
		}
	}
}


template <typename Type>
void cpu_raycast(
	const Eigen::Matrix<Type, 3, 1>& volume_size,
	const Eigen::Matrix<Type, 3, 1>& voxel_size,
	const Eigen::Matrix<Type, 3, 1>& origin,
	const Eigen::Matrix<Type, 3, 1>& target,
	const std::vector<Eigen::Matrix<Type, 4, 1>>& vertices)
{
	std::vector<int> intersections;
	Eigen::Matrix<Type, 3, 1> direction = (target - origin).normalized();

	Type ray_near = 0; // atof(argv[1]);
	Type ray_far = 100; // atof(argv[2]);

	Timer timer;
	timer.start();
	intersections = raycast_all(origin, direction, ray_near, ray_far, voxel_size, vertices);
	std::cout << "\nIntersections All: " << intersections.size() << '\t' << timer.diff_msec() << " msec" << std::endl;
	for (const auto i : intersections)
		std::cout << i << ' ';
	std::cout << std::endl;
}



//// Load raw data from disk
//void saveRawFile(char *filename, size_t size, void* data)
//{
//	FILE *fp = fopen(filename, "wb");
//
//	if (!fp)
//	{
//		fprintf(stderr, "Error opening file '%s'\n", filename);
//		return;
//	}
//
//	fwrite(data, 1, size, fp);
//	fclose(fp);
//}
//
//void save_raw_file(const std::string& filename, const std::size_t size, const void* data)
//{
//	std::ofstream file(filename, std::ios::binary);
//	if (file.is_open())
//	{
//		file.write(reinterpret_cast<const char*>(data), size);
//		file.close();
//	}
//}

void export_sphere_volume()
{
	std::vector<float> raw;

	std::ofstream file;
	file.open("../../data/sphere_float2_32.txt");
	int v = 0;
	int s = 32;
	Eigen::Vector3i center(16, 16, 16);
	float max_distance = (Eigen::Vector3i(s, s, s) - center).norm() * 0.5f;
	for (int z = 0; z < s; ++z)
	{
		for (int y = 0; y < s; ++y)
		{
			for (int x = 0; x < s; ++x)
			{
				Eigen::Vector3i vec(x, y, z);
				float distance = (vec - center).norm();

				float c = -1;

				//if (x > 8 && x < 24 && y > 8 && y < 24 && z == 16)
				if (distance > max_distance - 1 && distance < max_distance + 1)
					c = 255;

				//if (distance > max_distance)
				//	c = 0;
				//else
				//	c = static_cast<uchar>((distance / max_distance) * 255.f);

				raw.push_back(c);
				raw.push_back(c);
				file << (int)c << ' ';
			}
			file << std::endl;
		}
		file << std::endl;
	}
	file.close();

//	saveRawFile("../../data/sphere_float_32.raw", sizeof(float) * raw.size(), &raw[0]);
}

void export_monkey_volume()
{
	std::vector<float> raw;

	std::ofstream file;
	file.open("../../data/monkey_float2_32.txt");
	int v = 0;
	int s = 32;
	Eigen::Vector3i center(16, 16, 16);
	float max_distance = (Eigen::Vector3i(s, s, s) - center).norm() * 0.5f;
	for (int z = 0; z < s; ++z)
	{
		for (int y = 0; y < s; ++y)
		{
			for (int x = 0; x < s; ++x)
			{
				Eigen::Vector3i vec(x, y, z);
				float distance = (vec - center).norm();

				float c = -1;

				//if (x > 8 && x < 24 && y > 8 && y < 24 && z == 16)
				if (distance > max_distance - 1 && distance < max_distance + 1)
					c = 255;

				//if (distance > max_distance)
				//	c = 0;
				//else
				//	c = static_cast<uchar>((distance / max_distance) * 255.f);

				raw.push_back(c);
				raw.push_back(c);
				file << (int)c << ' ';
			}
			file << std::endl;
		}
		file << std::endl;
	}
	file.close();

//	saveRawFile("../../data/sphere_float_32.raw", sizeof(float) * raw.size(), &raw[0]);
}




bool runSingleTest()
{
	bool bTestResult = true;

	uint *d_output;
	checkCudaErrors(cudaMalloc((void **)&d_output, cuda_width*cuda_height*sizeof(uint)));
	checkCudaErrors(cudaMemset(d_output, 0, cuda_width*cuda_height*sizeof(uint)));

	float modelView[16] =
	{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 4.0f, 1.0f
	};

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	// call CUDA kernel, writing results to PBO
	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

	gridSize = dim3(iDivUp(cuda_width, blockSize.x), iDivUp(cuda_height, blockSize.y));

	// Start timer 0 and process n loops on the GPU
	int nIter = 10;

	for (int i = -1; i < nIter; i++)
	{
		if (i == 0)
		{
			cudaDeviceSynchronize();
			sdkStartTimer(&cuda_timer);
		}

		render_volume(gridSize, blockSize, d_output, cuda_width, cuda_height, density, brightness, transferOffset, transferScale);
	}

	cudaDeviceSynchronize();
	sdkStopTimer(&cuda_timer);
	// Get elapsed time and throughput, then log to sample and master logs
	double dAvgTime = sdkGetTimerValue(&cuda_timer) / (nIter * 1000.0);
	printf("volumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n",
		(1.0e-6 * cuda_width * cuda_height) / dAvgTime, dAvgTime, (cuda_width * cuda_height), 1, blockSize.x * blockSize.y);


	getLastCudaError("Error: render_kernel() execution FAILED");
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned char *h_output = (unsigned char *)malloc(cuda_width*cuda_height * 4);
	checkCudaErrors(cudaMemcpy(h_output, d_output, cuda_width*cuda_height * 4, cudaMemcpyDeviceToHost));

	sdkSavePPM4ub("../../data/volume.ppm", h_output, cuda_width, cuda_height);

	cudaFree(d_output);
	free(h_output);

	sdkDeleteTimer(&cuda_timer);

	return bTestResult;
}




// Usage: ./Raycastingd.exe 2 1 0 2 -3 0.72 -1.2 2 7
int main(int argc, char **argv)
{
	//export_sphere_volume();
	//return 0;

	Timer timer;
	int vol_size = atoi(argv[1]); //16;
	int vx_size = atoi(argv[2]); //1;
	

	Eigen::Matrix<Decimal, 3, 1> volume_size(vol_size, vol_size, vol_size);
	Eigen::Matrix<Decimal, 3, 1> voxel_size(vx_size, vx_size, vx_size);
	std::vector<Eigen::Matrix<Decimal, 4, 1>> grid_vertices;
	
	Eigen::Transform<Decimal, 3, Eigen::Affine> grid_affine = Eigen::Transform<Decimal, 3, Eigen::Affine>::Identity();
	grid_affine.translate(Eigen::Matrix<Decimal, 3, 1>(0, 0, -256));
	grid_affine.scale(Eigen::Matrix<Decimal, 3, 1>(1, 1, -1));	// z is negative inside of screen

	Eigen::Matrix<Decimal, 4, 4> to_origin = Eigen::Matrix<Decimal, 4, 4>::Identity();
	to_origin.col(3) << -(volume_size.x() / 2.0), -(volume_size.y() / 2.0), -(volume_size.z() / 2.0), 1.0;	// set translate
	
	const Eigen::Vector3i voxel_count = Eigen::Vector3i(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	create_grid<Decimal>(volume_size, voxel_size, Eigen::Matrix<Decimal, 4, 4>::Identity(), grid_vertices);

	std::vector<int> intersections;
	Eigen::Matrix<Decimal, 3, 1> origin(atof(argv[3]), atof(argv[4]), atof(argv[5]));
	Eigen::Matrix<Decimal, 3, 1> target(atof(argv[6]), atof(argv[7]), atof(argv[8]));
	Eigen::Matrix<Decimal, 3, 1> direction = (target - origin).normalized();

#if 0
	Decimal ray_near = 0; // atof(argv[1]);
	Decimal ray_far = 100; // atof(argv[2]);

	timer.start();
	intersections = raycast_all(origin, direction, ray_near, ray_far, voxel_size, grid_vertices);
	std::cout << "\nIntersections All: " << intersections.size() << '\t' << timer.diff_msec() << " msec" << std::endl;
	for (const auto i : intersections)
		std::cout << i << ' ';

	cpu_raycast(volume_size, voxel_size, origin, target, grid_vertices);
#endif







	std::vector<Eigen::Matrix<Decimal, 2, 1>> grid_params;
	tsdf_grid_load("../../data/monkey_256_8_2_90.buff", volume_size, voxel_size, grid_params);

	to_origin = Eigen::Matrix<Decimal, 4, 4>::Identity();
	to_origin.col(3) << -(volume_size.x() / 2.0), -(volume_size.y() / 2.0), -(volume_size.z() / 2.0), 1.0;	// set translate

	QApplication app(argc, argv);


	//
	// setup opengl viewer
	// 
	GLModelViewer glwidget;
	glwidget.resize(640, 480);
	glwidget.setPerspective(60.0f, 0.1f, 1024.0f);
	glwidget.move(320, 0);
	glwidget.setWindowTitle("Point Cloud");
	glwidget.setWeelSpeed(0.1f);
	glwidget.setDistance(-0.5f);
	glwidget.show();


	create_grid<Decimal>(volume_size, voxel_size, Eigen::Matrix<Decimal, 4, 4>::Identity(), grid_vertices);

	std::vector<Eigen::Vector4f> vertices, colors;

	int i = 0;
	for (int z = 0; z <= volume_size.z(); z += voxel_size.z())
	{
		for (int y = 0; y <= volume_size.y(); y += voxel_size.y())
		{
			for (int x = 0; x <= volume_size.x(); x += voxel_size.x(), i++)
			{
				//const float tsdf = grid.data.at(i).tsdf;
				const float tsdf = grid_params.at(i)[0];

				Eigen::Matrix<Decimal, 4, 1> p = to_origin * Eigen::Matrix<Decimal, 4, 1>(x, y, z, 1);
				p /= p.w();

				if (tsdf > 0.1)
				{
					//Eigen::Matrix<Decimal, 4, 1> p = grid_affine.matrix() * to_origin * Eigen::Matrix<Decimal, 4, 1>(x, y, z, 1);
					vertices.push_back(p);
					colors.push_back(Eigen::Matrix<Decimal, 4, 1>(0, 1, 0, 1));
				}
				else if (tsdf < -0.1)
				{
					//Eigen::Matrix<Decimal, 4, 1> p = grid_affine.matrix() * to_origin * Eigen::Matrix<Decimal, 4, 1>(x, y, z, 1);
					vertices.push_back(p);
					colors.push_back(Eigen::Matrix<Decimal, 4, 1>(1, 0, 0, 1));
				}
				//else
				//{
				//	vertices.push_back(p);
				//	colors.push_back(Eigen::Matrix<Decimal, 4, 1>(1, 1, 1, 1));
				//}
			}
		}
	}


		//timer.start();
		//export_obj("../../data/vertices.obj", vertices);
		//export_obj("../../data/grid_vertices.obj", grid_vertices);
		//timer.print_interval("Exporting volume    : ");
		//return 0;
		
//	save_raw_file("../../data/monkey_tsdf_float2_33.raw", sizeof(float) * 2 * grid_params.size(), &grid_params[0][0]);
//	void* data_ptr = load_raw_file("../../data/monkey_tsdf_float2_33.raw", sizeof(float) * 2 * grid_params.size());
//	save_raw_file("../../data/monkey_tsdf_float2_33_2.raw", sizeof(float) * 2 * grid_params.size(), data_ptr);



	int v_size = volume_size.x() / voxel_size.x() + 1;
	cudaExtent cudaVolumeSize = make_cudaExtent(v_size, v_size, v_size);
	initCuda_render_volume(&grid_params[0][0], cudaVolumeSize);


	runSingleTest();
	
	freeCudaBuffers_render_volume();
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();





	//
	// setup model
	// 
	std::shared_ptr<GLModel> model(new GLModel);
	model->initGL();
	//model->setVertices(&grid_vertices[0][0], grid_vertices.size(), 4);
	model->setVertices(&vertices[0][0], vertices.size(), 4);
	model->setColors(&colors[0][0], colors.size(), 4);
	glwidget.addModel(model);


	//
	// setup kinect shader program
	// 
	std::shared_ptr<GLShaderProgram> kinectShaderProgram(new GLShaderProgram);
	if (kinectShaderProgram->build("color.vert", "color.frag"))
		model->setShaderProgram(kinectShaderProgram);










	//const ushort2 image_size = make_ushort2(640, 480);
	//std::vector<uchar3> pixel_buffer(image_size.x * image_size.y);


	//raycast_grid(
	//	reinterpret_cast<float4*>(&grid_vertices[0][0]),
	//	reinterpret_cast<float2*>(&grid_params[0][0]),
	//	volume_size.x(),
	//	voxel_size.x(),
	//	make_float3(origin.x(), origin.y(), origin.z()),
	//	make_float3(direction.x(), direction.y(), direction.z()),
	//	ray_near,
	//	ray_far,
	//	pixel_buffer.data(),
	//	image_size.x,
	//	image_size.y
	//	);

	//QImage image((uchar*)pixel_buffer.data(), image_size.x, image_size.y, QImage::Format_RGB888);
	////image.fill(Qt::white);
	//std::cout << "Saving to image..." << std::endl;
	//image.save("../../data/raycasting_gpu.png");

	

	

	return app.exec();
}
