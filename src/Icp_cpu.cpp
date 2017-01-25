
#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "Timer.h"
#include "Projection.h"
#include "Volumetric_helper.h"
#include "KinectFrame.h"
#include "KinectSpecs.h"
#include "ICP.h"

typedef double Decimal;





void compute_vertices(std::vector<Eigen::Matrix<Decimal, 4, 1>>& vertices, const KinectFrame& frame, Decimal fovy, Decimal aspect_ratio, Decimal near_plane, Decimal far_plane)
{
	Eigen::Matrix<Decimal, 3, 1> vert_uv, vert_u1v, vert_uv1;

	for (int x = 0; x < frame.depth_width() - 1; ++x)
	{
		for (int y = 0; y < frame.depth_height() - 1; ++y)
		{
			const Decimal depth = frame.depth[y * frame.depth_width() + x];

			if (depth > 0.01)
			{
				vert_uv = window_coord_to_3d(Eigen::Matrix<Decimal, 2, 1>(x, y), depth, fovy, aspect_ratio, near_plane, far_plane, frame.depth_width(), frame.depth_height());

				int i = y * frame.depth_width() + x;

#if 1
				vertices[i] = vert_uv.homogeneous();
#else

				if (!vert_uv.isZero(0.0001))
				{
					vertices.push_back(vert_uv.homogeneous());
				}
#endif
			}

		}
	}
}



// Usage: ./Icp.exe ../../data/frame_0.knt ../../data/frame_1.knt 0 10 3
int main(int argc, char **argv)
{

	if (argc < 3)
	{
		std::cerr
			<< "Missing parameters. Abort."
			<< std::endl
			<< "Usage: ./Icp.exe <knt_frame> <knt_frame> <filter_width> <max_distance_threshold> "
			<< "Usage: ./Icp.exe ../../data/frame_1.knt ../../data/frame_1.knt 10 3"
			<< std::endl;
		return EXIT_FAILURE;
	}


	Timer timer;
	const std::string input_filename[] = { argv[1], argv[2] };

	const int filter_width = (argc > 2) ? atoi(argv[3]) : 10;
	const Decimal max_distance = (Decimal)((argc > 3) ? atof(argv[4]) : 100);

	try
	{
		timer.start();
		KinectFrame frame[2] =
		{
			KinectFrame(input_filename[0]),
			KinectFrame(input_filename[1])
		};
		timer.print_interval("Importing frames          : ");
		

		if (frame[0].depth_width() != frame[1].depth_width() ||
			frame[0].depth_height() != frame[1].depth_height())
		{
			std::cout
				<< "Frame size doens not match: " << std::endl
				<< frame[0].depth_width() << ", " << frame[0].depth_height() << std::endl
				<< frame[1].depth_width() << ", " << frame[1].depth_height() << std::endl
				<< std::endl;
			return EXIT_FAILURE;
		}

		const uint16_t width = frame[0].depth_width();
		const uint16_t height = frame[0].depth_height();
		const uint32_t pixel_count = width * height;

		Decimal fovy = KINECT_V2_FOVY;
		Decimal aspect_ratio = KINECT_V2_DEPTH_ASPECT_RATIO;
		Decimal near_plane = KINECT_V2_DEPTH_MIN;
		Decimal far_plane = KINECT_V2_DEPTH_MAX;

		if (width != 512)	// it is not kinect version 2. It is version 1
		{
			fovy = KINECT_V1_FOVY;
			aspect_ratio = KINECT_V1_ASPECT_RATIO;
			near_plane = KINECT_V1_DEPTH_MIN;
			far_plane = KINECT_V1_DEPTH_MAX;
		}


		std::vector<Eigen::Matrix<Decimal, 4, 1>> vertices[2] = 
		{
			std::vector<Eigen::Matrix<Decimal, 4, 1>>(pixel_count, Eigen::Matrix<Decimal, 4, 1>(0, 0, 0, 0)), 
			std::vector<Eigen::Matrix<Decimal, 4, 1>>(pixel_count, Eigen::Matrix<Decimal, 4, 1>(0, 0, 0, 0))
		};


		for (int i = 0; i < 2; ++i)
		{
			timer.start();
			compute_vertices(vertices[i], frame[i], fovy, aspect_ratio, near_plane, far_plane);
			timer.print_interval("Depth map back projection : ");
		}

	

		ICP<Decimal> icp;
		icp.setInputCloud(vertices[0]);
		icp.setTargetCloud(vertices[1]);


		Eigen::Matrix<Decimal, 4, 4> identity = Eigen::Matrix<Decimal, 4, 4>::Identity();
		Eigen::Matrix<Decimal, 4, 4> rigidTransform = Eigen::Matrix<Decimal, 4, 4>::Identity();
		Eigen::Matrix<Decimal, 3, 3> R;
		Eigen::Matrix<Decimal, 3, 1> t;
	
		for (int i = 0; i < 1; i++)
		{

			timer.start();
			bool icp_success = icp.align_iteration(vertices[0], vertices[1], 
				filter_width, max_distance, width, height, 
				R, t);
			
			timer.print_interval("Computing icp             : ");

			if (icp_success)
			{
				std::cout << "Icp Success" << std::endl;
			}
			else
			{
				std::cout << "Icp Failed" << std::endl;
			}

			std::cout 
				<< std::fixed << std::endl
				<< R << std::endl << std::endl
				<< t.transpose() << std::endl << std::endl;
		}


	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
	}



	return 0;
}



