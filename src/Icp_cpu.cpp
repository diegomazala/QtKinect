
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


// Usage: ./Icp.exe ../../data/room.knt ../../data/room_normals.obj 0 10 3
int main(int argc, char **argv)
{

	if (argc < 3)
	{
		std::cerr
			<< "Missing parameters. Abort."
			<< std::endl
			<< "Usage: ./Icp.exe <knt_frame> <normals_out_file> <distance_method [point=0, plane=1]> <filter_width> <max_distance_threshold> "
			<< "Usage: ./Icp.exe ../../data/room.knt ../../data/room_normals.obj 0 10 3"
			<< std::endl;
		return EXIT_FAILURE;
	}


	Timer timer;
	const std::string input_filename = argv[1];
	const std::string output_filename = argv[2];

	const int distance_method = (argc > 2) ? atoi(argv[3]) : 0;
	const int filter_width = (argc > 3) ? atoi(argv[4]) : 10;
	const Decimal max_distance = (Decimal)((argc > 4) ? atof(argv[5]) : 100);


	try
	{
		timer.start();
		KinectFrame frame;
		KinectFrame::load(input_filename, frame);
		timer.print_interval("Importing kinect frame    : ");



		const Decimal fovy = KINECT_V2_FOVY;
		const Decimal aspect_ratio = KINECT_V2_DEPTH_ASPECT_RATIO;
		const Decimal near_plane = KINECT_V2_DEPTH_MIN;
		const Decimal far_plane = KINECT_V2_DEPTH_MAX;

#if 1
		std::vector<Eigen::Matrix<Decimal, 4, 1>> vertices(frame.depth.size(), Eigen::Matrix<Decimal, 4, 1>(0, 0, 0, 0));
		std::vector<Eigen::Matrix<Decimal, 3, 1>> normals(frame.depth.size(), Eigen::Matrix<Decimal, 3, 1>(0, 0, -1));
		std::vector<Eigen::Vector3i> colors(frame.depth.size(), Eigen::Vector3i(0, 0, 255));
#else
		std::vector<Eigen::Matrix<Decimal, 4, 1>> vertices;
		std::vector<Eigen::Matrix<Decimal, 3, 1>> normals;
		std::vector<Eigen::Vector3i> colors;
#endif
		timer.start();

		Eigen::Matrix<Decimal, 3, 1> vert_uv, vert_u1v, vert_uv1;


		for (int x = 0; x < frame.depth_width() - 1; ++x)
		{
			for (int y = 0; y < frame.depth_height() - 1; ++y)
			{
				const Decimal depth = frame.depth[y * frame.depth_width() + x];
				const Decimal depth_u1v = frame.depth[y * frame.depth_width() + x + 1];
				const Decimal depth_uv1 = frame.depth[(y + 1) * frame.depth_width() + x];

				if (depth > 0.01 && depth_u1v > 0.01 && depth_uv1 > 0.01)
				{
					vert_uv = window_coord_to_3d(Eigen::Matrix<Decimal, 2, 1>(x, y), depth, fovy, aspect_ratio, near_plane, far_plane, frame.depth_width(), frame.depth_height());
					vert_u1v = window_coord_to_3d(Eigen::Matrix<Decimal, 2, 1>(x + 1, y), depth_u1v, fovy, aspect_ratio, near_plane, far_plane, frame.depth_width(), frame.depth_height());
					vert_uv1 = window_coord_to_3d(Eigen::Matrix<Decimal, 2, 1>(x, y + 1), depth_uv1, fovy, aspect_ratio, near_plane, far_plane, frame.depth_width(), frame.depth_height());

					const Eigen::Matrix<Decimal, 3, 1> n1 = vert_u1v - vert_uv;
					const Eigen::Matrix<Decimal, 3, 1> n2 = vert_uv1 - vert_uv;
					const Eigen::Matrix<Decimal, 3, 1> n = n1.cross(n2).normalized();

					int i = y * frame.depth_width() + x;

#if 1
					vertices[i] = vert_uv.homogeneous();
					normals[i] = n;
					colors[i] = ((n * 0.5f + Eigen::Matrix<Decimal, 3, 1>(0.5, 0.5, 0.5)) * 255.0f).cast<int>();
#else

					if (!vert_uv.isZero(0.0001))
					{
						vertices.push_back(vert_uv.homogeneous());
						normals.push_back(n);
						colors.push_back(((n * 0.5f + Eigen::Matrix<Decimal, 3, 1>(0.5, 0.5, 0.5)) * 255.0f).cast<int>());
					}
#endif
				}

			}
		}
		timer.print_interval("Depth map back projection : ");


		ICP<Decimal> icp;
		icp.setInputCloud(vertices, normals);
		icp.setTargetCloud(vertices);


		Eigen::Matrix<Decimal, 4, 4> identity = Eigen::Matrix<Decimal, 4, 4>::Identity();
		Eigen::Matrix<Decimal, 4, 4> rigidTransform = Eigen::Matrix<Decimal, 4, 4>::Identity();
		Eigen::Matrix<Decimal, 3, 3> R;
		Eigen::Matrix<Decimal, 3, 1> t;
	
		for (int i = 0; i < 1; i++)
		{

			timer.start();
			bool icp_success = icp.align_iteration(vertices, normals, vertices, 
				filter_width, max_distance, frame.depth_width(), frame.depth_height(), 
				R, t, (ICP<Decimal>::DistanceMethod)distance_method);
				//ICP<Decimal>::DistanceMethod::PointToPoint);
			
			//icp.align(1, 1, ICP::DistanceMethod::PointToPoint);
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




		//timer.start();
		////export_obj_with_normals(output_filename, vertices, normals);
		//export_obj_with_colors(output_filename, vertices, colors);
		//timer.print_interval("Exporting output .obj     : ");

	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
	}



	return 0;
}



