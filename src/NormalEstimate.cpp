
#include "Grid.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "Timer.h"
#include "Projection.h"
#include "Volumetric_helper.h"
#include "KinectFrame.h"



// Usage: ./NormalEstimate.exe ../../data/room.knt ../../data/room_normals.obj
int main(int argc, char **argv)
{

	if (argc < 3)
	{
		std::cerr 
			<< "Missing parameters. Abort." 
			<< std::endl
			<< "Usage: ./NormalEstimate.exe ../../data/room.knt ../../data/room_normals.obj"
			<< std::endl;
		return EXIT_FAILURE;
	}


	Timer timer;
	const std::string input_filename = argv[1];
	const std::string output_filename = argv[2];
	std::vector<Eigen::Vector3f> vertices;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector3f> colors;


	

	try
	{
		timer.start();
		KinectFrame frame;
		KinectFrame::load(input_filename, frame);
		timer.print_interval("Importing kinect frame    : ");

		const float fovy = 70.0f;
		const float aspect_ratio = static_cast<float>(frame.depth_width()) / static_cast<float>(frame.depth_height());
		const float near_plane = 0.1f;
		const float far_plane = 10240.0f;

		timer.start();
		
		Eigen::Vector3f vert_uv, vert_u1v, vert_uv1;
		Eigen::Vector3f v, vup, vdown, vleft, vright;

		int i = 0;

		for (int x = 0; x < frame.depth_width() - 1; ++x)
		{
			for (int y = 0; y < frame.depth_height() - 1; ++y)
			{
				const float depth = frame.depth[y * frame.depth_width() + x];
				const float depth_u1v = frame.depth[y * frame.depth_width() + x + 1];
				const float depth_uv1 = frame.depth[(y + 1) * frame.depth_width() + x];

				if (depth < 0.01 || depth_u1v < 0.01 || depth_uv1 < 0.01)
				{
					continue; 

					vert_uv = window_coord_to_3d(
						Eigen::Vector2f(x, y), 
						depth, 
						fovy, 
						aspect_ratio, 
						near_plane, 
						far_plane, 
						frame.depth_width(),
						frame.depth_height());

					const Eigen::Vector3f n(0, 0, 1);

					vertices.push_back(vert_uv);
					normals.push_back(n);
					colors.push_back(n * 255.0f);
				}
				else
				{
					vert_uv = window_coord_to_3d(Eigen::Vector2f(x, y), depth, fovy, aspect_ratio, near_plane, far_plane, frame.depth_width(), frame.depth_height());
					vert_u1v = window_coord_to_3d(Eigen::Vector2f(x + 1, y), depth_u1v, fovy, aspect_ratio, near_plane, far_plane, frame.depth_width(), frame.depth_height());
					vert_uv1 = window_coord_to_3d(Eigen::Vector2f(x, y + 1), depth_uv1, fovy, aspect_ratio, near_plane, far_plane, frame.depth_width(), frame.depth_height());

					const Eigen::Vector3f n1 = vert_u1v - vert_uv;
					const Eigen::Vector3f n2 = vert_uv1 - vert_uv;
					const Eigen::Vector3f n = n1.cross(n2).normalized();

					vertices.push_back(vert_uv);
					normals.push_back(n);
					colors.push_back((n * 0.5f + Eigen::Vector3f(0.5, 0.5, 0.5)) * 255.0f);
				}
			}
		}
		timer.print_interval("Depth map back projection : ");

		timer.start();
		//export_obj_with_normals(output_filename, vertices, normals);
		export_obj_with_colors(output_filename, vertices, colors);
		timer.print_interval("Exporting output .obj     : ");
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
	}



	return 0;
}



