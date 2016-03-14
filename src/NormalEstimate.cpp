
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





void depth_map_to_vertex_map(const std::string& input_filename, const std::string& output_filename, std::vector<Eigen::Vector3f>& points3d)
{
	Timer timer;

	const int depth_map_width = 512;
	const int depth_map_height = 424;
	const float fovy = 70.0f;
	const float aspect_ratio = static_cast<float>(depth_map_width) / static_cast<float>(depth_map_height);
	const float near_plane = 0.1f;
	const float far_plane = 10240.0f;

	try
	{
		timer.start();
		KinectFrame frame;
		KinectFrame::load(input_filename, frame);

		timer.print_interval("Importing kinect frame    : ");

		timer.start();
		
		for (int x = 0; x < depth_map_width; ++x)
		{
			for (int y = 0; y < depth_map_height; ++y)
			{
				const Eigen::Vector2f pixel(x, y);
				const float depth = frame.depth[y * depth_map_width + x];

				const Eigen::Vector3f p3d = window_coord_to_3d(pixel, depth, fovy, aspect_ratio, near_plane, far_plane, depth_map_width, depth_map_height);
				points3d.push_back(p3d);
			}
		}
		timer.print_interval("Depth map back projection : ");

		//timer.start();
		//export_obj(output_filename, points3d);
		//timer.print_interval("Exporting output .obj     : ");
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
	}

}

// Usage: ./NormalEstimate.exe ../../data/room.knt ../../data/room_normals.obj
int main(int argc, char **argv)
{
	//Eigen::Vector3f v2(0, 1, 0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
	//Eigen::Vector3f v1(10, 1, 0);

	//std::cout << v1.cross(v2).transpose() << std::endl;
	//std::cout << v1.cross(v2).normalized().transpose() << std::endl;
	//std::cout << (v1.cross(v2).normalized() * 0.5f + Eigen::Vector3f(0.5, 0.5, 0.5)).transpose() << std::endl;

	//std::cout << std::endl << std::endl;

	//v2 = Eigen::Vector3f(0, 1, 0);
	//v1 = Eigen::Vector3f(0, 1, -10);

	//std::cout << v1.cross(v2).transpose() << std::endl;
	//std::cout << v1.cross(v2).normalized().transpose() << std::endl;
	//std::cout << (v1.cross(v2).normalized() * 0.5f + Eigen::Vector3f(0.5, 0.5, 0.5)).transpose() << std::endl;

	//std::cout << std::endl << std::endl;

	//v2 = Eigen::Vector3f(0, 1, 0);
	//v1 = Eigen::Vector3f(0, 1, 10);

	//std::cout << v1.cross(v2).transpose() << std::endl;
	//std::cout << v1.cross(v2).normalized().transpose() << std::endl;
	//std::cout << (v1.cross(v2).normalized() * 0.5f + Eigen::Vector3f(0.5, 0.5, 0.5)).transpose() << std::endl;

	//return 0;


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


	const int depth_map_width = 512;
	const int depth_map_height = 424;
	const float fovy = 70.0f;
	const float aspect_ratio = static_cast<float>(depth_map_width) / static_cast<float>(depth_map_height);
	const float near_plane = 0.1f;
	const float far_plane = 10240.0f;

	try
	{
		timer.start();
		KinectFrame frame;
		KinectFrame::load(input_filename, frame);
	
		timer.print_interval("Importing kinect frame    : ");

		timer.start();
		
		Eigen::Vector3f vert_uv, vert_u1v, vert_uv1;
		Eigen::Vector3f v, vup, vdown, vleft, vright;

		int i = 0;

		for (int x = 1; x < depth_map_width - 1; ++x)
		{
			for (int y = 1; y < depth_map_height - 1; ++y)
			{
				const float depth = frame.depth[y * depth_map_width + x];
				const float depth_u1v = frame.depth[y * depth_map_width + x + 1];
				const float depth_uv1 = frame.depth[(y + 1) * depth_map_width + x];

				if (depth < 0.01 || depth_u1v < 0.01 || depth_uv1 < 0.01)
					continue;

				vert_uv  = window_coord_to_3d(Eigen::Vector2f(x, y), depth, fovy, aspect_ratio, near_plane, far_plane, depth_map_width, depth_map_height);
				vert_u1v = window_coord_to_3d(Eigen::Vector2f(x + 1, y), depth_u1v, fovy, aspect_ratio, near_plane, far_plane, depth_map_width, depth_map_height);
				vert_uv1 = window_coord_to_3d(Eigen::Vector2f(x, y + 1), depth_uv1, fovy, aspect_ratio, near_plane, far_plane, depth_map_width, depth_map_height);


				if (!vert_uv.isZero() && !vert_u1v.isZero() && !vert_uv1.isZero())
				{

					const Eigen::Vector3f n1 = vert_u1v - vert_uv;
					const Eigen::Vector3f n2 = vert_uv1 - vert_uv;
					const Eigen::Vector3f n = n1.cross(n2).normalized();
					
					vertices.push_back(vert_uv);
					normals.push_back(n);
					colors.push_back((n * 0.5f + Eigen::Vector3f(0.5, 0.5, 0.5)) * 255.0f);

					//std::cout << "n " << n.transpose() << std::endl << std::endl;
				}
				//else
				//{
				//	vertices.push_back(vert_uv);
				//	normals.push_back(Eigen::Vector3f::Zero());
				//	colors.push_back(Eigen::Vector3f::Zero());
				//}

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



