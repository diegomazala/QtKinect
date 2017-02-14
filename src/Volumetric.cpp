
#include <QApplication>
#include <QKeyEvent>
#include <QPushButton>
#include "GLModelViewer.h"
#include "GLModel.h"
#include "GLShaderProgram.h"
#include "Grid.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include "Timer.h"
#include "Projection.h"
#include "Volumetric_helper.h"





#if 0
void raycast_volume()
{
	timer.start();

	Eigen::Vector3d origin = T.first.col(3).head<3>();
	Eigen::Vector3d window_coord_norm;

	std::vector<Eigen::Vector3d> output_cloud;

	// Sweep the volume looking for the zero crossing
	for (int y = 0; y < window_height * 0.1; ++y)
	{
		std::cout << "Ray casting to image... " << (double)y / window_height * 100 << "%" << std::endl;

		for (int x = 0; x < window_width * 0.1; ++x)
		{
			window_coord_norm.x() = ((double)x / window_width * 2.0) - 1.0;
			window_coord_norm.y() = ((double)y / window_height * 2.0) - 1.0;
			window_coord_norm.z() = origin.z() + near_plane;
			Eigen::Vector3d direction = (window_coord_norm - origin).normalized();

			std::vector<int> intersections = Grid::find_intersections(grid.data, volume_size, voxel_size, grid.transformation, origin, direction, near_plane, far_plane);
			Grid::sort_intersections(intersections, grid.data, origin);

			for (int i = 1; i < intersections.size(); ++i)
			{
				const Voxeld& prev = grid.data.at(i - 1);
				const Voxeld& curr = grid.data.at(i);

				const bool& same_sign = ((prev.tsdf < 0) == (curr.tsdf < 0));

				if (!same_sign)		// it is a zero-crossing
				{
					output_cloud.push_back(curr.point);
				}
			}
		}
	}

	timer.print_interval("Raycasting volume   : ");


	export_obj("../../data/output_cloud.obj", output_cloud);
}
#endif

// Usage: ./Volumetricd.exe ../../data/monkey.obj 256 4 2 90
int main(int argc, char **argv)
{
	if (argc < 6)
	{
		std::cerr << "Missing parameters. Abort." 
			<< std::endl
			<< "Usage:  ./Volumetricd.exe ../../data/monkey.obj 256 8 2 90"
			<< std::endl;
		return EXIT_FAILURE;
	}
	Timer timer;
	const std::string filepath = argv[1];
	const int vol_size = atoi(argv[2]);
	const int vx_size = atoi(argv[3]);
	const int cloud_count = atoi(argv[4]);
	const int rot_interval = atoi(argv[5]);

	std::pair<std::vector<double>, std::vector<double>> depth_buffer;

	//
	// Projection and Modelview Matrices
	//
	Eigen::Matrix4d K = perspective_matrix(fov_y, aspect_ratio, near_plane, far_plane);
	std::pair<Eigen::Matrix4d, Eigen::Matrix4d>	T(Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());


	//
	// Creating volume
	//
	Eigen::Vector3d voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3d volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3d voxel_count(volume_size.x() / voxel_size.x(), volume_size.y() / voxel_size.y(), volume_size.z() / voxel_size.z());
	//
	Eigen::Affine3d grid_affine = Eigen::Affine3d::Identity();
	grid_affine.translate(Eigen::Vector3d(0, 0, -256));
	grid_affine.scale(Eigen::Vector3d(1, 1, -1));	// z is negative inside of screen


	Grid<double> grid(volume_size, voxel_size, grid_affine.matrix());


	//
	// Importing .obj
	//
	timer.start();
	std::vector<Eigen::Vector3d> points3DOrig, pointsTmp;
	import_obj(filepath, points3DOrig);
	timer.print_interval("Importing monkey    : ");
	std::cout << "Monkey point count  : " << points3DOrig.size() << std::endl;

	// 
	// Translating and rotating monkey point cloud 
	std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> cloud;
	//
	Eigen::Affine3d rotate = Eigen::Affine3d::Identity();
	Eigen::Affine3d translate = Eigen::Affine3d::Identity();
	translate.translate(Eigen::Vector3d(0, 0, -256));


	// 
	// Compute first cloud
	//
	for (Eigen::Vector3d p3d : points3DOrig)
	{
		Eigen::Vector4d rot = translate.matrix() * rotate.matrix() * p3d.homogeneous();
		rot /= rot.w();
		cloud.first.push_back(rot.head<3>());
	}
	//
	// Update grid with first cloud
	//
	timer.start();
	create_depth_buffer<double>(depth_buffer.first, cloud.first, K, Eigen::Matrix4d::Identity(), far_plane);
	timer.print_interval("CPU compute depth   : ");

	timer.start();
	update_volume(grid, depth_buffer.first, K, T.first);
	timer.print_interval("CPU Update volume   : ");

	//
	// Compute next clouds
	Eigen::Matrix4d cloud_mat = Eigen::Matrix4d::Identity();
	Timer iter_timer;
	for (int i = 1; i < cloud_count; ++i)
	{
		std::cout << std::endl << i << " : " << i * rot_interval << std::endl;
		iter_timer.start();

		// Rotation matrix
		rotate = Eigen::Affine3d::Identity();
		rotate.rotate(Eigen::AngleAxisd(DegToRad(i * rot_interval), Eigen::Vector3d::UnitY()));

		cloud.second.clear();
		for (Eigen::Vector3d p3d : points3DOrig)
		{
			Eigen::Vector4d rot = translate.matrix() * rotate.matrix() * p3d.homogeneous();
			rot /= rot.w();
			cloud.second.push_back(rot.head<3>());
		}

		//export_obj("../../data/cloud_cpu_2.obj", cloud.second);

		timer.start();
		create_depth_buffer<double>(depth_buffer.second, cloud.second, K, Eigen::Matrix4d::Identity(), far_plane);
		timer.print_interval("Compute depth buffer: ");

		//export_depth_buffer("../../data/cpu_depth_buffer_2.obj", depth_buffer.second);

		timer.start();
		Eigen::Matrix4d icp_mat;
		ComputeRigidTransform(cloud.first, cloud.second, icp_mat);
		timer.print_interval("Compute rigid transf: ");

		//std::cout << std::fixed << std::endl << "icp_mat " << std::endl << icp_mat << std::endl;

		// accumulate matrix
		cloud_mat = cloud_mat * icp_mat;

		//std::cout << std::fixed << std::endl << "cloud_mat " << std::endl << cloud_mat << std::endl;

		timer.start();
		//update_volume(grid, depth_buffer.second, K, cloud_mat.inverse());
		update_volume(grid, depth_buffer.second, K, cloud_mat.inverse());
		timer.print_interval("Update volume       : ");


		// copy second point cloud to first
		cloud.first = cloud.second;
		//depth_buffer.first = depth_buffer.second;

		iter_timer.print_interval("Iteration time      : ");
	}


	//std::cout << "------- // --------" << std::endl;
	//for (int i = 0; i <  grid.data.size(); ++i)
	//{
	//	const Eigen::Vector3d& point = grid.data[i].point;

	//	std::cout << point.transpose() << "\t\t" << grid.data[i].tsdf << " " << grid.data[i].weight << std::endl;
	//}
	//std::cout << "------- // --------" << std::endl;

//	timer.start();
//	export_volume("../../data/grid_volume_cpu.obj", grid.data);
//	timer.print_interval("Exporting volume    : ");
//	return 0;


	QApplication app(argc, argv);

	//
	// setup opengl viewer
	// 
	GLModelViewer glwidget;
	glwidget.resize(640, 480);
	glwidget.setPerspective(60.0f, 0.1f, 10240.0f);
	glwidget.move(320, 0);
	glwidget.setWindowTitle("Point Cloud");
	glwidget.setWeelSpeed(0.1f);
	glwidget.setPosition(0, 0, -0.5f);
	glwidget.show();

	
	Eigen::Matrix4d to_origin = Eigen::Matrix4d::Identity();
	to_origin.col(3) << -(volume_size.x() / 2.0), -(volume_size.y() / 2.0), -(volume_size.z() / 2.0), 1.0;	// set translate


	std::vector<Eigen::Vector4f> vertices, colors;

	int i = 0;
	for (int z = 0; z <= volume_size.z(); z += voxel_size.z())
	{
		for (int y = 0; y <= volume_size.y(); y += voxel_size.y())
		{
			for (int x = 0; x <= volume_size.x(); x += voxel_size.x(), i++)
			{
				const float tsdf = grid.data.at(i).tsdf;

				//Eigen::Vector4d p = grid_affine.matrix() * to_origin * Eigen::Vector4d(x, y, z, 1);
				Eigen::Vector4d p = to_origin * Eigen::Vector4d(x, y, z, 1);
				p /= p.w();

				if (tsdf > 0.1)
				{
					vertices.push_back(p.cast<float>());
					colors.push_back(Eigen::Vector4f(0, 1, 0, 1));
				}
				else if (tsdf < -0.1)
				{
					vertices.push_back(p.cast<float>());
					colors.push_back(Eigen::Vector4f(1, 0, 0, 1));
				}
			}
		}
	}




	//
	// setup model
	// 
	std::shared_ptr<GLModel> model(new GLModel);
	model->initGL();
	model->setVertices(&vertices[0][0], vertices.size(), 4);
	model->setColors(&colors[0][0], colors.size(), 4);
	glwidget.addModel(model);


	//
	// setup kinect shader program
	// 
	std::shared_ptr<GLShaderProgram> kinectShaderProgram(new GLShaderProgram);
	if (kinectShaderProgram->build("color.vert", "color.frag"))
		model->setShaderProgram(kinectShaderProgram);

	return app.exec();
}



