
#include <QApplication>
#include <QKeyEvent>
#include "GLPointCloudViewer.h"
#include <iostream>
#include "Timer.h"
#include "ObjFile.h"
#include "ComputeRigidTransform.h"
#include "ICP.h"

bool icp_iteration(const std::vector<Eigen::Vector4f>& points_src, const std::vector<Eigen::Vector4f>& points_dst, Eigen::Matrix3f& R, Eigen::Vector3f t)
{	
	std::vector<Eigen::Vector4f> points_match;
	
	//
	// Computing ICP
	//
	for (const Eigen::Vector4f& p1 : points_src)
	{
		Eigen::Vector4f closer = points_dst[0];
		float min_distance = (p1 - points_dst[0]).squaredNorm();

		for (const Eigen::Vector4f& p2 : points_dst)
		{
			float dist = (p1 - p2).squaredNorm();
			if (dist < min_distance)
			{
				closer = p2;
				min_distance = dist;
			}
		}
		points_match.push_back(closer);
	}

	return ComputeRigidTransform(points_src, points_match, R, t);
	
}



static float distance_point_to_point(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2)
{
	//return (v1 - v2).norm();
	return (v1 - v2).squaredNorm();
}

static float distance_point_to_plane_v1(const Eigen::Vector3f& v0, const Eigen::Vector3f& normal, const Eigen::Vector3f& v)
{
	Eigen::Vector3f n = normal.normalized();
	const Eigen::Vector4f plane(n.x(), n.y(), n.z(), -v0.dot(n));	// a*x + b*y + c*z + d = 0
	return std::fabs((plane.x() * v.x() + plane.y() * v.y() + plane.z() * v.z() + plane.w()));
}

static float distance_point_to_plane_v2(const Eigen::Vector3f& v0, const Eigen::Vector3f& n, const Eigen::Vector3f& v)
{
	return std::fabs(
		n.x() * (v.x() - v0.x()) +
		n.y() * (v.y() - v0.y()) +
		n.z() * (v.z() - v0.z()));
}

static float distance_point_to_plane_v3(const Eigen::Vector3f& xp, const Eigen::Vector3f& n, const Eigen::Vector3f& x0)
{
	return n.dot(x0 - xp) / n.norm();
}

static float distance_point_to_plane_v4(const Eigen::Vector3f& xp, const Eigen::Vector3f& n, const Eigen::Vector3f& x0)
{
	return n.dot(x0 - xp) / n.squaredNorm();
}


void test_distances(float angle)
{
	std::vector<Eigen::Vector4f> vertices, points_dst;
	std::vector<Eigen::Vector3f> normals;
	import_obj("../../data/cube_with_normals.obj", vertices, normals);

	Eigen::Affine3f affine = Eigen::Affine3f::Identity();
	affine.rotate(Eigen::AngleAxisf(angle * M_PI / 180.f, Eigen::Vector3f::UnitY()));
	affine.translate(Eigen::Vector3f(1, 0, 0));

	for (int i = 0; i < vertices.size(); ++i)
	{
		Eigen::Vector4f v = vertices[i];
		v /= v.w();

		Eigen::Vector3f n = normals[i].normalized();

		Eigen::Vector4f pt = affine.matrix() * v;
		pt /= pt.w();

		Eigen::Vector3f x0 = v.head<3>();
		Eigen::Vector3f x = pt.head<3>();
		

		std::cout << std::fixed
			<< "x0   : " << x0.transpose() << std::endl
			<< "x    : " << x.transpose() << std::endl
			<< "n    : " << n.transpose() << std::endl
			<< "dpt  : " << distance_point_to_point(x, x0) << std::endl
			<< "dpp1 : " << distance_point_to_plane_v1(x, n, x0) << std::endl
			<< "dpp2 : " << distance_point_to_plane_v2(x, n, x0) << std::endl
			<< "dpp3 : " << distance_point_to_plane_v3(x0, n, x) << std::endl
			<< "dpp4 : " << distance_point_to_plane_v4(x0, n, x) << std::endl
			<< std::endl;
	}

}


int main(int argc, char **argv)
{

	if (argc < 3)
	{
		std::cerr << "Usage: PointCloudViewer.exe obj_file_point_cloud <angle_rotaton> <number_of_iterations> <error_precision> <distance_method = 'pt' or 'pp'>" << std::endl;
		std::cerr << "Usage: PointCloudViewer.exe monkey_low.obj 5 3 0.025 pt" << std::endl;
		return EXIT_FAILURE;
	}

	Timer timer;
	std::string filename = argv[1];
	float angle = atof(argv[2]);
	int iterations = atoi(argv[3]);
	float error_precision = atof(argv[4]);
	std::string distance_method = "pp";
	if (argc > 4)
		distance_method = argv[5];


	//test_distances(angle);
	//return 0;


	//
	// Importing .obj
	//
	timer.start();
	std::vector<Eigen::Vector4f> vertices, points_dst, points_match;
	std::vector<Eigen::Vector3f> normals, normals_dst;
	import_obj(filename, vertices, normals);
	timer.print_interval("Importing monkey    : ");
	std::cout 
		<< "Vertices count  : " << vertices.size() << std::endl
		<< "Normals count   : " << normals.size() << std::endl;


	//
	// Rotating to generate second point cloud
	// 
	Eigen::Affine3f affine = Eigen::Affine3f::Identity();
	affine.rotate(Eigen::AngleAxisf(angle * M_PI / 180.f, Eigen::Vector3f::UnitY()));
	//affine.translate(Eigen::Vector3f(0, 0, 0));
	for (const Eigen::Vector4f& p : vertices)
		points_dst.push_back(affine.matrix() * p);

	Eigen::Matrix4f targetTransform = affine.matrix();
	Eigen::Matrix4f rigidTransform;


	ICP icp;
	icp.setInputCloud(vertices, normals);
	icp.setTargetCloud(points_dst);

	icp.align(iterations, error_precision, distance_method.compare("pp") ? ICP::DistanceMethod::PointToPlane : ICP::DistanceMethod::PointToPoint);

	rigidTransform = icp.getTransform();

	const std::vector<Eigen::Vector4f>& points_icp = icp.getResultCloud();

	
	std::cout << std::fixed
		<< "Rigid Transform Accum " << std::endl
		<< rigidTransform << std::endl
		<< std::endl;


	std::cout << std::fixed << std::endl
		<< "Target Transform        : " << std::endl << targetTransform << std::endl
		<< std::endl;


	
	//
	// Viewer
	//
	QApplication app(argc, argv);


#if 1
	GLPointCloudViewer glwidget;
	glwidget.resize(1024, 848);
	glwidget.move(0, 0);
	glwidget.setWindowTitle("ICP");
	glwidget.show();
	glwidget.addPointCloud(vertices, Eigen::Vector3f(1, 0, 0));
	glwidget.addPointCloud(points_dst, Eigen::Vector3f(0, 1, 0));
	glwidget.addPointCloud(points_icp, Eigen::Vector3f(1, 1, 1));
	glwidget.setWeelSpeed(0.01);
	glwidget.setDistance(-5);
#else // multiple windows
	GLPointCloudViewer glwidget[3];
	glwidget[0].resize(512, 424);
	glwidget[0].move(0, 0);
	glwidget[0].setWindowTitle("Point Cloud Source");
	glwidget[0].show();
	glwidget[0].addPointCloud(vertices, Eigen::Vector3f(1, 0, 0));
	glwidget[0].setWeelSpeed(0.01);
	glwidget[0].setDistance(-5);
	
	glwidget[1].resize(512, 424);
	glwidget[1].move(512, 0);
	glwidget[1].setWindowTitle("Point Cloud Destination");
	glwidget[1].show();
	glwidget[1].addPointCloud(points_dst, Eigen::Vector3f(1, 1, 0));
	glwidget[1].setWeelSpeed(0.01);
	glwidget[1].setDistance(-5);

	glwidget[2].resize(512, 424);
	glwidget[2].move(0, 425);
	glwidget[2].setWindowTitle("Point Cloud Result");
	glwidget[2].show();
	glwidget[2].addPointCloud(points_icp, Eigen::Vector3f(1, 1, 1));
	glwidget[2].setWeelSpeed(0.01);
	glwidget[2].setDistance(-5);
#endif


	return app.exec();
}
