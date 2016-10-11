
#include <QApplication>
#include "QRaycastImageWidget.h"
#include "Raycasting.h"
#include "KinectSpecs.h"


#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.

static void multDirMatrix(const Eigen::Vector3f &src, const Eigen::Matrix4f &mat, Eigen::Vector3f &dst)
{
	float a, b, c;

	a = src[0] * mat(0, 0) + src[1] * mat(1, 0) + src[2] * mat(2, 0);
	b = src[0] * mat(0, 1) + src[1] * mat(1, 1) + src[2] * mat(2, 1);
	c = src[0] * mat(0, 2) + src[1] * mat(1, 2) + src[2] * mat(2, 2);

	dst.x() = a;
	dst.y() = b;
	dst.z() = c;
}


template <typename Type>
static Eigen::Matrix<Type, 3, 1> compute_normal(
	const Eigen::Matrix<Type, 3, 1>& p1,
	const Eigen::Matrix<Type, 3, 1>& p2,
	const Eigen::Matrix<Type, 3, 1>& p3)
{
	Eigen::Matrix<Type, 3, 1> u = p2 - p1;
	Eigen::Matrix<Type, 3, 1> v = p3 - p1;

	return v.cross(u).normalized();
}

template <typename Type>
static Eigen::Matrix<Type, 3, 1> reflect(const Eigen::Matrix<Type, 3, 1>& i, const Eigen::Matrix<Type, 3, 1>& n)
{
	return i - 2.0 * n * n.dot(i);
}


int raycast_and_render_grid(int argc, char* argv[])
{
	//int vx_count = 3;
	//int vx_size = 1;
	int vx_count = 8;
	int vx_size = 16;

	Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

	//if (argc > 2)
	//{
	//	vx_count = atoi(argv[1]);
	//	vx_size = atoi(argv[2]);

	//	voxel_count = Eigen::Vector3i(vx_count, vx_count, vx_count);
	//	voxel_size = Eigen::Vector3i(vx_size, vx_size, vx_size);
	//}

	Eigen::Vector3i volume_size(voxel_count.x() * voxel_size.x(), voxel_count.y() * voxel_size.y(), voxel_count.z() * voxel_size.z());
	Eigen::Vector3f half_volume_size = volume_size.cast<float>() * 0.5f;
	const int total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();
	
	std::cout << std::fixed
		<< "Voxel Count  : " << voxel_count.transpose() << std::endl
		<< "Voxel Size   : " << voxel_size.transpose() << std::endl
		<< "Volume Size  : " << volume_size.transpose() << std::endl
		<< "Total Voxels : " << total_voxels << std::endl;

	//
	// create the grid params and set a few voxels with different signals 
	// in order to obtain zero crossings
	//
	std::vector<Eigen::Vector2f> tsdf(total_voxels, Eigen::Vector2f::Ones());
	//tsdf.at(13)[0] = 
	//tsdf.at(22)[0] = 
	//tsdf.at(18)[0] = 
	//tsdf.at(26)[0] = -1.0f;
	tsdf.at(0)[0] = -1.0f;
	tsdf.at(tsdf.size() - voxel_count.x())[0] =
	tsdf.at(tsdf.size() - 1)[0] = -1.0f;

	//
	// setup camera parameters
	//
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
#if 0
	float cam_z = (-vx_count - 1) * vx_size;
	camera_to_world.translate(Eigen::Vector3f(half_volume_size.x(), half_volume_size.y(), cam_z));
#else

	//camera_to_world.rotate(Eigen::AngleAxisf(DegToRad(-25), Eigen::Vector3f::UnitX()));
	//camera_to_world.translate(Eigen::Vector3f(half_volume_size.x(), half_volume_size.y(), 0));
	//camera_to_world.translate(Eigen::Vector3f(0, 7, -4));

	//camera_to_world.rotate(Eigen::AngleAxisf(DegToRad(-20), Eigen::Vector3f::UnitY()));
	//camera_to_world.translate(Eigen::Vector3f(half_volume_size.x(), half_volume_size.y(), 0));
	//camera_to_world.translate(Eigen::Vector3f(-5, 0, -6));

	camera_to_world.translate(Eigen::Vector3f(60, 256, -60));
	camera_to_world.rotate(Eigen::AngleAxisf(DegToRad(-60), Eigen::Vector3f::UnitX()));

#endif

	Eigen::Vector3f camera_pos = camera_to_world.matrix().col(3).head<3>();
	float scale = (float)tan(DegToRad(KINECT_V2_FOVY * 0.5f));
	float aspect_ratio = KINECT_V2_DEPTH_ASPECT_RATIO;

	// 
	// setup image parameters
	//
	unsigned short image_width = KINECT_V2_DEPTH_WIDTH;
	unsigned short image_height = image_width / aspect_ratio;
	QImage image(image_width, image_height, QImage::Format_RGB888);
	image.fill(Qt::GlobalColor::black);

	//
	// for each pixel, trace a ray
	//
	for (int y = 0; y < image_height; ++y)
	{
		for (int x = 0; x < image_width; ++x)
		{
			// Convert from image space (in pixels) to screen space
			// Screen Space alon X axis = [-aspect ratio, aspect ratio] 
			// Screen Space alon Y axis = [-1, 1]
			Eigen::Vector3f screen_coord(
				(2 * (x + 0.5f) / (float)image_width - 1) * aspect_ratio * scale,
				(1 - 2 * (y + 0.5f) / (float)image_height) * scale,
				1.0f);

			Eigen::Vector3f direction;
			multDirMatrix(screen_coord, camera_to_world.matrix(), direction);
			direction.normalize();

			long voxels_zero_crossing[2];
			if (raycast_tsdf_volume(camera_pos, direction, voxel_count, voxel_size, tsdf, voxels_zero_crossing) > 0)
			{
				if (voxels_zero_crossing[0] > -1 && voxels_zero_crossing[1] > -1)
				{
					image.setPixel(QPoint(x, y), qRgb(128, 128, 0));
				}
				else
				{
					image.setPixel(QPoint(x, y), qRgb(128, 0, 0));
				}
			}
		}
	}

	QApplication app(argc, argv);
	QRaycastImageWidget widget;
	widget.setImage(image);
	widget.show();

	return app.exec();
}


int raycast_and_render_two_triangles(int argc, char* argv[])
{
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	camera_to_world.translate(Eigen::Vector3f(2, 2, 5));
	Eigen::Vector3f camera_pos = camera_to_world.matrix().col(3).head<3>();
	float scale = tan(DegToRad(60.f * 0.5f));
	float aspect_ratio = 1.7778f;
	unsigned short image_width = 710;
	unsigned short image_height = 400;
	QImage image(image_width, image_height, QImage::Format_RGB888);

	Eigen::Vector3f hit;
	Eigen::Vector3f v1(0.0f, -1.0f, -2.0f);
	Eigen::Vector3f v2(0.0f, 1.0f, -4.0f);
	Eigen::Vector3f v3(-1.0f, -1.0f, -3.0f);
	Eigen::Vector3f v4(0.0f, -1.0f, -2.0f);
	Eigen::Vector3f v5(0.0f, 1.0f, -4.0f);
	Eigen::Vector3f v6(1.0f, -1.0f, -3.0f);

	Eigen::Vector3f diff_color(1, 0, 0);
	Eigen::Vector3f spec_color(1, 1, 0);
	float spec_shininess = 1.0f;
	Eigen::Vector3f E(0, 0, -1);				// view direction
	Eigen::Vector3f L = Eigen::Vector3f(0.2, -1, -1).normalized();	// light direction
	Eigen::Vector3f N[2] = {
		compute_normal(v1, v2, v3),
		compute_normal(v4, v5, v6) };
	Eigen::Vector3f R[2] = {
		-reflect(L, N[0]).normalized(),
		-reflect(L, N[1]).normalized() };

	
	for (int y = 0; y < image_height; ++y)
	{
		for (int x = 0; x < image_width; ++x)
		{
			// Convert from image space (in pixels) to screen space
			// Screen Space alon X axis = [-aspect ratio, aspect ratio] 
			// Screen Space alon Y axis = [-1, 1]
			Eigen::Vector3f screen_coord(
				(2 * (x + 0.5f) / (float)image_width - 1) * aspect_ratio * scale,
				(1 - 2 * (y + 0.5f) / (float)image_height) * scale,
				-1.0f);

			Eigen::Vector3f direction;
			multDirMatrix(screen_coord, camera_to_world.matrix(), direction);
			direction.normalize();

			bool intersec[2] = {
				triangle_intersection(camera_pos, direction, v1, v2, v3, hit),
				triangle_intersection(camera_pos, direction, v4, v5, v6, hit) };

			for (int i = 0; i < 2; ++i)
			{
				if (intersec[i])
				{
					Eigen::Vector3f diff = diff_color * std::fmax(N[i].dot(L), 0.0f);
					Eigen::Vector3f spec = spec_color * pow(std::fmax(R[i].dot(E), 0.0f), spec_shininess);
					Eigen::Vector3f color = eigen_clamp(diff + spec, 0.f, 1.f) * 255;
					image.setPixel(QPoint(x, y), qRgb(color.x(), color.y(), color.z()));
				}
			}
		}
	}
	


	QApplication app(argc, argv);
	QImageWidget widget;
	widget.setImage(image);
	widget.show();

	return app.exec();
}

int raycast_volume_test(int argc, char **argv)
{
	int vx_count = 3;
	int vx_size = 1;
	//int vx_count = 2;
	//int vx_size = 2;
	//int vx_count = 3;
	//int vx_size = 16;

	Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

	//Eigen::Vector3f ray_origin(0.5f, 0.5f, -1.0f);
	//Eigen::Vector3f ray_target(0.5f, 2.0f, 1.f);	// { 3, 6, 15, 24 }
	//Eigen::Vector3f ray_origin(2.f, 2.f, -2.0f);
	//Eigen::Vector3f ray_target(2.f, 2.f, 2.f);	// { 3, 7 }
	//Eigen::Vector3f ray_origin(64.f, 64.f, -32.0f);
	//Eigen::Vector3f ray_target(8.f, 8.f, 40.f);	
	//Eigen::Vector3f ray_origin(50.f, 2.9f, -1.0f);
	//Eigen::Vector3f ray_target(1.5f, 2.9f, 1.f);	// { 8, 7, 16, 15 }
	Eigen::Vector3f ray_origin(50.f, 50.0f, -1.0f);
	Eigen::Vector3f ray_target(1.5f, 2.9f, 1.f);	// { 7, 16, 15, 12 }
	
	Eigen::Vector3f ray_direction = (ray_target - ray_origin).normalized();

	//if (argc > 8)
	//{
	//	vx_count = atoi(argv[1]);
	//	vx_size = atoi(argv[2]);

	//	voxel_count = Eigen::Vector3i(vx_count, vx_count, vx_count);
	//	voxel_size = Eigen::Vector3i(vx_size, vx_size, vx_size);

	//	ray_origin = Eigen::Vector3f(atof(argv[3]), atof(argv[4]), atof(argv[5]));
	//	ray_target = Eigen::Vector3f(atof(argv[6]), atof(argv[7]), atof(argv[8]));
	//	ray_direction = (ray_target - ray_origin).normalized();
	//}

	std::vector<int> voxels_intersected;
	Timer t;
	t.start();
	raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
	t.print_interval("Raycasting volume       : ");


	std::cout << "Voxels Intersected: ";
	for (auto v : voxels_intersected)
		std::cout << v << ' ';
	std::cout << std::endl;

	return 0;
}




int main(int argc, char **argv)
{
#if 0
	// Usage: ./Raycastingd.exe vx_count vx_size cam_x y z target_x y z
	// Usage: ./Raycastingd.exe 3 1 1.5 1.5 -15 1.5 1.5 -10
	raycast_volume_test(argc, argv);
#endif

#if 0
	// Usage: ./Raycastingd.exe 
	Timer t;
	t.start();
	raycast_and_render_two_triangles(argc, argv);
	t.print_interval("Raycasting and render   : ");
#endif

#if 1
	// Usage: ./Raycastingd.exe vx_count vx_size
	// Usage: ./Raycastingd.exe 3 1 
	raycast_and_render_grid(argc, argv);
#endif

	return 0;
}
