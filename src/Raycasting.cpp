
#include <QApplication>
#include "QImageWidget.h"
#include "Raycasting.h"


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


int raycast_and_render_two_triangles(int argc, char* argv[])
{
	Eigen::Affine3f camera_to_world = Eigen::Affine3f::Identity();
	Eigen::Vector3f camera_pos(0, 0, 0);
	float scale = tan(DegToRad(60.f * 0.5f));
	float aspect_ratio = 1.7778f;
	unsigned short image_width = 710;
	unsigned short image_height = 400;
	unsigned char* image_data = new unsigned char[image_width * image_height * 3]{0}; // rgb
	QImage image(image_data, image_width, image_height, QImage::Format_RGB888);

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

	Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

	Eigen::Vector3f ray_origin(0.5f, 0.5f, -1.0f);
	Eigen::Vector3f ray_target(0.5f, 2.0f, 1.f);	// { 3, 6, 15, 24 }
	Eigen::Vector3f ray_direction = (ray_target - ray_origin).normalized();

	if (argc > 8)
	{
		vx_count = atoi(argv[1]);
		vx_size = atoi(argv[2]);

		voxel_count = Eigen::Vector3i(vx_count, vx_count, vx_count);
		voxel_size = Eigen::Vector3i(vx_size, vx_size, vx_size);

		ray_origin = Eigen::Vector3f(atof(argv[3]), atof(argv[4]), atof(argv[5]));
		ray_target = Eigen::Vector3f(atof(argv[6]), atof(argv[7]), atof(argv[8]));
		ray_direction = (ray_target - ray_origin).normalized();
	}

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


// Usage: ./Raycastingd.exe 3 1 1.5 1.5 -15 1.5 1.5 -10
int main(int argc, char **argv)
{
	raycast_volume_test(argc, argv);

	Timer t;
	t.start();
	raycast_and_render_two_triangles(argc, argv);
	t.print_interval("Raycasting and render   : ");


	return 0;
}


#if 0
const float window_width = 512.0f;
const float window_height = 424.0f;
QImage image(window_width, window_height, QImage::Format_RGB888);
image.fill(Qt::white);

Eigen::Vector3d window_coord_norm;

std::cout << "Ray casting to image..." << std::endl;
// sweeping the image
for (int y = 0; y < window_height; ++y)
{
	std::cout << "Ray casting to image... " << (double)y / window_height * 100 << "%" << std::endl;
	for (int x = 0; x < window_width; ++x)
	{
		window_coord_norm.x() = ((double)x / window_width * 2.0) - 1.0;
		window_coord_norm.y() = ((double)y / window_height * 2.0) - 1.0;;
		window_coord_norm.z() = origin.z() + near_plane;

		direction = (window_coord_norm - origin).normalized();

		for (const Voxeld v : volume)
		{
			Eigen::Vector3d corner_min = (volume_transformation * (v.point - half_voxel).homogeneous()).head<3>();
			Eigen::Vector3d corner_max = (volume_transformation * (v.point + half_voxel).homogeneous()).head<3>();

			Box box(corner_min, corner_max);
			Ray ray(origin, direction);


			if (box.intersect(ray, ray_near, ray_far))
			{
				//std::cout << "Box Intersected: " << v.point.transpose() << std::endl;
				//image.setPixel(QPoint(x, window_height - y), qRgb(255, 0, 0)); 
				image.setPixel(QPoint(x, y), qRgb(255, 0, 0));
			}
		}
	}
}

std::cout << "Saving to image..." << std::endl;
image.save("raycasting.png");
#endif
