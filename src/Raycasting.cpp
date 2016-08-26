
#include "Raycasting.h"


// Usage: ./Raycastingd.exe 3 1 1.5 1.5 -15 1.5 1.5 -10
int main(int argc, char **argv)
{
	int vol_size = atoi(argv[1]); 
	int vx_size = atoi(argv[2]);
	
	Eigen::Vector3i voxel_count(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);
	
	Eigen::Vector3f origin(atof(argv[3]), atof(argv[4]), atof(argv[5]));
	Eigen::Vector3f target(atof(argv[6]), atof(argv[7]), atof(argv[8]));
	Eigen::Vector3f direction = (target - origin).normalized();
	
	Timer t;
	t.start();
	raycast_with_triangle_intersections(origin, direction, voxel_count, voxel_size);
	double t1 = t.diff_msec();
	t.start();
	raycast_with_quad_intersections(origin, direction, voxel_count, voxel_size);
	double t2 = t.diff_msec();

	std::cout << "Time using triangle intersection : " << t1 << std::endl;
	std::cout << "Time using quad intersection     : " << t2 << std::endl;

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
