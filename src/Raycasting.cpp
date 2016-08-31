
#include "Raycasting.h"


// Usage: ./Raycastingd.exe 3 1 1.5 1.5 -15 1.5 1.5 -10
int main(int argc, char **argv)
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
	std::cout << "Time (ms) : " << t.diff_msec() << std::endl;
	

	std::cout << "Voxels Intersected: ";
	for (auto v : voxels_intersected)
		std::cout << v << ' ';
	std::cout << std::endl;

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
