// Including SDKDDKVer.h defines the highest available Windows platform.
// If you wish to build your application for a previous Windows platform, include WinSDKVer.h and
// set the _WIN32_WINNT macro to the platform you wish to support before including SDKDDKVer.h.
#include <SDKDDKVer.h>

#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include <algorithm>
#include <sstream>
#include <fstream>
#include "Raycasting.h"
#include "Eigen/Eigen"


namespace TestRaycast
{		
	TEST_CLASS(Intersections)
	{
	public:

		TEST_METHOD(QuadIntersections)
		{
			Eigen::Vector3f R1(0.0f, 0.0f, -1.0f);
			Eigen::Vector3f R2(0.0f, 0.0f, 1.0f);

			Eigen::Vector3f S1(-1.0f, 1.0f, 0.0f);
			Eigen::Vector3f S2(1.0f, 1.0f, 0.0f);
			Eigen::Vector3f S3(-1.0f, -1.0f, 0.0f);

			Assert::IsTrue(
				quad_intersection<float>(R1, R2, S1, S2, S3), 
				L"\n<Quad intersection returned false when true was expected>\n", 
				LINE_INFO());


			R1 = Eigen::Vector3f(1.5f, 1.5f, -1.0f);
			R2 = Eigen::Vector3f(1.5f, 1.5f, 1.0f);

			Assert::IsFalse(
				quad_intersection<float>(R1, R2, S1, S2, S3),
				L"\n<Quad intersection returned true when false was expected>\n",
				LINE_INFO());
		}
	};
	TEST_CLASS(RaycastCpu)
	{
	public:
		

		TEST_METHOD(RaycastWithTriangleIntersections)
		{
			typedef float Decimal;
			int vx_count = 3;
			int vx_size = 1;

			std::vector<int> voxels_expected = { 4, 13, 22 };
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Vector3f ray_origin(1.5f, 1.5f, -15.0f);
			Eigen::Vector3f ray_target(1.5f, 1.5f, -10.0f);
			Eigen::Vector3f ray_direction = (ray_target - ray_origin).normalized();

			Eigen::Vector3i volume_size(
				voxel_count.x() * voxel_size.x(),
				voxel_count.y() * voxel_size.y(),
				voxel_count.z() * voxel_size.z());

			Eigen::Matrix<Decimal, 3, 1> half_volume_size = volume_size.cast<Decimal>() * (Decimal)0.5;
			int total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();
			Decimal half_total_voxels = total_voxels / (Decimal)2.0;

			Eigen::Matrix<Decimal, 3, 1> half_voxel_size = voxel_size.cast<Decimal>() * (Decimal)0.5;
			Eigen::Matrix<Decimal, 3, 1> to_origin = (-volume_size.cast<Decimal>() * (Decimal)0.5);

			Eigen::Matrix<Decimal, 3, 1> hit1;
			Eigen::Matrix<Decimal, 3, 1> hit2;
			Eigen::Matrix<Decimal, 3, 1> hit1_normal;
			Eigen::Matrix<Decimal, 3, 1> hit2_normal;

			//
			// Check intersection with the whole volume
			//
			int intersections_count = box_intersection<Decimal>(
				ray_origin,
				ray_direction,
				half_volume_size,	//volume_center,
				volume_size.x(),
				volume_size.y(),
				volume_size.z(),
				hit1,
				hit2,
				hit1_normal,
				hit2_normal);

			Eigen::Vector3i hit_int = hit1.cast<int>();
			int voxel_index = get_index_test(hit_int, voxel_count);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();

			voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
			{
				intersections_count = box_intersection<Decimal>(
					hit1,
					ray_direction,
					last_voxel + half_voxel_size,
					voxel_size.x(),
					voxel_size.y(),
					voxel_size.z(),
					hit1,
					hit2,
					hit1_normal,
					hit2_normal);

				hit_int = hit1.cast<int>();
				voxel_index = get_index_test(hit_int, voxel_count);
				last_voxel = hit_int.cast<Decimal>();

				voxels_intersected.push_back(voxel_index);
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());

		}


		TEST_METHOD(RaycastWithQuadIntersections)
		{
			typedef float Decimal;
			int vx_count = 3;
			int vx_size = 1;

			std::vector<int> voxels_expected = {4, 13, 22};
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Vector3f origin(1.5f, 1.5f, -15.0f);
			Eigen::Vector3f target(1.5f, 1.5f, -10.0f);
			Eigen::Vector3f direction = (target - origin).normalized();

			Eigen::Vector3i volume_size(
				voxel_count.x() * voxel_size.x(),
				voxel_count.y() * voxel_size.y(),
				voxel_count.z() * voxel_size.z());

			Eigen::Matrix<Decimal, 3, 1> half_volume_size = volume_size.cast<Decimal>() * (Decimal)0.5;
			int total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();
			Decimal half_total_voxels = total_voxels / (Decimal)2.0;

			Eigen::Matrix<Decimal, 3, 1> half_voxel_size = voxel_size.cast<Decimal>() * (Decimal)0.5;
			Eigen::Matrix<Decimal, 3, 1> to_origin = (-volume_size.cast<Decimal>() * (Decimal)0.5);

			Eigen::Matrix<Decimal, 3, 1> hit1;
			Eigen::Matrix<Decimal, 3, 1> hit2;
			Eigen::Matrix<Decimal, 3, 1> hit1_normal;
			Eigen::Matrix<Decimal, 3, 1> hit2_normal;

			//
			// Check intersection with the whole volume
			//
			int intersections_count = box_intersection<Decimal>(
				origin,
				direction,
				half_volume_size,	//volume_center,
				volume_size.x(),
				volume_size.y(),
				volume_size.z(),
				hit1,
				hit2,
				hit1_normal,
				hit2_normal);

			Eigen::Vector3i hit_int = hit1.cast<int>();
			int voxel_index = get_index_test(hit_int, voxel_count);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();

			voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
			{
				int face = -1;
				intersections_count = box_intersection<Decimal>(
					origin,
					direction,
					last_voxel + half_voxel_size,
					voxel_size.x(),
					voxel_size.y(),
					voxel_size.z(),
					hit1_normal,
					hit2_normal,
					face);

				voxel_index = get_index_from_face(face, voxel_index, voxel_count);
				Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count);

				voxels_intersected.push_back(voxel_index);
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());
		}





	};


	TEST_CLASS(RaycastGpu)
	{
	public:

	};
}