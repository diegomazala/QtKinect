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
				L"\n< 1 Quad intersection returned false when true was expected>\n", 
				LINE_INFO());
			
			Assert::IsTrue(
				quad_intersection<float>(R1, R2, S3, S2, S1),
				L"\n< 2 Quad intersection returned false when true was expected>\n",
				LINE_INFO());

			R1 = Eigen::Vector3f(1.5f, 1.5f, -1.0f);
			R2 = Eigen::Vector3f(1.5f, 1.5f, 1.0f);

			Assert::IsFalse(
				quad_intersection<float>(R1, R2, S1, S2, S3),
				L"\n< 3 Quad intersection returned true when false was expected>\n",
				LINE_INFO());
		}
	};
	TEST_CLASS(RaycastCpu)
	{
	public:
		
		static bool IsInside(int intersections_count, int last_voxel_index, int curr_voxel_index, int total_voxels)
		{
			return intersections_count > 0 && last_voxel_index != curr_voxel_index && curr_voxel_index < total_voxels;
		}

		
		TEST_METHOD(RaycastWithTriangleIntersections_FrontBack)
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
			Eigen::Vector3f ray_dir_step = ray_direction * 0.01f;

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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();

			bool is_inside = intersections_count > 0;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside)
			{
				intersections_count = box_intersection<Decimal>(
					hit1 + ray_dir_step,
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
				const int hit_voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
				last_voxel = hit_int.cast<Decimal>();

				is_inside = IsInside(intersections_count, voxel_index, hit_voxel_index, total_voxels);

				if (is_inside)
				{
					voxel_index = hit_voxel_index;
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());

		}

		TEST_METHOD(RaycastWithTriangleIntersections_FrontRight)
		{
			typedef float Decimal;
			int vx_count = 3;
			int vx_size = 1;

			std::vector<int> voxels_expected = { 0, 1, 10, 19, 20 };
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Vector3f ray_origin(0.5f, 0.5f, -1.0f);
			Eigen::Vector3f ray_target(1.5f, 0.5f, 1.5f);
			Eigen::Vector3f ray_direction = (ray_target - ray_origin).normalized();
			Eigen::Vector3f ray_dir_step = ray_direction * 0.01f;

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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();

			bool is_inside = intersections_count > 0;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside)
			{
				intersections_count = box_intersection<Decimal>(
					hit1 + ray_dir_step,
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
				int hit_voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
				last_voxel = hit_int.cast<Decimal>();

				is_inside = IsInside(intersections_count, voxel_index, hit_voxel_index, total_voxels);

				if (is_inside)
				{
					voxel_index = hit_voxel_index;
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());

		}

		TEST_METHOD(RaycastWithTriangleIntersections_FrontBottomRight)
		{
			typedef float Decimal;
			int vx_count = 3;
			int vx_size = 1;

			std::vector<int> voxels_expected = { 1, 10, 11, 20 };
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Vector3f ray_origin(0.5f, 0.5f, -1.0f);
			Eigen::Vector3f ray_target(2.5f, 0.0f, 2.5f);
			Eigen::Vector3f ray_direction = (ray_target - ray_origin).normalized();
			Eigen::Vector3f ray_dir_step = ray_direction * 0.01f;

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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();

			bool is_inside = intersections_count > 0;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside)
			{
				intersections_count = box_intersection<Decimal>(
					hit1 + ray_dir_step,
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
				int hit_voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
				last_voxel = hit_int.cast<Decimal>();

				is_inside = IsInside(intersections_count, voxel_index, hit_voxel_index, total_voxels);

				if (is_inside)
				{
					voxel_index = hit_voxel_index;
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());

		}

		TEST_METHOD(RaycastWithTriangleIntersections_FrontTopRight)
		{
			typedef float Decimal;
			int vx_count = 3;
			int vx_size = 1;

			std::vector<int> voxels_expected = { 4, 13, 17, 26 };
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Vector3f ray_origin(0.5f, 0.5f, -1.0f);
			Eigen::Vector3f ray_target(2.5f, 2.5f, 2.5f);
			Eigen::Vector3f ray_direction = (ray_target - ray_origin).normalized();
			Eigen::Vector3f ray_dir_step = ray_direction * 0.01f;

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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();

			bool is_inside = intersections_count > 0;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside)
			{
				intersections_count = box_intersection<Decimal>(
					hit1 + ray_dir_step,
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
				int hit_voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
				last_voxel = hit_int.cast<Decimal>();

				is_inside = IsInside(intersections_count, voxel_index, hit_voxel_index, total_voxels);

				if (is_inside)
				{
					voxel_index = hit_voxel_index;
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());

		}






		TEST_METHOD(RaycastWithQuadIntersections_3x1_FrontBack)
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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();



			bool is_inside = voxel_index > -1 && voxel_index < total_voxels;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside) //voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
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

				is_inside = voxel_index > -1 && voxel_index < total_voxels;

				if (is_inside)
				{
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());
		}

		TEST_METHOD(RaycastWithQuadIntersections_3x1_FrontTop)
		{
			typedef float Decimal;
			int vx_count = 3;
			int vx_size = 1;

			std::vector<int> voxels_expected = { 8 };
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Vector3f origin(1.f, 0.f, -15.0f);
			Eigen::Vector3f target(1.5f, 0.95f, -10.0f);
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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();

			

			bool is_inside = voxel_index > -1 && voxel_index < total_voxels;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside) //voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
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

				is_inside = voxel_index > -1 && voxel_index < total_voxels;
				
				if (is_inside)
				{
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());
		}

		TEST_METHOD(RaycastWithQuadIntersections_2x1_FrontMiddle)
		{
			typedef float Decimal;
			int vx_count = 2;
			int vx_size = 1;

			std::vector<int> voxels_expected = { 3, 7 };
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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();



			bool is_inside = voxel_index > -1 && voxel_index < total_voxels;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside) //voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
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

				is_inside = voxel_index > -1 && voxel_index < total_voxels;

				if (is_inside)
				{
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());
		}

		TEST_METHOD(RaycastWithQuadIntersections_2x1_FrontBottomRight)
		{
			typedef float Decimal;
			int vx_count = 2;
			int vx_size = 1;

			std::vector<int> voxels_expected = { 1, 10, 11 };
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Vector3f origin(-1.f, 1.f, -15.0f);
			Eigen::Vector3f target(-0.1f, 0.7f, -10.0f);
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
			int voxel_index = get_index_from_3d(hit_int, voxel_count, voxel_size);
			Eigen::Matrix<Decimal, 3, 1> last_voxel = hit_int.cast<Decimal>();



			bool is_inside = voxel_index > -1 && voxel_index < total_voxels;

			if (is_inside)
				voxels_intersected.push_back(voxel_index);

			// 
			// Check intersection with each box inside of volume
			// 
			while (is_inside) //voxel_index > -1 && voxel_index < (total_voxels - voxel_count.x() * voxel_count.y()))
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

				is_inside = voxel_index > -1 && voxel_index < total_voxels;

				if (is_inside)
				{
					Eigen::Vector3i last_voxel_index = index_3d_from_array(voxel_index, voxel_count, voxel_size);
					voxels_intersected.push_back(voxel_index);
				}
			}


			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxels>\n", LINE_INFO());
		}
	};


	TEST_CLASS(RaycastFace)
	{
	public:
		TEST_METHOD(RaycastFirstVoxel)
		{
			typedef float Decimal;

			BoxFace face = BoxFace::Undefined;
			int voxel_index = -1;
			int voxel_expected = -1;

			int vx_count = 3;
			int vx_size = 1;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Matrix<Decimal, 3, 1> hit;
			Eigen::Matrix<Decimal, 3, 1> ray_origin(0.5f, 0.5f, -1.0f);
			Eigen::Matrix<Decimal, 3, 1> ray_target;
			Eigen::Matrix<Decimal, 3, 1> ray_direction = (ray_target - ray_origin).normalized();

			voxel_expected = 0;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 0.5f, 2.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			face = raycast_face_volume<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, voxel_index, hit);
			Assert::IsTrue(voxel_index == voxel_expected, L"\n<Raycast did not result with the expected voxel>\n", LINE_INFO());

			voxel_expected = 6;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 3.0f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			face = raycast_face_volume<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, voxel_index, hit);
			Assert::IsTrue(voxel_index == voxel_expected, L"\n<Raycast did not result with the expected voxel>\n", LINE_INFO());

			voxel_expected = 2;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(4.0f, 0.5f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			face = raycast_face_volume<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, voxel_index, hit);
			Assert::IsTrue(voxel_index == voxel_expected, L"\n<Raycast did not result with the expected voxel>\n", LINE_INFO());


		}

		TEST_METHOD(RaycastFaceInOut)
		{
			typedef float Decimal;
			int vx_count = 3;
			int vx_size = 1;

			BoxFace face_in = BoxFace::Undefined;
			BoxFace face_out = BoxFace::Undefined;
			int intersections_expected = 0;
			BoxFace face_in_expected = BoxFace::Undefined;
			BoxFace face_out_expected = BoxFace::Undefined;

			std::vector<int> voxels_expected = { 1, 10, 11 };
			std::vector<int> voxels_intersected;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Matrix<Decimal, 3, 1> ray_origin(0.5f, 0.5f, -1.0f);
			Eigen::Matrix<Decimal, 3, 1> ray_target;
			Eigen::Matrix<Decimal, 3, 1> ray_direction;
			Eigen::Matrix<Decimal, 3, 1> hit_in, hit_out;
			

			intersections_expected = 2;
			face_in_expected = BoxFace::Front;
			face_out_expected = BoxFace::Rear;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 0.5f, 2.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			intersections_expected = raycast_face_in_out<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out);

			Assert::IsTrue(
				face_in_expected == face_in && 
				face_out_expected == face_out, 
				L"\n<Raycast did not result with the expected faces>\n", LINE_INFO());

			intersections_expected = 2;
			face_in_expected = BoxFace::Front;
			face_out_expected = BoxFace::Top;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 3.0f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			intersections_expected = raycast_face_in_out<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out);

			Assert::IsTrue(
				face_in_expected == face_in &&
				face_out_expected == face_out,
				L"\n<Raycast did not result with the expected faces>\n", LINE_INFO());


			intersections_expected = 2;
			face_in_expected = BoxFace::Front;
			face_out_expected = BoxFace::Left;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.0f, 0.5f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			intersections_expected = raycast_face_in_out<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out);

			Assert::IsTrue(
				face_in_expected == face_in &&
				face_out_expected == face_out,
				L"\n<Raycast did not result with the expected faces>\n", LINE_INFO());

			intersections_expected = 2;
			face_in_expected = BoxFace::Front;
			face_out_expected = BoxFace::Right;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(4.0f, 0.5f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			intersections_expected = raycast_face_in_out<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out);

			Assert::IsTrue(
				face_in_expected == face_in &&
				face_out_expected == face_out,
				L"\n<Raycast did not result with the expected faces>\n", LINE_INFO());

			intersections_expected = 2;
			face_in_expected = BoxFace::Front;
			face_out_expected = BoxFace::Bottom;
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 0.0f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			intersections_expected = raycast_face_in_out<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out);

			Assert::IsTrue(
				face_in_expected == face_in &&
				face_out_expected == face_out,
				L"\n<Raycast did not result with the expected faces>\n", LINE_INFO());


			intersections_expected = 2;
			face_in_expected = BoxFace::Bottom;
			face_out_expected = BoxFace::Top;
			ray_origin = Eigen::Matrix<Decimal, 3, 1>(0.5f, -1.f, 0.5f);
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 10.0f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			intersections_expected = raycast_face_in_out<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out);

			Assert::IsTrue(
				face_in_expected == face_in &&
				face_out_expected == face_out,
				L"\n<Raycast did not result with the expected faces>\n", LINE_INFO());


			intersections_expected = 2;
			face_in_expected = BoxFace::Top;
			face_out_expected = BoxFace::Bottom;
			ray_origin = Eigen::Matrix<Decimal, 3, 1>(0.5f, 10.f, 0.5f);
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, -10.0f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			intersections_expected = raycast_face_in_out<Decimal>(ray_origin, ray_direction, voxel_count, voxel_size, face_in, face_out, hit_in, hit_out);

			Assert::IsTrue(
				face_in_expected == face_in &&
				face_out_expected == face_out,
				L"\n<Raycast did not result with the expected faces>\n", LINE_INFO());


		}


		TEST_METHOD(RaycastVolume)
		{
			typedef float Decimal;

			std::vector<int> voxels_expected;
			std::vector<int> voxels_intersected;

			int vx_count = 3;
			int vx_size = 1;

			Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
			Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);

			Eigen::Matrix<Decimal, 3, 1> ray_origin(0.5f, 0.5f, -1.0f);
			Eigen::Matrix<Decimal, 3, 1> ray_target;
			Eigen::Matrix<Decimal, 3, 1> ray_direction;

			voxels_expected = { 0, 9, 18 };
			voxels_intersected.clear();
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 0.5f, 2.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxel list>\n", LINE_INFO());

			voxels_expected = { 0, 9 };
			voxels_intersected.clear();
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 0.0f, 1.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxel list>\n", LINE_INFO());

			voxels_expected = { 3, 6 };
			voxels_intersected.clear();
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 2.5f, 0.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxel list>\n", LINE_INFO());

			voxels_expected = { 4, 5, 14, 17, 26 };
			voxels_intersected.clear();
			ray_target = Eigen::Matrix<Decimal, 3, 1>(2.5f, 2.f, 1.5f);
			ray_direction = (ray_target - ray_origin).normalized();
			raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxel list>\n", LINE_INFO());

			voxels_expected = { 3, 6, 15, 24 };
			voxels_intersected.clear();
			ray_target = Eigen::Matrix<Decimal, 3, 1>(0.5f, 2.0f, 1.f);
			ray_direction = (ray_target - ray_origin).normalized();
			raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxel list>\n", LINE_INFO());


			voxel_count = Eigen::Vector3i(2, 2, 2);
			voxels_expected = { 3, 7 };
			voxels_intersected.clear();
			ray_origin = Eigen::Matrix<Decimal, 3, 1>(1.f, 1.f, -1.f);
			ray_target = Eigen::Matrix<Decimal, 3, 1>(1.f, 1.f, 10.f);
			ray_direction = (ray_target - ray_origin).normalized();
			raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxel list>\n", LINE_INFO());

			voxel_count = Eigen::Vector3i(3, 3, 3);
			voxel_size = Eigen::Vector3i(16, 16, 16);
			voxels_expected = { 8, 5, 4, 13, 10, 9, 18 };
			voxels_intersected.clear();
			ray_origin = Eigen::Matrix<Decimal, 3, 1>(64.f, 64.f, -32.0f);
			ray_target = Eigen::Matrix<Decimal, 3, 1>(8.f, 8.f, 40.f);
			ray_direction = (ray_target - ray_origin).normalized();
			raycast_volume<float>(ray_origin, ray_direction, voxel_count, voxel_size, voxels_intersected);
			Assert::IsTrue(voxels_intersected == voxels_expected, L"\n<Raycast did not result with the expected voxel list>\n", LINE_INFO());
		}
	};
}