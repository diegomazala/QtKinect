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