// Including SDKDDKVer.h defines the highest available Windows platform.
// If you wish to build your application for a previous Windows platform, include WinSDKVer.h and
// set the _WIN32_WINNT macro to the platform you wish to support before including SDKDDKVer.h.
#include <SDKDDKVer.h>

#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


#include "ComputeRigidTransform.h"
#include <Eigen/Dense>
#include <time.h>
#include <vector>



namespace TestComputeRigidTransform
{		
	TEST_CLASS(UnitTestRigidTransform)
	{
	public:
		
		TEST_METHOD(TestNoTransform)
		{
			typedef double Type;
			srand(time(NULL));
			
			std::vector<Eigen::Matrix<Type, 3, 1>> vertices;

			for (int i = 0; i < 100; ++i)
			{
				const Eigen::Matrix<Type, 3, 1> v(rand(), rand(), rand());
				vertices.push_back(v);
			}

			Eigen::Matrix<Type, 3, 3> R = Eigen::Matrix<Type, 3, 3>::Zero();
			Eigen::Matrix<Type, 3, 1> t = Eigen::Matrix<Type, 3, 1>::Zero();
			ComputeRigidTransform(vertices, vertices, R, t);

			Assert::IsTrue(R.isIdentity(0.00001), L"\n<Rotation matrix is not identity>\n", LINE_INFO());
			Assert::IsTrue(t.isZero(0.00001), L"\n<Translation vector is not zero>\n", LINE_INFO());
		}

	};


}