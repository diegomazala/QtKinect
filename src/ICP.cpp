
#include "ICP.h"
#include <iostream>

static float distance_point_to_point(const Eigen::Vector4f& v1, const Eigen::Vector4f& v2)
{
	//return (v1 - v2).norm();
	return (v1 - v2).squaredNorm();
}

#if 0
static float distance_point_to_plane(const Eigen::Vector4f& v, const Eigen::Vector3f& normal, const Eigen::Vector4f& v2)
{
	Eigen::Vector3f n = normal.normalized();
	const Eigen::Vector3f pt = (v / v.w()).head<3>();
	const Eigen::Vector4f plane(n.x(), n.y(), n.z(), -pt.dot(n));	// a*x + b*y + c*z + d = 0
	return (plane.x() * v2.x() + plane.y() * v2.y() + plane.z() * v2.z() + plane.w());
}
#else

static float distance_point_to_plane(const Eigen::Vector3f& xp, const Eigen::Vector3f& n, const Eigen::Vector3f& x0)
{
	return n.dot(x0 - xp) / n.norm();
}
static float distance_point_to_plane(const Eigen::Vector4f& xp, const Eigen::Vector3f& n, const Eigen::Vector4f& x0)
{
	Eigen::Vector3f xx0 = (x0 / x0.w()).head<3>();
	Eigen::Vector3f xxp = (xp / xp.w()).head<3>();
	return n.dot(xx0 - xxp) / n.norm();
}
#endif



ICP::ICP():
	input_vertices_ptr(nullptr),
	input_normals_ptr(nullptr),
	target_vertices_ptr(nullptr)
{
}


ICP::~ICP()
{
	input_vertices_ptr = nullptr;
	input_normals_ptr = nullptr;
	target_vertices_ptr = nullptr;
}


Eigen::Matrix3f ICP::getRotation() const
{
	return rotation;
}


Eigen::Vector3f ICP::getTranslation() const
{
	return translation;
}


Eigen::Matrix4f ICP::getTransform() const
{
	Eigen::Matrix4f transform;
	transform.block(0, 0, 3, 3) = rotation;
	transform.row(3).setZero();
	transform.col(3) = translation.homogeneous();
	return transform;
}


Eigen::Matrix4f ICP::getTransform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
	Eigen::Matrix4f transform;
	transform.block(0, 0, 3, 3) = R;
	transform.row(3).setZero();
	transform.col(3) = t.homogeneous();
	return transform;
}



void ICP::setInputCloud(const std::vector<Eigen::Vector4f>& vertices)
{
	input_vertices_ptr = &vertices;
	vertices_icp = vertices;
}


void ICP::setInputCloud(const std::vector<Eigen::Vector4f>& vertices, const std::vector<Eigen::Vector3f>& normals)
{
	input_vertices_ptr = &vertices;
	input_normals_ptr = &normals;
	vertices_icp = vertices;
	normals_icp = normals;
}


void ICP::setTargetCloud(const std::vector<Eigen::Vector4f>& vertices)
{
	target_vertices_ptr = &vertices;
}


const std::vector<Eigen::Vector4f>& ICP::getResultCloud() const
{
	return vertices_icp;
}


bool ICP::align(const int iterations, const float error_precision, const DistanceMethod distance_method)
{
	Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f rigidTransform = Eigen::Matrix4f::Identity();
	Eigen::Matrix3f R;
	Eigen::Vector3f t;

	rotation = Eigen::Matrix3f::Identity();
	translation = Eigen::Vector3f::Zero();

	for (int i = 0; i < iterations; ++i)
	{
		//if (align_iteration(vertices_icp, *input_normals_ptr, *target_vertices_ptr, R, t, distance_method))
		if (align_iteration(vertices_icp, normals_icp, *target_vertices_ptr, R, t, distance_method))
		{
			rotation *= R;
			translation += t;

			//if (R.isIdentity(error_precision) && t.isZero(error_precision))
			if (getTransform(R, t).isIdentity(error_precision))
			{
				std::cout << "[OK] They are approx. Iteration = " << i << std::endl;
				return true;
			}

			rigidTransform = getTransform(rotation, translation);

			//for (Eigen::Vector4f& p : vertices_icp)
			for (int p = 0; p < vertices_icp.size(); ++p)
			{
				Eigen::Vector4f& v = vertices_icp[p];
				v = rigidTransform * v;
				v /= v.w();

				Eigen::Vector4f nn = normals_icp[i].homogeneous();
				nn = rigidTransform * nn;
				nn /= nn.w();
			}

			std::cout << i << std::endl << std::fixed << R << std::endl << t.transpose() << std::endl;
			//std::cout << i << std::endl<< std::fixed << rigidTransform << std::endl << std::endl;
						
		}
		else
		{
			std::cout << "[FAIL] Could not compute ICP" << std::endl;
			return false;
		}

		//std::cout << std::fixed
		//	<< "Iteration Transform " << std::endl
		//	<< rigidTransform << std::endl 
		//	<< std::endl;
	}
	return false;
}



bool ICP::align_iteration(const std::vector<Eigen::Vector4f>& points_src, const std::vector<Eigen::Vector4f>& points_dst, Eigen::Matrix3f& R, Eigen::Vector3f& t)
{	
	std::vector<Eigen::Vector4f> points_match;
	
	for (const Eigen::Vector4f& p1 : points_src)
	{
		Eigen::Vector4f closer = points_dst[0];
		float min_distance = distance_point_to_point(p1, points_dst[0]);

		for (const Eigen::Vector4f& p2 : points_dst)
		{
			float dist = distance_point_to_point(p1, p2);
			if (dist < min_distance)
			{
				closer = p2;
				min_distance = dist;
			}
		}
		points_match.push_back(closer);
	}

	return computeRigidTransform(points_src, points_match, R, t);
}


bool ICP::align_iteration(const std::vector<Eigen::Vector4f>& points_src, const std::vector<Eigen::Vector3f>& normals, const std::vector<Eigen::Vector4f>& points_dst, Eigen::Matrix3f& R, Eigen::Vector3f& t, const DistanceMethod distance_method)
{
	std::vector<Eigen::Vector4f> points_match;

	int i = 0;
	for (const Eigen::Vector4f& p1 : points_src)
	{
		Eigen::Vector4f closer = points_dst[0];
		float min_distance, dist;
		if(distance_method == PointToPlane)
			min_distance = distance_point_to_plane(p1, normals[i].normalized(), points_dst[0]);
		else
			min_distance = distance_point_to_point(p1, points_dst[0]);


		for (const Eigen::Vector4f& p2 : points_dst)
		{
			if (distance_method == PointToPlane)
				dist = distance_point_to_plane(p1, normals[i].normalized(), p2);
			else
				dist = distance_point_to_point(p1, p2);

			if (dist < min_distance)
			{
				closer = p2;
				min_distance = dist;
			}
		}
		points_match.push_back(closer);

		++i;
	}

	return computeRigidTransform(points_src, points_match, R, t);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	static bool ComputeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix3f& R, Eigen::Vector3f& t);
///
/// @brief	Compute the rotation and translation that transform a source point set to a target point set
///
/// @author	Diego
/// @date	07/10/2015
///
/// @param	src		   		The source point set.
/// @param	dst		   		The target point set.
/// @param [in,out]	pts_dst	The rotation matrix.
/// @param [in,out]	pts_dst	The translation vector.
/// @return	True if found the transformation, false otherwise.
////////////////////////////////////////////////////////////////////////////////////////////////////
bool ICP::computeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.size() == dst.size());
	int pairSize = (int)src.size();
	Eigen::Vector4f center_src(0, 0, 0, 1), center_dst(0, 0, 0, 1);

	// 
	// Compute centroid
	//
	for (int i = 0; i < pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (float)pairSize;
	center_dst /= (float)pairSize;


	Eigen::MatrixXf S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i < pairSize; ++i)
	{
		const Eigen::Vector4f src4f = src[i] - center_src;
		const Eigen::Vector4f dst4f = dst[i] - center_dst;

		S.row(i) = (src4f / src4f.w()).head<3>();
		D.row(i) = (dst4f / dst4f.w()).head<3>();
	}
	Eigen::MatrixXf Dt = D.transpose();
	Eigen::Matrix3f H = Dt * S;
	Eigen::Matrix3f W, U, V;


	//
	// Compute SVD
	//
	//Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::FullPivHouseholderQRPreconditioner> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ColPivHouseholderQRPreconditioner> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
		std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	Eigen::Vector3f center_src_3f = (center_src / center_src.w()).head<3>();
	Eigen::Vector3f center_dst_3f = (center_dst / center_dst.w()).head<3>();

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix3f Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst_3f - R * center_src_3f;

	return true;
}