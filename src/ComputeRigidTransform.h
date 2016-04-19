
#ifndef _COMPUTE_RIGID_TRANSFORM_H_
#define _COMPUTE_RIGID_TRANSFORM_H_

#include <Eigen/Dense>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix3d& R, Eigen::Vector3d& t);
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
template<typename Type>
static bool ComputeRigidTransform(const std::vector<Eigen::Matrix<Type, 3, 1>>& src, const std::vector<Eigen::Matrix<Type, 3, 1>>& dst, Eigen::Matrix<Type, 3, 3>& R, Eigen::Matrix<Type, 3, 1>& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.size() == dst.size());
	int pairSize = (int)src.size();
	Eigen::Matrix<Type, 3, 1> center_src(0, 0, 0), center_dst(0, 0, 0);

	// 
	// Compute centroid
	//
	for (int i = 0; i<pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (Type)pairSize;
	center_dst /= (Type)pairSize;


	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i<pairSize; ++i)
	{
		S.row(i) = src[i] - center_src;
		D.row(i) = dst[i] - center_dst;
	}
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> Dt = D.transpose();
	Eigen::Matrix<Type, 3, 3> H = Dt * S;
	Eigen::Matrix<Type, 3, 3> W, U, V;

	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
	//	std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix<Type, 3, 3> Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst - R * center_src;
	
	return true;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix3d& R, Eigen::Vector3d& t);
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
template<typename Type>
static bool ComputeRigidTransform(const std::vector<Eigen::Matrix<Type, 4, 1>>& src, const std::vector<Eigen::Matrix<Type, 4, 1>>& dst, Eigen::Matrix<Type, 3, 3>& R, Eigen::Matrix<Type, 3, 1>& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.size() == dst.size());
	int pairSize = (int)src.size();
	Eigen::Matrix<Type, 3, 1> center_src(0, 0, 0), center_dst(0, 0, 0);

	// 
	// Compute centroid
	//
	for (int i = 0; i < pairSize; ++i)
	{
		center_src += (src[i] / src[i][3]).head<3>();
		center_dst += (dst[i] / dst[i][3]).head<3>();
	}
	center_src /= (Type)pairSize;
	center_dst /= (Type)pairSize;


	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i < pairSize; ++i)
	{
		S.row(i) = (src[i] / src[i][3]).head<3>() - center_src;
		D.row(i) = (dst[i] / dst[i][3]).head<3>() - center_dst;
	}
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> Dt = D.transpose();
	Eigen::Matrix<Type, 3, 3> H = Dt * S;
	Eigen::Matrix<Type, 3, 3> W, U, V;

	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
		//	std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix<Type, 3, 3> Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst - R * center_src;

	return true;
}



template<typename Type, const int Rows>
static bool ComputeRigidTransform(const std::vector<Eigen::Matrix<Type, Rows, 1>>& src, const std::vector<Eigen::Matrix<Type, Rows, 1>>& dst, Eigen::Matrix<Type, 4, 4>& mat)
{
	Eigen::Matrix<Type, 3, 3> R;
	Eigen::Matrix<Type, 3, 1> t;
	if (ComputeRigidTransform(src, dst, R, t))
	{
		mat.block(0, 0, 3, 3) = R;
		mat.row(3).setZero();
		mat.col(3) = t.homogeneous();
		return true;
	}
	return false;
}


template<typename Type>
static Eigen::Matrix<Type, 4, 4> ComposeRigidTransform(const Eigen::Matrix<Type, 3, 3>& R, const Eigen::Matrix<Type, 3, 1>& t)
{
	Eigen::Matrix<Type, 4, 4> rigidTransform;
	rigidTransform.block(0, 0, 3, 3) = R;
	rigidTransform.row(3).setZero();
	rigidTransform.col(3) = t.homogeneous();
	return rigidTransform;
}


#endif // _COMPUTE_RIGID_TRANSFORM_H_