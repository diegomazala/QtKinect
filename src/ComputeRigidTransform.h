
#ifndef _COMPUTE_RIGID_TRANSFORM_H_
#define _COMPUTE_RIGID_TRANSFORM_H_

#include <Eigen/Dense>


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
static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.size() == dst.size());
	int pairSize = (int)src.size();
	Eigen::Vector3d center_src(0, 0, 0), center_dst(0, 0, 0);

	// 
	// Compute centroid
	//
	for (int i = 0; i<pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (double)pairSize;
	center_dst /= (double)pairSize;


	Eigen::MatrixXd S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i<pairSize; ++i)
	{
		S.row(i) = src[i] - center_src;
		D.row(i) = dst[i] - center_dst;
	}
	Eigen::MatrixXd Dt = D.transpose();
	Eigen::Matrix3d H = Dt * S;
	Eigen::Matrix3d W, U, V;

	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
	//	std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix3d Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst - R * center_src;
	
	return true;
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
static bool ComputeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix3f& R, Eigen::Vector3f& t)
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
	for (int i = 0; i<pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (float)pairSize;
	center_dst /= (float)pairSize;


	Eigen::MatrixXf S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i<pairSize; ++i)
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
		//std::cerr << "<Error> Decomposition error" << std::endl;
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

static bool ComputeRigidTransform(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, Eigen::Matrix4d& mat)
{
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	if (ComputeRigidTransform(src, dst, R, t))
	{
		mat.block(0, 0, 3, 3) = R;
		mat.row(3).setZero();
		mat.col(3) = t.homogeneous();
		return true;
	}
	return false;
}

static bool ComputeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix4f& mat)
{
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	if (ComputeRigidTransform(src, dst, R, t))
	{
		mat.block(0, 0, 3, 3) = R;
		mat.row(3).setZero();
		mat.col(3) = t.homogeneous();
		return true;
	}
	return false;
}


static Eigen::Matrix4f ComposeRigidTransform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
	Eigen::Matrix4f rigidTransform;
	rigidTransform.block(0, 0, 3, 3) = R;
	rigidTransform.row(3).setZero();
	rigidTransform.col(3) = t.homogeneous();
	return rigidTransform;
}


#endif // _COMPUTE_RIGID_TRANSFORM_H_