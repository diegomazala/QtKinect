#ifndef _ICP_H_
#define _ICP_H_

#include <Eigen/Dense>
#include <vector>


class ICP
{
public:

	enum DistanceMethod
	{
		PointToPoint,
		PointToPlane
	};

	ICP();
	virtual ~ICP();
	
	void setInputCloud(const std::vector<Eigen::Vector4f>& vertices);
	void setInputCloud(const std::vector<Eigen::Vector4f>& vertices, const std::vector<Eigen::Vector3f>& normals);
	void setTargetCloud(const std::vector<Eigen::Vector4f>& vertices);
	const std::vector<Eigen::Vector4f>& getResultCloud() const;

	bool align(const int iterations, const float error_precision, const DistanceMethod distance_method = PointToPoint);

	static bool align_iteration(const std::vector<Eigen::Vector4f>& points_src, const std::vector<Eigen::Vector4f>& points_dst, Eigen::Matrix3f& R, Eigen::Vector3f& t);
	static bool align_iteration(const std::vector<Eigen::Vector4f>& points_src, const std::vector<Eigen::Vector3f>& normals, const std::vector<Eigen::Vector4f>& points_dst, Eigen::Matrix3f& R, Eigen::Vector3f& t, const DistanceMethod distance_method = PointToPlane);

	Eigen::Matrix3f getRotation() const;
	Eigen::Vector3f getTranslation() const;
	Eigen::Matrix4f getTransform() const;
	static Eigen::Matrix4f getTransform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);

	static bool computeRigidTransform(const std::vector<Eigen::Vector4f>& src, const std::vector<Eigen::Vector4f>& dst, Eigen::Matrix3f& R, Eigen::Vector3f& t);

protected:

	const std::vector<Eigen::Vector4f>* input_vertices_ptr;
	const std::vector<Eigen::Vector3f>* input_normals_ptr;
	const std::vector<Eigen::Vector4f>* target_vertices_ptr;
	std::vector<Eigen::Vector4f>		vertices_icp;
	std::vector<Eigen::Vector3f>		normals_icp;
	Eigen::Matrix3f						rotation;
	Eigen::Vector3f						translation;
};




#endif // _OBJ_FILE_H_
