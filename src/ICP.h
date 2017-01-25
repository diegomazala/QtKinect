#ifndef _ICP_H_
#define _ICP_H_

#include <Eigen/Dense>
#include <vector>



template <typename Type>
static Type distance_point_to_point(const Eigen::Matrix<Type, 4, 1>& v1, const Eigen::Matrix<Type, 4, 1>& v2)
{
	//return (v1 - v2).norm();
	return (v1 - v2).squaredNorm();
	//return (v1.normalized() - v2.normalized()).squaredNorm();
}

#if 1
template <typename Type>
static Type distance_point_to_plane(const Eigen::Matrix<Type, 4, 1>& v, const Eigen::Matrix<Type, 3, 1>& normal, const Eigen::Matrix<Type, 4, 1>& v2)
{
	Eigen::Matrix<Type, 3, 1> n = normal.normalized();
	const Eigen::Matrix<Type, 3, 1> pt = (v / v.w()).head<3>();
	const Eigen::Matrix<Type, 4, 1> plane(n.x(), n.y(), n.z(), -pt.dot(n));	// a*x + b*y + c*z + d = 0
	return (plane.x() * v2.x() + plane.y() * v2.y() + plane.z() * v2.z() + plane.w());
}
#else

template <typename Type>
static Type distance_point_to_plane(const Eigen::Matrix<Type, 3, 1>& xp, const Eigen::Matrix<Type, 3, 1>& n, const Eigen::Matrix<Type, 3, 1>& x0)
{
	return n.dot(x0 - xp) / n.norm();
}

template <typename Type>
static Type distance_point_to_plane(const Eigen::Matrix<Type, 4, 1>& xp, const Eigen::Matrix<Type, 3, 1>& n, const Eigen::Matrix<Type, 4, 1>& x0)
{
	Eigen::Matrix<Type, 3, 1> xx0 = (x0 / x0.w()).head<3>();
	Eigen::Matrix<Type, 3, 1> xxp = (xp / xp.w()).head<3>();
	return n.dot(xx0 - xxp) / n.norm();
}
#endif






template <class Type>
class ICP
{
public:

	enum DistanceMethod
	{
		PointToPoint,
		PointToPlane
	};

	ICP(){};
	virtual ~ICP(){};
	
	void setInputCloud(const std::vector<Eigen::Matrix<Type, 4, 1>>& vertices)
	{
		input_vertices_ptr = &vertices;
		vertices_icp = vertices;
	}


	void setInputCloud(const std::vector<Eigen::Matrix<Type, 4, 1>>& vertices, const std::vector<Eigen::Matrix<Type, 3, 1>>& normals)
	{
		input_vertices_ptr = &vertices;
		input_normals_ptr = &normals;
		vertices_icp = vertices;
		normals_icp = normals;
	}


	void setTargetCloud(const std::vector<Eigen::Matrix<Type, 4, 1>>& vertices)
	{
		target_vertices_ptr = &vertices;
	}


	const std::vector<Eigen::Matrix<Type, 4, 1>>& getResultCloud() const
	{
		return vertices_icp;
	}

	bool align(const int iterations, const Type error_precision, const DistanceMethod distance_method = PointToPoint)
	{
		Eigen::Matrix<Type, 4, 4> identity = Eigen::Matrix<Type, 4, 4>::Identity();
		Eigen::Matrix<Type, 4, 4> rigidTransform = Eigen::Matrix<Type, 4, 4>::Identity();
		Eigen::Matrix<Type, 3, 3> R;
		Eigen::Matrix<Type, 3, 1> t;

		rotation = Eigen::Matrix<Type, 3, 3>::Identity();
		translation = Eigen::Matrix<Type, 3, 1>::Zero();

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

				//for (Eigen::Matrix<Type, 4, 1>& p : vertices_icp)
				for (int p = 0; p < vertices_icp.size(); ++p)
				{
					Eigen::Matrix<Type, 4, 1>& v = vertices_icp[p];
					v = rigidTransform * v;
					v /= v.w();

					Eigen::Matrix<Type, 4, 1> nn = normals_icp[i].homogeneous();
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

	static bool align_iteration(
		const std::vector<Eigen::Matrix<Type, 4, 1>>& points_src,
		const std::vector<Eigen::Matrix<Type, 4, 1>>& points_dst,
		const int filter_width,
		const Type max_distance,
		const int frame_width,
		const int frame_height,
		Eigen::Matrix<Type, 3, 3>& R,
		Eigen::Matrix<Type, 3, 1>& t)
	{
		Type sum_distances = 0;
		const int numCols = frame_width;
		const int numRows = frame_height;

		const int half_fw = filter_width / 2;

		std::vector<Eigen::Matrix<Type, 4, 1>> points_match_src;
		std::vector<Eigen::Matrix<Type, 4, 1>> points_match_dst;

		int i = 0;
		for (const Eigen::Matrix<Type, 4, 1>& p1 : points_src)
		{
			Eigen::Matrix<Type, 4, 1> closer;

			if (!p1.isZero(0.001f))
			{
				Type min_distance = (Type)FLT_MAX;
				Type dist = (Type)FLT_MAX;

				int x = i % numCols;
				int y = i / numCols;

				// x filter borders 
				int x_begin = std::max(x - half_fw, 0);
				int x_end = std::min(x + half_fw, numCols - 1);

				// y filter borders 
				int y_begin = std::max(y - half_fw, 0);
				int y_end = std::min(y + half_fw, numRows - 1);


				// computing neighbours
				for (int yy = y_begin; yy <= y_end; ++yy)
				{
					for (int xx = x_begin; xx <= x_end; ++xx)
					{
						const Eigen::Matrix<Type, 4, 1>& p2 = points_dst[yy * numCols + xx];

						dist = distance_point_to_point(p1, p2);

						if (dist < min_distance)
						{
							closer = p2;
							min_distance = dist;
						}
					}
				}

				if (min_distance < max_distance)
				{
					points_match_src.push_back(points_src[i]);
					points_match_dst.push_back(closer);

					sum_distances += min_distance;
				}
			}
			++i;
		}


		std::cout << "Size of points match      : " << points_match_dst.size() << std::endl;
		std::cout << "Sum of distances          : " << sum_distances << std::endl;
		std::cout << "Max distance allowed      : " << max_distance << std::endl;


		//export_obj("../match_src.obj", points_match_src);
		//export_obj("../match_dst.obj", points_match_dst);

		return computeRigidTransform(points_match_src, points_match_dst, R, t);
	}

	static bool align_iteration(
		const std::vector<Eigen::Matrix<Type, 4, 1>>& points_src,
		const std::vector<Eigen::Matrix<Type, 3, 1>>& normals,
		const std::vector<Eigen::Matrix<Type, 4, 1>>& points_dst,
		const int filter_width,
		const Type max_distance,
		const int frame_width,
		const int frame_height,
		Eigen::Matrix<Type, 3, 3>& R,
		Eigen::Matrix<Type, 3, 1>& t,
		const DistanceMethod distance_method = PointToPlane)
	{
		const int numCols = frame_width;
		const int numRows = frame_height;
		
		const int half_fw = filter_width / 2;

		std::vector<Eigen::Matrix<Type, 4, 1>> points_match_src;
		std::vector<Eigen::Matrix<Type, 4, 1>> points_match_dst;

		int i = 0;
		for (const Eigen::Matrix<Type, 4, 1>& p1 : points_src)
		{
			Eigen::Matrix<Type, 4, 1> closer;

			if (!p1.isZero(0.001f))
			//if (!p1.isZero(0.001f) && normals[i].z() > 0)
			{
				Type min_distance = (Type)FLT_MAX;
				Type dist = (Type)FLT_MAX;

				int x = i % numCols;
				int y = i / numCols;
				
				// x filter borders 
				int x_begin = std::max(x - half_fw, 0);
				int x_end = std::min(x + half_fw, numCols - 1);

				// y filter borders 
				int y_begin = std::max(y - half_fw, 0);
				int y_end = std::min(y + half_fw, numRows - 1);


				// computing neighbours
				for (int yy = y_begin; yy <= y_end; ++yy)
				{
					for (int xx = x_begin; xx <= x_end; ++xx)
					{
						const Eigen::Matrix<Type, 4, 1>& p2 = points_dst[yy * numCols + xx];

						if (distance_method == DistanceMethod::PointToPoint)
							dist = distance_point_to_point(p1, p2);
						else
							dist = distance_point_to_plane(p1, normals[i], p2);
							
						

						if (dist < min_distance)
						{
							closer = p2;
							min_distance = dist;
						}
					}
				}

				if (min_distance < max_distance)
				{
					points_match_src.push_back(points_src[i]);
					points_match_dst.push_back(closer);
				}


			}
			

			++i;
		}


		std::cout << "Size of points match      : " << points_match_dst.size() << std::endl;

		return computeRigidTransform(points_match_src, points_match_dst, R, t);
	}

	Eigen::Matrix<Type, 3, 3> getRotation() const { return rotation; }
	Eigen::Matrix<Type, 3, 1> getTranslation() const { return rotation; }

	Eigen::Matrix<Type, 4, 4> getTransform() const
	{
		Eigen::Matrix<Type, 4, 4> transform;
		transform.block(0, 0, 3, 3) = rotation;
		transform.row(3).setZero();
		transform.col(3) = translation.homogeneous();
		return transform;
	}

	static Eigen::Matrix<Type, 4, 4> getTransform(
		const Eigen::Matrix<Type, 3, 3>& R,
		const Eigen::Matrix<Type, 3, 1>& t)
	{
		Eigen::Matrix<Type, 4, 4> transform;
		transform.block(0, 0, 3, 3) = R;
		transform.row(3).setZero();
		transform.col(3) = t.homogeneous();
		return transform;
	}

	static bool computeRigidTransform(
		const std::vector<Eigen::Matrix<Type, 4, 1>>& src,
		const std::vector<Eigen::Matrix<Type, 4, 1>>& dst,
		Eigen::Matrix<Type, 3, 3>& R, 
		Eigen::Matrix<Type, 3, 1>& t)
	{
		//
		// Verify if the sizes of point arrays are the same 
		//
		assert(src.size() == dst.size());
		int pairSize = (int)src.size();
		Eigen::Matrix<Type, 4, 1> center_src(0, 0, 0, 1), center_dst(0, 0, 0, 1);

		// 
		// Compute centroid
		//
		for (int i = 0; i < pairSize; ++i)
		{
			center_src += src[i];
			center_dst += dst[i];
		}
		center_src /= (Type)pairSize;
		center_dst /= (Type)pairSize;


		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S(pairSize, 3), D(pairSize, 3);
		for (int i = 0; i < pairSize; ++i)
		{
			const Eigen::Matrix<Type, 4, 1> src4f = src[i] - center_src;
			const Eigen::Matrix<Type, 4, 1> dst4f = dst[i] - center_dst;

			S.row(i) = (src4f / src4f.w()).head<3>();
			D.row(i) = (dst4f / dst4f.w()).head<3>();
		}
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> Dt = D.transpose();
		Eigen::Matrix<Type, 3, 3> H = Dt * S;
		Eigen::Matrix<Type, 3, 3> W, U, V;


		//
		// Compute SVD
		//
		//Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::FullPivHouseholderQRPreconditioner> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::JacobiSVD<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>, Eigen::ColPivHouseholderQRPreconditioner> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

		if (!svd.computeU() || !svd.computeV())
		{
			std::cerr << "<Error> Decomposition error" << std::endl;
			return false;
		}

		Eigen::Matrix<Type, 3, 1> center_src_3f = (center_src / center_src.w()).head<3>();
		Eigen::Matrix<Type, 3, 1> center_dst_3f = (center_dst / center_dst.w()).head<3>();

		//
		// Compute rotation matrix and translation vector
		// 
		Eigen::Matrix<Type, 3, 3> Vt = svd.matrixV().transpose();
		R = svd.matrixU() * Vt;
		t = center_dst_3f - R * center_src_3f;

		return true;
	}

protected:

	const std::vector<Eigen::Matrix<Type, 4, 1>>*	input_vertices_ptr = nullptr;
	const std::vector<Eigen::Matrix<Type, 3, 1>>*	input_normals_ptr = nullptr;
	const std::vector<Eigen::Matrix<Type, 4, 1>>*	target_vertices_ptr = nullptr;
	std::vector<Eigen::Matrix<Type, 4, 1>>			vertices_icp;
	std::vector<Eigen::Matrix<Type, 3, 1>>			normals_icp;
	Eigen::Matrix<Type, 3, 3>						rotation;
	Eigen::Matrix<Type, 3, 1>						translation;
};
	

#endif // _ICP_H_
