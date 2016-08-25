#ifndef __RAY_BOX_H__
#define __RAY_BOX_H__


#include <Eigen/Dense>


template <typename Type>
class Ray 
{
public:
	Ray(const Eigen::Matrix<Type, 3, 1>& o, const Eigen::Matrix<Type, 3, 1>& d)
	{
		origin = o;
		direction = d;
		inv_direction = Eigen::Matrix<Type, 3, 1>(1 / d.x(), 1 / d.y(), 1 / d.z());
		sign[0] = (inv_direction.x() < 0);
		sign[1] = (inv_direction.y() < 0);
		sign[2] = (inv_direction.z() < 0);
	}

	Eigen::Matrix<Type, 3, 1> origin;
	Eigen::Matrix<Type, 3, 1> direction;
	Eigen::Matrix<Type, 3, 1> inv_direction;
	int sign[3];
};



template <typename Type>
class Box
{
public:
	Box(const Eigen::Matrix<Type, 3, 1>& vmin, const Eigen::Matrix<Type, 3, 1>& vmax)
	{
		bounds[0] = vmin;
		bounds[1] = vmax;
	}

	bool intersect(const Ray<Type> &r, Type t0, Type t1) const
	{
		Type txmin, txmax, tymin, tymax, tzmin, tzmax;

		txmin = (bounds[r.sign[0]].x() - r.origin.x()) * r.inv_direction.x();
		txmax = (bounds[1 - r.sign[0]].x() - r.origin.x()) * r.inv_direction.x();

		tymin = (bounds[r.sign[1]].y() - r.origin.y()) * r.inv_direction.y();
		tymax = (bounds[1 - r.sign[1]].y() - r.origin.y()) * r.inv_direction.y();

	
		if ((txmin > tymax) || (tymin > txmax))
			return false;

		if (tymin > txmin)
			txmin = tymin;
		if (tymax < txmax)
			txmax = tymax;

		tzmin = (bounds[r.sign[2]].z() - r.origin.z()) * r.inv_direction.z();
		tzmax = (bounds[1 - r.sign[2]].z() - r.origin.z()) * r.inv_direction.z();

		Eigen::Matrix<Type, 3, 1> tmin(txmin, tymin, tzmin);
		Eigen::Matrix<Type, 3, 1> tmax(txmax, tymax, tzmax);
				
		if ((txmin > tzmax) || (tzmin > txmax))
			return false;

		if (tzmin > txmin)
			txmin = tzmin;

		if (tzmax < txmax)
			txmax = tzmax;

		return ((txmin < t1) && (txmax > t0));
	}

	Eigen::Matrix<Type, 3, 1> bounds[2];
};


#endif // __RAY_BOX_H__