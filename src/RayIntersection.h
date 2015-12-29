#ifndef __RAY_INTERSECTION_H__
#define __RAY_INTERSECTION_H__

#include <Eigen/Dense>


template <typename Type>
static Eigen::Matrix<Type, 3, 1> get_normal(
	const Eigen::Matrix<Type, 3, 1>& p1, 
	const Eigen::Matrix<Type, 3, 1>& p2, 
	const Eigen::Matrix<Type, 3, 1>& p3)
{
	Eigen::Matrix<Type, 3, 1> u = p2 - p1;
	Eigen::Matrix<Type, 3, 1> v = p3 - p1;

	return v.cross(u).normalized();
}



// http://www.graphics.cornell.edu/pubs/1997/MT97.html
template <typename Type>
static bool triangle_intersection(
	const Eigen::Matrix<Type, 3, 1>& p,
	const Eigen::Matrix<Type, 3, 1>& d,
	const Eigen::Matrix<Type, 3, 1>& v0,
	const Eigen::Matrix<Type, 3, 1>& v1,
	const Eigen::Matrix<Type, 3, 1>& v2,
	Eigen::Matrix<Type, 3, 1>& hit)
{
	float a, f, u, v;
	const Eigen::Matrix<Type, 3, 1> e1 = v1 - v0;
	const Eigen::Matrix<Type, 3, 1> e2 = v2 - v0;

	const Eigen::Matrix<Type, 3, 1> h = d.cross(e2);
	a = e1.dot(h);

	if (a > -0.00001f && a < 0.00001f)
		return false;

	f = 1.0f / a;
	const Eigen::Matrix<Type, 3, 1> s = p - v0;
	u = f * s.dot(h);

	if (u < 0.0f || u > 1.0f)
		return false;

	const Eigen::Matrix<Type, 3, 1> q = s.cross(e1);
	v = f * d.dot(q);

	if (v < 0.0f || u + v > 1.0f)
		return false;

	float t = f * e2.dot(q);

	if (t > 0.00001f) // ray intersection
	{
		hit = p + (d * t);
		return true;
	}
	else
		return false;
}



// http://www.openprocessing.org/sketch/45539
template <typename Type>
static int sphere_intersection(
	const Eigen::Matrix<Type, 3, 1>& rayP,
	const Eigen::Matrix<Type, 3, 1>& dir,
	const Eigen::Matrix<Type, 3, 1>& sphereCenter, 
	Type sphereRadius,
	Eigen::Matrix<Type, 3, 1>& hit1, 
	Eigen::Matrix<Type, 3, 1>& hit2,
	Eigen::Matrix<Type, 3, 1>& hit1Normal, 
	Eigen::Matrix<Type, 3, 1>& hit2Normal)
{
	Eigen::Matrix<Type, 3, 1> e = dir.normalized();
	Eigen::Matrix<Type, 3, 1> h = sphereCenter - rayP;
	float lf = e.dot(h);                      // lf=e.h
	float s = pow(sphereRadius, 2) - h.dot(h) + pow(lf, 2);   // s=r^2-h^2+lf^2
	if (s < 0.0)
		return 0;                    // no intersection points ?
	s = sqrt(s);                              // s=sqrt(r^2-h^2+lf^2)

	int result = 0;
	if (lf < s)                               // S1 behind A ?
	{
		if (lf + s >= 0)                          // S2 before A ?}
		{
			s = -s;                               // swap S1 <-> S2}
			result = 1;                           // one intersection point
		}
	}
	else
		result = 2;                          // 2 intersection points

	hit1 = e * (lf - s) + rayP;
	hit2 = e * (lf + s) + rayP;

	hit1Normal = hit1 - sphereCenter;
	hit1Normal.normalize();

	hit2Normal = hit2 - sphereCenter;
	hit2Normal.normalize();

	return result;
}


template <typename Type>
bool plane_intersection(
	const Eigen::Matrix<Type, 3, 1>& p,
	const Eigen::Matrix<Type, 3, 1>& dir,
	const Eigen::Matrix<Type, 3, 1>& p1,
	const Eigen::Matrix<Type, 3, 1>& p2,
	const Eigen::Matrix<Type, 3, 1>& p3,
	Eigen::Matrix<Type, 3, 1>& hit)
{
	Eigen::Matrix<Type, 3, 1> r1 = p;
	Eigen::Matrix<Type, 3, 1> r2 = r1 + dir;

	Eigen::Matrix<Type, 3, 1> v1 = p2 - p1;
	Eigen::Matrix<Type, 3, 1> v2 = p3 - p1;
	Eigen::Matrix<Type, 3, 1> v3 = v1.cross(v2);

	Eigen::Matrix<Type, 3, 1> vRotRay1 = Eigen::Matrix<Type, 3, 1>(v1.dot(r1 - p1), v2.dot(r1 - p1), v3.dot(r1 - p1));
	Eigen::Matrix<Type, 3, 1> vRotRay2 = Eigen::Matrix<Type, 3, 1>(v1.dot(r2 - p1), v2.dot(r2 - p1), v3.dot(r2 - p1));
	
	// Return now if ray will never intersect plane (they're parallel)
	if (vRotRay1.z() == vRotRay2.z())
		return false;

	// Find 2D plane coordinates (fX, fY) that the ray interesects with
	float fPercent = vRotRay1.z() / (vRotRay2.z() - vRotRay1.z());

	hit = Eigen::Matrix<Type, 3, 1>(r1 + (r1 - r2) * fPercent);

	return true;
}



template <typename Type>
static bool plane_intersection(
	const Eigen::Matrix<Type, 3, 1>& ray_p,
	const Eigen::Matrix<Type, 3, 1>& ray_dir,
	const Eigen::Matrix<Type, 3, 1>& plane_p,
	const Eigen::Matrix<Type, 3, 1>& plane_dir,
	Eigen::Vector3f& hit)
{
	const Eigen::Matrix<Type, 3, 1>& plane_p1 = plane_p;
	Eigen::Matrix<Type, 3, 1> dir = plane_dir;
	dir.normalize();

	Eigen::Matrix<Type, 3, 1> difNorm;
	if (dir.x() == 1.0f)
	{
		difNorm = Eigen::Matrix<Type, 3, 1>(0, 1, 0);
	}
	else if (dir.y() == 1.0f)
	{
		difNorm = Eigen::Matrix<Type, 3, 1>(1, 0, 0);
	}
	else if (dir.z() == 1.0f)
	{
		difNorm = Eigen::Matrix<Type, 3, 1>(1, 0, 0);
	}
	else
	{
		difNorm = Eigen::Matrix<Type, 3, 1>(1, 1, 1);
		difNorm.normalize();
	}

	Eigen::Matrix<Type, 3, 1> u = dir.cross(difNorm);
	Eigen::Matrix<Type, 3, 1> v = dir.cross(u);
	Eigen::Matrix<Type, 3, 1> plane_p2 = plane_p1 + u.normalized();
	Eigen::Matrix<Type, 3, 1> plane_p3 = plane_p1 + v.normalized();

	return plane_intersection(
		ray_p,
		ray_dir,
		plane_p1,
		plane_p2,
		plane_p3,
		hit);
}



template <typename Type>
static bool quad_intersection(
	const Eigen::Matrix<Type, 3, 1>& p,
	const Eigen::Matrix<Type, 3, 1>& d,
	const Eigen::Matrix<Type, 3, 1>& p1,
	const Eigen::Matrix<Type, 3, 1>& p2,
	const Eigen::Matrix<Type, 3, 1>& p3,
	const Eigen::Matrix<Type, 3, 1>& p4,
	Eigen::Matrix<Type, 3, 1>& hit)
{

	return (triangle_intersection(p, d, p1, p2, p3, hit) || triangle_intersection(p, d, p3, p4, p1, hit));
}



template <typename Type>
static int box_ntersection(const Eigen::Vector3f& p,
	const Eigen::Matrix<Type, 3, 1>& dir,
	const Eigen::Matrix<Type, 3, 1>& boxCenter,
	Type boxWidth,
	Type boxHeigth,
	Type boxDepth,
	Eigen::Matrix<Type, 3, 1>& hit1, 
	Eigen::Matrix<Type, 3, 1>& hit2,
	Eigen::Matrix<Type, 3, 1>& hit1Normal, 
	Eigen::Matrix<Type, 3, 1>& hit2Normal)
{
	Type x2 = boxWidth * 0.5f;
	Type y2 = boxHeigth * 0.5f;
	Type z2 = boxDepth * 0.5f;

	Eigen::Matrix<Type, 3, 1> p1(-x2, y2, -z2);
	Eigen::Matrix<Type, 3, 1> p2(x2, y2, -z2);
	Eigen::Matrix<Type, 3, 1> p3(x2, y2, z2);
	Eigen::Matrix<Type, 3, 1> p4(-x2, y2, z2);

	Eigen::Matrix<Type, 3, 1> p5(-x2, -y2, -z2);
	Eigen::Matrix<Type, 3, 1> p6(x2, -y2, -z2);
	Eigen::Matrix<Type, 3, 1> p7(x2, -y2, z2);
	Eigen::Matrix<Type, 3, 1> p8(-x2, -y2, z2);

	p1 += boxCenter;
	p2 += boxCenter;
	p3 += boxCenter;
	p4 += boxCenter;
	p5 += boxCenter;
	p6 += boxCenter;
	p7 += boxCenter;
	p8 += boxCenter;

	Eigen::Matrix<Type, 3, 1> hit[2];
	Eigen::Matrix<Type, 3, 1> hitNormal[2];
	int hitCount = 0;

	// check top
	if (quad_intersection(p,
		dir,
		p1, p2, p3, p4,
		&hit[hitCount]))
	{
		hitNormal[hitCount] = get_normal(p1, p2, p3);
		hitCount++;
	}

	// check bottom
	if (quad_intersection(p,
		dir,
		p5, p8, p7, p6,
		&hit[hitCount]))
	{
		hitNormal[hitCount] = get_normal(p5, p8, p7);
		hitCount++;
	}

	// check front
	if (hitCount < 2 && quad_intersection(p,
		dir,
		p4, p3, p7, p8,
		&hit[hitCount]))
	{
		hitNormal[hitCount] = get_normal(p4, p3, p7);
		hitCount++;
	}

	// check back
	if (hitCount < 2 && quad_intersection(p,
		dir,
		p1, p5, p6, p2,
		&hit[hitCount]))
	{
		hitNormal[hitCount] = get_normal(p1, p5, p6);
		hitCount++;
	}

	// check left
	if (hitCount < 2 && quad_intersection(p,
		dir,
		p1, p4, p8, p5,
		&hit[hitCount]))
	{
		hitNormal[hitCount] = get_normal(p1, p4, p8);
		hitCount++;
	}

	// check right
	if (hitCount < 2 && quad_intersection(p,
		dir,
		p2, p6, p7, p3,
		&hit[hitCount]))
	{
		hitNormal[hitCount] = get_normal(p2, p6, p7);
		hitCount++;
	}

	if (hitCount > 0)
	{
		if (hitCount > 1)
		{
			if ((p - hit[0]).norm() < (p - hit[1]).norm())
			{
				hit1 = hit[0];
				hit2 = hit[1];

				hit1Normal = hitNormal[0];
				hit2Normal = hitNormal[1];
			}
			else
			{
				hit1 = hit[1];
				hit2 = hit[0];

				hit1Normal = hitNormal[1];
				hit2Normal = hitNormal[0];
			}
		}
		else
		{
			hit1 = hit[0];
			hit1Normal = hitNormal[0];
		}
	}

	return hitCount;
}



template <typename Type>
static int box_intersection(
	const Eigen::Vector3f& p,
	const Eigen::Vector3f& dir,
	const Eigen::Vector3f& boxCenter,
	Type boxWidth,
	Type boxHeigth,
	Type boxDepth,
	Eigen::Vector3f& hit1, 
	Eigen::Vector3f& hit2,
	Eigen::Vector3f& hit1Normal,
	Eigen::Vector3f& hit2Normal)
{
	Eigen::Vector3f hitRet1, hitRet2;
	Eigen::Vector3f hitRet1Normal, hitRet2Normal;

	int ret = box_intersection(
		p, 
		dir,
		boxCenter,
		boxWidth, boxHeigth, boxDepth,
		&hitRet1, 
		&hitRet2,
		&hitRet1Normal, 
		&hitRet2Normal);

	if (ret > 0)
	{
		hit1 = hitRet1;
		hit1Normal = hitRet1Normal;

		if (ret > 1)
		{
			hit2 = hitRet2;
			hit2Normal = hitRet2Normal;
		}
	}

	return ret;
}



#endif // __RAY_INTERSECTION_H__