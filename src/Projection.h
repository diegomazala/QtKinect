#ifndef __PROJECTION_H__
#define __PROJECTION_H__

#include <Eigen/Dense>
#include <vector>


#define DegToRad(angle_degrees) (angle_degrees * M_PI / 180.0)		// Converts degrees to radians.
#define RadToDeg(angle_radians) (angle_radians * 180.0 / M_PI)		// Converts radians to degrees.


template<typename Type>
static Eigen::Matrix<Type, 4, 4> perspective_matrix(Type fovy, Type aspect_ratio, Type near_plane, Type far_plane)
{
	Eigen::Matrix<Type, 4, 4> out = Eigen::Matrix<Type, 4, 4>::Zero();

	const float	y_scale = (float)(1.0 / tan((fovy / 2.0)*(M_PI / 180.0)));
	const float	x_scale = y_scale / aspect_ratio;
	const float	depth_length = far_plane - near_plane;

	out(0, 0) = x_scale;
	out(1, 1) = y_scale;
	out(2, 2) = -((far_plane + near_plane) / depth_length);
	out(3, 2) = -1.0;
	out(2, 3) = -((2 * near_plane * far_plane) / depth_length);

	return out;
}


template<typename Type>
static Eigen::Matrix<Type, 4, 4> perspective_matrix_inverse(Type fovy, Type aspect_ratio, Type near_plane, Type far_plane)
{
	Eigen::Matrix<Type, 4, 4> out = Eigen::Matrix<Type, 4, 4>::Zero();

	const float	y_scale = (float)(1.0 / tan((fovy / 2.0)*(M_PI / 180.0)));
	const float	x_scale = y_scale / aspect_ratio;
	const float	depth_length = far_plane - near_plane;

	out(0, 0) = static_cast<Type>(1.0 / x_scale);
	out(1, 1) = static_cast<Type>(1.0 / y_scale);
	out(2, 3) = static_cast<Type>(-1.0f);
	out(3, 2) = static_cast<Type>(-1.0f / ((2 * near_plane * far_plane) / depth_length));
	out(3, 3) = static_cast<Type>(((far_plane + near_plane) / depth_length) / ((2 * near_plane * far_plane) / depth_length));

	return out;
}


template<typename Type>
static Eigen::Matrix<Type, 3, 1> vertex_to_window_coord(Eigen::Matrix<Type, 4, 1> p3d, Type fovy, Type aspect_ratio, Type near_plane, Type far_plane, int window_width, int window_height)
{
	const Eigen::Matrix<Type, 4, 4> proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

	const Eigen::Matrix<Type, 4, 1> p_clip = proj * p3d;

	const Eigen::Matrix<Type, 3, 1> p_ndc = (p_clip / p_clip.w()).head<3>();

	Eigen::Matrix<Type, 3, 1> p_window;
	p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
	p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;
	p_window.z() = (far_plane - near_plane) / 2.0f * p_ndc.z() + (far_plane + near_plane) / 2.0f;

	return p_window;
}


template<typename Type>
static Eigen::Matrix<Type, 2, 1> vertex_to_window_coord(Eigen::Matrix<Type, 4, 1> p3d, const Eigen::Matrix<Type, 4, 4>& proj, int window_width, int window_height)
{
	const Eigen::Matrix<Type, 4, 1> p_clip = proj * p3d;

	const Eigen::Matrix<Type, 3, 1> p_ndc = (p_clip / p_clip.w()).head<3>();

	Eigen::Matrix<Type, 2, 1> p_window;
	p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
	p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;

	return p_window;
}


template<typename Type>
static Eigen::Matrix<Type, 3, 1> window_coord_to_3d(Eigen::Matrix<Type, 2, 1> pixel, Type depth, Type fovy, Type aspect_ratio, Type near_plane, Type far_plane, int window_width, int window_height)
{
	Eigen::Matrix<Type, 3, 1> ndc;
	ndc.x() = (pixel.x() - (window_width / static_cast<Type>(2.0))) / (window_width / static_cast<Type>(2.0));
	ndc.y() = (pixel.y() - (window_height / static_cast<Type>(2.0))) / (window_height / static_cast<Type>(2.0));
	ndc.z() = static_cast<Type>(-1.0);

	const Eigen::Matrix<Type, 3, 1> clip = ndc * depth;

	const Eigen::Matrix<Type, 4, 4> proj_inv = perspective_matrix_inverse(fovy, aspect_ratio, near_plane, far_plane);
	const Eigen::Matrix<Type, 4, 1> vertex_proj_inv = proj_inv * clip.homogeneous();

	Eigen::Matrix<Type, 3, 1> p3d_final;
	p3d_final.x() = -vertex_proj_inv.x();
	p3d_final.y() = -vertex_proj_inv.y();
	p3d_final.z() = -depth;

	return p3d_final;
}




template<typename Type>
static Eigen::Matrix<Type, 3, 1> window_coord_to_3d(Eigen::Matrix<Type, 2, 1> pixel, Type depth, const Eigen::Matrix<Type, 4, 4>& inverse_projection, int window_width, int window_height)
{
	Eigen::Matrix<Type, 3, 1> ndc;
	ndc.x() = (pixel.x() - (window_width / static_cast<Type>(2.0))) / (window_width / static_cast<Type>(2.0));
	ndc.y() = (pixel.y() - (window_height / static_cast<Type>(2.0))) / (window_height / static_cast<Type>(2.0));
	ndc.z() = static_cast<Type>(-1.0);

	const Eigen::Matrix<Type, 3, 1> clip = ndc * depth;

	const Eigen::Matrix<Type, 4, 1> vertex_proj_inv = inverse_projection * clip.homogeneous();

	Eigen::Matrix<Type, 3, 1> p3d_final;
	p3d_final.x() = -vertex_proj_inv.x();
	p3d_final.y() = -vertex_proj_inv.y();
	p3d_final.z() = depth;

	return p3d_final;
}


template<class Type>
Eigen::Matrix<Type, 4, 4> lookat_matrix(Eigen::Matrix<Type, 3, 1> const & eye, Eigen::Matrix<Type, 3, 1> const & center, Eigen::Matrix<Type, 3, 1> const & up)
{
	typedef Eigen::Matrix<Type, 4, 4> Matrix4;
	typedef Eigen::Matrix<Type, 3, 1> Vector3;

	Vector3 f = (center - eye).normalized();
	Vector3 u = up.normalized();
	Vector3 s = f.cross(u).normalized();
	u = s.cross(f);

	Matrix4 res;
	res <<  s.x(),s.y(),s.z(),-s.dot(eye),
		u.x(),u.y(),u.z(),-u.dot(eye),
		-f.x(),-f.y(),-f.z(),f.dot(eye),
		0,0,0,1;

	return res;
}

#if 0
static void test_projection()
{
	Eigen::Vector4f p3d(-0.5f, -0.5f, -0.88f, 1.0f);
	Eigen::Vector3f pixel(285.71f, 5.71f, 88.73f);

	float window_width = 1280.0f;
	float window_height = 720.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float fovy = 60.0f;
	float aspect_ratio = window_width / window_height;
	float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
	float x_scale = y_scale / aspect_ratio;
	float depth_length = far_plane - near_plane;

	Eigen::Matrix4f Mdv = Eigen::Matrix4f::Identity();
	Mdv.col(3) << 0.f, 0.f, 0.0f, 1.f;

	Eigen::Matrix4f Proj = perspective_matrix(fovy, aspect_ratio, near_plane, far_plane);

	Eigen::Vector4f p_clip = Proj * Mdv * p3d;

	Eigen::Vector3f p_ndc = (p_clip / p_clip.w()).head<3>();

	Eigen::Vector3f p_window;
	p_window.x() = window_width / 2.0f * p_ndc.x() + window_width / 2.0f;
	p_window.y() = window_height / 2.0f * p_ndc.y() + window_height / 2.0f;
	p_window.z() = (far_plane - near_plane) / 2.0f * p_ndc.z() + (far_plane + near_plane) / 2.0f;

	//Assert::IsTrue(pixel.isApprox(p_window, 0.01f));
}


static void test_window_coord_to_3d_world()
{
	Eigen::Vector3f p3d(-0.5f, -0.5f, -0.88f);
	Eigen::Vector2f pixel(285.716888f, 5.716888f);
	float depth = p3d.z();

	float window_width = 1280.0f;
	float window_height = 720.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float fovy = 60.0f;
	float aspect_ratio = window_width / window_height;
	float y_scale = (float)1.0 / tan((fovy / 2.0)*(M_PI / 180.0));
	float x_scale = y_scale / aspect_ratio;
	float depth_length = far_plane - near_plane;

	Eigen::Vector3f ndc;
	ndc.x() = (pixel.x() - (window_width / 2.0f)) / (window_width / 2.0f);
	ndc.y() = (pixel.y() - (window_height / 2.0f)) / (window_height / 2.0f);
	ndc.z() = -1.0f;

	Eigen::Vector3f clip = ndc * depth;

	Eigen::Matrix4f proj_inv = perspective_matrix_inverse(fovy, aspect_ratio, near_plane, far_plane);
	Eigen::Vector4f vertex_proj_inv = proj_inv * clip.homogeneous();

	Eigen::Vector3f p3d_out = -vertex_proj_inv.head<3>();
	p3d_out.z() = depth;

	//Assert::IsTrue(p3d_out.isApprox(p3d, 0.01f));
}
#endif

#endif // __PROJECTION_H__