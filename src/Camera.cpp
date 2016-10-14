#include "Camera.h"
#include <math.h>
#include <iostream>

#define SQR(x) (x*x)

#define NULL_VECTOR F3dVector(0.0f,0.0f,0.0f)

SF3dVector F3dVector ( float x, float y, float z )
{
	SF3dVector tmp;
	tmp.x = x;
	tmp.y = y;
	tmp.z = z;
	return tmp;
}

float GetF3dVectorLength( SF3dVector * v)
{
	return (float)(sqrt(SQR(v->x)+SQR(v->y)+SQR(v->z)));
}

SF3dVector Normalize3dVector( SF3dVector v)
{
	SF3dVector res;
	float l = GetF3dVectorLength(&v);
	if (l == 0.0f) return NULL_VECTOR;
	res.x = v.x / l;
	res.y = v.y / l;
	res.z = v.z / l;
	return res;
}

SF3dVector operator+ (SF3dVector v, SF3dVector u)
{
	SF3dVector res;
	res.x = v.x+u.x;
	res.y = v.y+u.y;
	res.z = v.z+u.z;
	return res;
}
SF3dVector operator- (SF3dVector v, SF3dVector u)
{
	SF3dVector res;
	res.x = v.x-u.x;
	res.y = v.y-u.y;
	res.z = v.z-u.z;
	return res;
}


SF3dVector operator* (SF3dVector v, float r)
{
	SF3dVector res;
	res.x = v.x*r;
	res.y = v.y*r;
	res.z = v.z*r;
	return res;
}

SF3dVector CrossProduct (const SF3dVector& u, const SF3dVector& v)
{
	SF3dVector resVector;
	resVector.x = u.y * v.z - u.z * v.y;
	resVector.y = u.z * v.x - u.x * v.z;
	resVector.z = u.x * v.y - u.y * v.x;

	return resVector;
}

float DotProduct(const SF3dVector& u, const SF3dVector& v)
{
	return u.x * v.x + u.y * v.y + u.z * v.z;
}

float operator* (SF3dVector v, SF3dVector u)	//dot product
{
	return v.x*u.x+v.y*u.y+v.z*u.z;
}




/***************************************************************************************/

Camera::Camera():
	FovY(45.f),
	Aspect(1.778f),
	Near(0.1f),
	Far(512.f),
	Position(0.0f, 0.0f, 0.0f),
	ViewDir( 0.0f, 0.0f, -1.0f),
	RightVector(1.0f, 0.0f, 0.0f),
	UpVector(0.0f, 1.0f, 0.0f)
{
}

void Camera::Move(float x, float y, float z)
{
	Position = Position + SF3dVector(x, y, z);
}

void Camera::Move (SF3dVector Direction)
{
	Position = Position + Direction;
}

void Camera::RotateX (float Angle)
{
	RotatedX += Angle;
	
	//Rotate viewdir around the right vector:
	ViewDir = Normalize3dVector(
		ViewDir * cos(Angle * (float)PIdiv180)
		+ UpVector * sin(Angle * (float)PIdiv180));

	//now compute the new UpVector (by cross product)
	UpVector = CrossProduct(ViewDir, RightVector) * (-1.f);

	
}

void Camera::RotateY (float Angle)
{
	RotatedY += Angle;
	
	//Rotate viewdir around the up vector:
	ViewDir = Normalize3dVector(
		ViewDir * cos(Angle * (float)PIdiv180)
		- RightVector * sin(Angle * (float)PIdiv180));

	//now compute the new RightVector (by cross product)
	RightVector = CrossProduct(ViewDir, UpVector);
}

void Camera::RotateZ (float Angle)
{
	RotatedZ += Angle;
	
	//Rotate viewdir around the right vector:
	RightVector = Normalize3dVector(
		RightVector * cos(Angle * (float)PIdiv180)
		+ UpVector * sin(Angle * (float)PIdiv180));

	//now compute the new UpVector (by cross product)
	UpVector = CrossProduct(ViewDir,  RightVector) * (-1.f);
}

void Camera::Render( void )
{

	//The point at which the camera looks:
	SF3dVector ViewPoint = Position+ViewDir;

	//as we know the up vector, we can easily use gluLookAt:
	//gluLookAt(	Position.x,Position.y,Position.z,
	//			ViewPoint.x,ViewPoint.y,ViewPoint.z,
	//			UpVector.x,UpVector.y,UpVector.z);

}

void Camera::MoveForward( float Distance )
{
	Position = Position + (ViewDir*-Distance);
}

void Camera::StrafeRight ( float Distance )
{
	Position = Position + (RightVector*Distance);
}

void Camera::MoveUpward( float Distance )
{
	Position = Position + (UpVector*Distance);
}

float* Camera::ViewMatrix()
{
	SF3dVector ViewPoint = Position + ViewDir;
	SF3dVector f = Normalize3dVector(ViewPoint - Position);
	SF3dVector u(0, 1, 0);
	SF3dVector s = Normalize3dVector(CrossProduct(f, u));
	u = CrossProduct(s, f);

	
#if 0
	// Row Major
	viewMatrix[0] = s.x;
	viewMatrix[1] = s.y;
	viewMatrix[2] = s.z;
	viewMatrix[3] = -DotProduct(s, ViewDir);

	viewMatrix[4] = u.x;
	viewMatrix[5] = u.y;
	viewMatrix[6] = u.z;
	viewMatrix[7] = -DotProduct(u, ViewDir);

	viewMatrix[8]  = -f.x;
	viewMatrix[9]  = -f.y;
	viewMatrix[10] = -f.z;
	viewMatrix[11] = DotProduct(f, ViewDir);

	viewMatrix[12] = Position.x;
	viewMatrix[13] = Position.y;
	viewMatrix[14] = Position.z;
	viewMatrix[15] = 1;
#else
	// Column Major
	viewMatrix[0] = s.x;
	viewMatrix[4] = s.y;
	viewMatrix[8] = s.z;
	viewMatrix[12] = -DotProduct(s, ViewPoint);

	viewMatrix[1] = u.x;
	viewMatrix[5] = u.y;
	viewMatrix[9] = u.z;
	viewMatrix[13] = -DotProduct(u, ViewPoint);

	viewMatrix[2] = -f.x;
	viewMatrix[6] = -f.y;
	viewMatrix[10] = -f.z;
	viewMatrix[14] = DotProduct(f, ViewPoint);

	viewMatrix[3] = Position.x;
	viewMatrix[7] = Position.y;
	viewMatrix[11] = Position.z;
	viewMatrix[15] = 1;
#endif
	return viewMatrix;
}




float* Camera::ProjectionMatrix()
{
	//assert(Aspect > 0);
	//assert(Far > Near);

	// matriz zero
	memset(projectionMatrix, 0, sizeof(float) * 16);

	const float fovy_rad = FovY * PIdiv180;
	const float tan_half_fovy = tan(fovy_rad / 2.0f);

	projectionMatrix[0] = 1.0f / (Aspect * tan_half_fovy);
	projectionMatrix[5] = 1.0f / (tan_half_fovy);
	projectionMatrix[5] = -(Far + Near) / (Far - Near);

#if 0
	// Row Major
	projectionMatrix[14] = -1.0f;
	projectionMatrix[11] = -(2.0f * Far * Near) / (Far - Near);
#else
	// Column Major
	projectionMatrix[11] = -1.0f;
	projectionMatrix[14] = -(2.0f * Far * Near) / (Far - Near);
#endif 

	return projectionMatrix;
}