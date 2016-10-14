//***************************************************************************
//
// Advanced CodeColony Camera
// Philipp Crocoll, 2003
//
//***************************************************************************


#define PI 3.1415926535897932384626433832795
#define PIdiv180 (PI/180.0)

/////////////////////////////////
//Note: All angles in degrees  //
/////////////////////////////////

struct SF3dVector  //Float 3d-vect, normally used
{
	float x,y,z;
	SF3dVector(){}
	SF3dVector(float _x, float _y, float _z) :x(_x), y(_y), z(_z){}
};
struct SF2dVector
{
	float x,y;
	SF2dVector(){}
	SF2dVector(float _x, float _y) :x(_x), y(_y){}
};
SF3dVector F3dVector ( float x, float y, float z );

class Camera
{
public:
	Camera();				//inits the values (Position: (0|0|0) Target: (0|0|-1) )
	void Render ( void );	//executes some glRotates and a glTranslate command
							//Note: You should call glLoadIdentity before using Render

	void Move ( SF3dVector Direction );
	void Move( float x, float y, float z);
	void RotateX ( float Angle );
	void RotateY ( float Angle );
	void RotateZ ( float Angle );

	void MoveForward ( float Distance );
	void MoveUpward ( float Distance );
	void StrafeRight ( float Distance );
	
	float* ProjectionMatrix();
	float* ViewMatrix();

	float FovY;
	float Aspect;
	float Near;
	float Far;

	SF3dVector Position;
	SF3dVector ViewDir;
	SF3dVector RightVector;
	SF3dVector UpVector;

private:
	float RotatedX, RotatedY, RotatedZ;
	float projectionMatrix[16];
	float viewMatrix[16];
};


