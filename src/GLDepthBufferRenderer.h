
#ifndef __GL_DEPTH_BUFFER_RENDERER_H__
#define __GL_DEPTH_BUFFER_RENDERER_H__

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>


class QKinectGrabber;
class QKinectPlayerCtrl;



//struct DepthPoint
//{
//	unsigned short x, y, z;
//	DepthPoint() {}
//	DepthPoint(unsigned short _x, unsigned short _y, unsigned short _z) : x(_x), y(_y), z(z) {};
//	DepthPoint(const DepthPoint& d) : x(d.x), y(d.y), z(d.z) {};
//};



class GLDepthBufferRenderer : protected QOpenGLFunctions
{
public:
	GLDepthBufferRenderer();
	virtual ~GLDepthBufferRenderer();

    void render(QOpenGLShaderProgram *program);

	void setController(QKinectPlayerCtrl& kinect_ctrl);
	void setKinectReader(QKinectGrabber& kinect_reader);
	void setDepthBuffer(const std::vector<unsigned short>& depthBuffer, unsigned short width, unsigned short height);

	std::vector<unsigned short>& getDepthBufferCloud() { return depthBufferCloud; };

	void updatePoints();

public slots:
	
	void initGL();
	void cleanupGL();


private:
    void initCubeGeometry();

    QOpenGLBuffer arrayBuf;
    QOpenGLBuffer indexBuf;

	std::vector<unsigned short> depthBufferCloud;

	float mVertexData[512];

	QKinectGrabber*			kinectReader;
	QKinectPlayerCtrl*		kinectCtrl;
};

#endif // __GL_DEPTH_BUFFER_RENDERER_H__
