
#ifndef _GL_KINECT_FRAME_H_
#define _GL_KINECT_FRAME_H_

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>

class KinectFrame;

class GLKinectFrame : protected QOpenGLFunctions
{
public:
    GLKinectFrame();
    virtual ~GLKinectFrame();

	void render(QOpenGLShaderProgram *program);

	void setFrame(KinectFrame* frame);

public slots:

	void initGL();
	void cleanupGL();

private:

    QOpenGLBuffer vertexBuf;
    QOpenGLBuffer colorBuf;
	QOpenGLBuffer normalBuf;
};

#endif // _GL_KINECT_FRAME_H_
