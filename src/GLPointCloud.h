
#ifndef _GL_POINT_CLOUD_H_
#define _GL_POINT_CLOUD_H_

#include "GLModel.h"

class GLPointCloud : public GLModel
{
public:
	GLPointCloud();
	virtual ~GLPointCloud();

	void render(QOpenGLShaderProgram *program);

	void setVertices(const float* vertices, uint count, uint tuple_size);

	GLuint vertexBufferId() const;

public slots:

	void initGL();
	void cleanupGL();

private:

    QOpenGLBuffer vertexBuf;
	uint vertexCount;
	uint tupleSize;
	uint stride;
};

#endif // _GL_POINT_CLOUD_H_
