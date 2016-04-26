
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

	void setNormals(const float* normals, uint count, uint tuple_size);
	GLuint normalBufferId() const;

	void setColor(const QVector3D& c){ color = c; }

public slots:

	void initGL();
	void cleanupGL();

private:

    QOpenGLBuffer vertexBuf;
	uint vertexCount;
	uint vertexTupleSize;
	uint vertexStride;

	QOpenGLBuffer normalBuf;
	uint normalCount;
	uint normalTupleSize;
	uint normalStride;

	QVector3D color;
};


#endif // _GL_POINT_CLOUD_H_
