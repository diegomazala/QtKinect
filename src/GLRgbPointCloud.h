
#ifndef _GL_RGB_POINT_CLOUD_H_
#define _GL_RGB_POINT_CLOUD_H_

#include "GLModel.h"


class GLRgbPointCloud : public GLModel
{
public:
	GLRgbPointCloud();
	virtual ~GLRgbPointCloud();

	void render(QOpenGLShaderProgram *program);

	void setVertices(const float* vertices, uint count, uint tuple_size);
	GLuint vertexBufferId() const;

	void setColors(const float* vertices, uint count, uint tuple_size);
	GLuint colorBufferId() const;

public slots:

	void initGL();
	void cleanupGL();

private:

	uint vertexCount;
    QOpenGLBuffer vertexBuf;
	uint vertexTupleSize;
	uint vertexStride;

	uint colorCount;
	QOpenGLBuffer colorBuf;
	uint colorTupleSize;
	uint colorStride;
};


#endif // _GL_POINT_CLOUD_H_
