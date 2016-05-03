
#ifndef _GL_POINT_CLOUD_H_
#define _GL_POINT_CLOUD_H_

#include "GLModel.h"


class GLPointCloud : public QOpenGLFunctions
{
public:
	GLPointCloud();
	virtual ~GLPointCloud();

	void render(QOpenGLShaderProgram *program);
	
	QMatrix4x4& transform();

	void setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program);
	std::shared_ptr<QOpenGLShaderProgram> getShaderProgram();

	void setVertices(const float* vertices, uint count, uint tuple_size);
	void updateVertices(const float* vertices);
	GLuint vertexBufferId() const;

	void setNormals(const float* normals, uint count, uint tuple_size);
	void updateNormals(const float* vertices);
	GLuint normalBufferId() const;

	void setColor(const QVector3D& c){ color = c; }

public slots:

	virtual void initGL();
	virtual void cleanupGL();

private:
	std::shared_ptr<QOpenGLShaderProgram>	shaderProgram;
	QMatrix4x4								transformMatrix;

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
