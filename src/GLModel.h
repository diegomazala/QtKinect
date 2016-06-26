
#ifndef _GL_MODEL_H_
#define _GL_MODEL_H_

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <memory>

class GLModel : protected QOpenGLFunctions
{
public:
	GLModel();
	virtual ~GLModel();

	virtual void render(QOpenGLShaderProgram *program);

	QMatrix4x4& transform();

	void setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program);
	std::shared_ptr<QOpenGLShaderProgram> getShaderProgram();

	void setVertices(const float* vertices, uint count, uint tuple_size);
	void updateVertices(const float* vertices);
	GLuint vertexBufferId() const;

	void setNormals(const float* normals, uint count, uint tuple_size);
	void updateNormals(const float* colors);
	GLuint normalBufferId() const;


	void setColors(const float* colors, uint count, uint tuple_size);
	void updateColors(const float* colors);
	GLuint colorBufferId() const;

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

	QOpenGLBuffer colorBuf;
	uint colorCount;
	uint colorTupleSize;
	uint colorStride;

	QVector3D color;
};

#endif // _GL_MODEL_H_
