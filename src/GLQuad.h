
#ifndef _GL_QUAD_H_
#define _GL_QUAD_H_

#include "GLModel.h"

class QOpenGLTexture;

class GLQuad : public GLModel
{


public:
	GLQuad();
	virtual ~GLQuad();

	void render();
	void render(QOpenGLShaderProgram *program_ptr);

	GLuint vertexBufferId() const;

public slots:

	void initGL();
	void cleanupGL();

private:

	void initShaders();

	QOpenGLShaderProgram program;
	QOpenGLBuffer* vertexBuf;
	QOpenGLBuffer* texCoordBuf;
	QOpenGLBuffer* pixelBuf;
	QOpenGLTexture*  texture;

};

#endif // _GL_QUAD_H_
