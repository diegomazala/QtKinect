
#ifndef _GL_MODEL_H_
#define _GL_MODEL_H_

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QMatrix4x4>

class GLModel : protected QOpenGLFunctions
{
public:
	GLModel() : QOpenGLFunctions()
	{
		transform.setToIdentity();
	};

	virtual ~GLModel(){};

	virtual void render(QOpenGLShaderProgram *program) = 0;

public slots:

	virtual void initGL() = 0;
	virtual void cleanupGL() = 0;

protected:
	QMatrix4x4 transform;
};

#endif // _GL_MODEL_H_
