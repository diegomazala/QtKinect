
#ifndef _GL_MODEL_VIEWER_H_
#define _GL_MODEL_VIEWER_H_

#include "QOpenGLTrackballWidget.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <memory>
#include "GLModel.h"


class GLModelViewer : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
	Q_OBJECT

public:
	explicit GLModelViewer(QWidget *parent = 0);
	~GLModelViewer();

	void addModel(const std::shared_ptr<GLModel>& model);
	void updateModel(int index, const float* vertices, const float* normals, size_t count, size_t tuple_size);
	std::shared_ptr<GLModel> getModel(int index = 0);


protected:

	void initializeGL() Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void keyReleaseEvent(QKeyEvent *) Q_DECL_OVERRIDE;
	void renderModel(QOpenGLShaderProgram* program, GLModel* cloud);

private:

	std::vector<std::shared_ptr<GLModel>> models;
};

#endif // _GL_MODEL_VIEWER_H_
