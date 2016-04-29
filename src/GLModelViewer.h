
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

class GLModel;


class GLModelViewer : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit GLModelViewer(QWidget *parent = 0);
    ~GLModelViewer();

	void setModel(GLModel* gl_model);

protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;

    void initShaders();

private:
    QOpenGLShaderProgram program;
    GLModel* model;

    QMatrix4x4 projection;
};

#endif // _GL_MODEL_VIEWER_H_
