
#ifndef _GL_KINECT_WIDGET_H_
#define _GL_KINECT_WIDGET_H_

#include "GLKinectFrame.h"
#include "QOpenGLTrackballWidget.h"
#include "GLKinectFrame.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

class GeometryEngine;

class GLKinectWidget : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit GLKinectWidget(QWidget *parent = 0);
    ~GLKinectWidget();

	void setFrame(KinectFrame* kinect_frame);

protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;

    void initShaders();

private:
    QOpenGLShaderProgram program;
    GLKinectFrame frame;

    QMatrix4x4 projection;
};

#endif // _GL_KINECT_WIDGET_H_
