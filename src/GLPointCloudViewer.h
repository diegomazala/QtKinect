
#ifndef _GL_POINT_CLOUD_VIEWER_H_
#define _GL_POINT_CLOUD_VIEWER_H_

#include "QOpenGLTrackballWidget.h"
#include "PointCloud.h"
#include "GLPointCloud.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

class GeometryEngine;

class GLPointCloudViewer : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
	explicit GLPointCloudViewer(QWidget *parent = 0);
	~GLPointCloudViewer();

	void addPointCloud(PointCloudXYZW* point_cloud);

protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;

    void initShaders();

private:
    QOpenGLShaderProgram program;
	std::vector<GLPointCloud*> pointCloud;

    QMatrix4x4 projection;
};

#endif // _GL_POINT_CLOUD_VIEWER_H_
