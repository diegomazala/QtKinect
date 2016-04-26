
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
#include <memory>

class GeometryEngine;

class GLPointCloudViewer : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
	explicit GLPointCloudViewer(QWidget *parent = 0);
	~GLPointCloudViewer();

	void addPointCloud(const std::shared_ptr<GLModel>& point_cloud);

	void setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program);

	void updateCloud(const float* vertices, const float* normals, uint count, uint tuple_size);

protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
	void keyReleaseEvent(QKeyEvent *) Q_DECL_OVERRIDE;
    void initShaders();


private:
    
	std::shared_ptr<QOpenGLShaderProgram> shaderProgram;
	
	std::vector<std::shared_ptr<GLModel>> pointCloud;

    QMatrix4x4 projection;

	int currentCloud;
};

#endif // _GL_POINT_CLOUD_VIEWER_H_
