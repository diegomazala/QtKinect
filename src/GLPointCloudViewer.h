
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

	void addPointCloud(const std::shared_ptr<GLPointCloud>& point_cloud);

	void setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program);

	void updateCloud(const float* vertices, const float* normals, size_t count, size_t tuple_size);

	std::shared_ptr<GLPointCloud> getCloud(int index = 0);


protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
	void keyReleaseEvent(QKeyEvent *) Q_DECL_OVERRIDE;
    void initShaders();


private:
    
	std::shared_ptr<QOpenGLShaderProgram> shaderProgram;
	
	std::vector<std::shared_ptr<GLPointCloud>> pointCloud;

	
	int currentCloud;
};

#endif // _GL_POINT_CLOUD_VIEWER_H_
