
#ifndef _GL_RAYCAST_TEXTURE_WIDGET_H_
#define _GL_RAYCAST_TEXTURE_WIDGET_H_

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


class GLRaycastTextureWidget : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit GLRaycastTextureWidget(QWidget *parent = 0);
    ~GLRaycastTextureWidget();

	void setModel(GLModel* gl_model);

	void setup(const std::string& filepath, ushort vx_count, ushort vx_size);
	void cleanup();
	void computeRaycast();

protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;

    void initShaders();

private:
    QOpenGLShaderProgram program;
    GLModel* model;

    QMatrix4x4 projection;


	QImage raycastImage;
	float cam_z_coord;
	float fov;
	ushort voxel_count;
	ushort voxel_size;
};

#endif // _GL_RAYCAST_TEXTURE_WIDGET_H_
