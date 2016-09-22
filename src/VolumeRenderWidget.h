
#ifndef _VOLUME_RENDER_WIDGET_H_
#define _VOLUME_RENDER_WIDGET_H_

#if 0

#include "QOpenGLTrackballWidget.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>
#include <QOpenGLDebugLogger>

class GLModel;
class GLQuad;


class VolumeRenderWidget : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit VolumeRenderWidget(QWidget *parent = 0);
    ~VolumeRenderWidget();

	void setup(const std::string& buffer_file_path, const size_t x, const size_t y, const size_t z);
	void initPixelBuffer();

protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
	void keyPressEvent(QKeyEvent* e);

	void cudaInit();
	void cudaRender();

private:
    QOpenGLShaderProgram program;
	

	QOpenGLBuffer* pixelBuf;
	QOpenGLTexture*  texture;
	GLQuad* quad;

    QMatrix4x4 projection;

	QOpenGLDebugLogger *logger;

	bool cudaInitialized;
};

#endif

#endif // _VOLUME_RENDER_WIDGET_H_
