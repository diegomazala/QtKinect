
#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include "GLDepthBufferRenderer.h"
#include "QOpenGLTrackballWidget.h"

#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

class GLWidget : public QOpenGLTrackballWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit GLWidget(QWidget *parent = 0);
    ~GLWidget();

	GLDepthBufferRenderer& glDepthRenderer(){ return geometries; }

protected:

    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;

    void initShaders(); 
    void initTextures();

private:
    
    QOpenGLShaderProgram program;
	GLDepthBufferRenderer geometries;

    QOpenGLTexture *texture;

	QMatrix4x4 projection;
	
};

#endif // MAINWIDGET_H
