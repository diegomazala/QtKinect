

#include "GLWidget.h"
#include <QMouseEvent>
#include <QTimer>



GLWidget::GLWidget(QWidget *parent) :
	QOpenGLTrackballWidget(parent),
    texture(0)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);

}

GLWidget::~GLWidget()
{
    // Make sure the context is current when deleting the texture
    // and the buffers.
    makeCurrent();
    delete texture;
	geometries.cleanupGL();
    doneCurrent();
}




void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0, 0, 0, 1);

    initShaders();
    initTextures();

    // Enable depth buffer
    glEnable(GL_DEPTH_TEST);

    // Enable back face culling
    glEnable(GL_CULL_FACE);

	geometries.initGL();
}


void GLWidget::initShaders()
{
    // Compile vertex shader
	if (!program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/resources/shaders/color.vert"))
        close();

    // Compile fragment shader
	if (!program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/resources/shaders/color.frag"))
        close();

    // Link shader pipeline
    if (!program.link())
        close();

    // Bind shader pipeline for use
    if (!program.bind())
        close();
}



void GLWidget::initTextures()
{
    // Load cube.png image
    texture = new QOpenGLTexture(QImage(":/resources/images/cube.png").mirrored());

    // Set nearest filtering mode for texture minification
    texture->setMinificationFilter(QOpenGLTexture::Nearest);

    // Set bilinear filtering mode for texture magnification
    texture->setMagnificationFilter(QOpenGLTexture::Linear);

    // Wrap texture coordinates by repeating
    // f.ex. texture coordinate (1.1, 1.2) is same as (0.1, 0.2)
    texture->setWrapMode(QOpenGLTexture::Repeat);
}



void GLWidget::resizeGL(int w, int h)
{
	float aspect = float(w) / float(h ? h : 1);
	const float zNear = 0.1f, zFar = 10240.0f, fov = 60.0f;

    // Reset projection
    projection.setToIdentity();

    // Set perspective projection
    projection.perspective(fov, aspect, zNear, zFar);
}


void GLWidget::paintGL()
{
    // Clear color and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    texture->bind();


    // Calculate model view transformation
    QMatrix4x4 matrix;
	matrix.translate(0.0, 0.0, -distance);
    matrix.rotate(rotation);

    // Set modelview-projection matrix
    program.setUniformValue("mvp_matrix", projection * matrix);


    // Use texture unit 0 which contains cube.png
    program.setUniformValue("texture", 0);

    // Draw cube geometry
    geometries.render(&program);
}
