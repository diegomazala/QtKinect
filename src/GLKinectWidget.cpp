
#include "GLKinectWidget.h"

#include <QMouseEvent>
#include <QTimer>
#include <math.h>

GLKinectWidget::GLKinectWidget(QWidget *parent) :
	QOpenGLTrackballWidget(parent)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);
}


GLKinectWidget::~GLKinectWidget()
{
    // Make sure the context is current when deleting the texture
    // and the buffers.
    makeCurrent();
	frame.cleanupGL();
    doneCurrent();
}


void GLKinectWidget::setFrame(KinectFrame* kinect_frame)
{
	frame.setFrame(kinect_frame);
}


void GLKinectWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0, 0, 0, 1);

    initShaders();

    // Enable depth buffer
    glEnable(GL_DEPTH_TEST);

    // Enable back face culling
    glEnable(GL_CULL_FACE);

	frame.initGL();
}


void GLKinectWidget::initShaders()
{
    // Compile vertex shader
	if (!program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/resources/shaders/kinect.vert"))
        close();

    // Compile fragment shader
	if (!program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/resources/shaders/kinect.frag"))
        close();

    // Link shader pipeline
    if (!program.link())
        close();

    // Bind shader pipeline for use
    if (!program.bind())
        close();
}





void GLKinectWidget::resizeGL(int w, int h)
{
	const float depth_map_width = 512;
	const float depth_map_height = 424;
	const float fovy = 70.0f;
	const float aspect_ratio = depth_map_width / depth_map_height;
	const float near_plane = 0.1f;
	const float far_plane = 10240.0f;

    // Reset projection
    projection.setToIdentity();

    // Set perspective projection
	projection.perspective(fovy, aspect_ratio, near_plane, far_plane);
}



void GLKinectWidget::paintGL()
{
    // Clear color and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Calculate model view transformation
    QMatrix4x4 matrix;
	matrix.translate(position);
    matrix.rotate(rotation);
	
	QMatrix4x4 model;
	model.rotate(180, 0, 1, 0);

    // Set modelview-projection matrix
    program.setUniformValue("mvp_matrix", projection * matrix * model);

    // Draw cube geometry
    frame.render(&program);
}
