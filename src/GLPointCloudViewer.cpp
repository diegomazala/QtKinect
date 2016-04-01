
#include "GLPointCloudViewer.h"

#include <QMouseEvent>
#include <QTimer>
#include <math.h>
#include <iostream>
#include <time.h>

static QVector3D colours[10] = {
	QVector3D(1, 0, 0), QVector3D(0, 1, 0), QVector3D(0, 0, 1),
	QVector3D(1, 1, 0), QVector3D(1, 0, 1), QVector3D(0, 1, 1),
	QVector3D(1, 1, 1), QVector3D(0, 0.5, 0.25), QVector3D(0.75, 0.5, 0),
	QVector3D(0.25, 0.5, 0.75) };



GLPointCloudViewer::GLPointCloudViewer(QWidget *parent) :
	QOpenGLTrackballWidget(parent)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);
}


GLPointCloudViewer::~GLPointCloudViewer()
{
    // Make sure the context is current when deleting the texture
    // and the buffers.
    makeCurrent();
	for (auto cloud : pointCloud)
		cloud->cleanupGL();
    doneCurrent();
}


void GLPointCloudViewer::addPointCloud(PointCloudXYZW* point_cloud)
{
	if (point_cloud == nullptr)
		return;

	pointCloud.push_back(new GLPointCloud());
	GLPointCloud* cloud = pointCloud.back();
	cloud->initGL();
	cloud->setVertices(point_cloud->data()->data(), point_cloud->size(), 4);
}


void GLPointCloudViewer::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0, 0, 0, 1);

    initShaders();

    // Enable depth buffer
    glEnable(GL_DEPTH_TEST);

    // Enable back face culling
    glEnable(GL_CULL_FACE);

	for (auto cloud : pointCloud)
		cloud->initGL();
	
}


void GLPointCloudViewer::initShaders()
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
	
	srand(time(NULL));
	program.setUniformValue("color", colours[rand() % 10]);
}





void GLPointCloudViewer::resizeGL(int w, int h)
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



void GLPointCloudViewer::paintGL()
{
    // Clear color and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Calculate model view transformation
    QMatrix4x4 matrix;
	matrix.translate(0, 0, distance);
    matrix.rotate(rotation);
	
	QMatrix4x4 model;
	model.rotate(180, 0, 1, 0);

    // Set modelview-projection matrix
    program.setUniformValue("mvp_matrix", projection * matrix * model);

    // Draw cube geometry
	for (auto cloud : pointCloud)
		cloud->render(&program);
}
