
#include "GLPointCloudViewer.h"

#include <QMouseEvent>
#include <QTimer>
#include <QDir>
#include <math.h>
#include <iostream>



GLPointCloudViewer::GLPointCloudViewer(QWidget *parent) :
	QOpenGLTrackballWidget(parent)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);

	currentCloud = -1;
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


void GLPointCloudViewer::keyReleaseEvent(QKeyEvent *e)
{
	if (e->key() >= Qt::Key_0 && e->key() <= Qt::Key_9)
		currentCloud = e->key() - Qt::Key_0 - 1;

	if (e->key() == Qt::Key_0)
		currentCloud = -1;

	QOpenGLTrackballWidget::keyReleaseEvent(e);
}


void GLPointCloudViewer::addPointCloud(const std::shared_ptr<GLPointCloud>& point_cloud)
{
	pointCloud.push_back(point_cloud);
}

std::shared_ptr<GLPointCloud> GLPointCloudViewer::getCloud(int index)
{
	if (index > -1 && index < pointCloud.size())
		return pointCloud.at(index);
	else
		return nullptr;
}

void GLPointCloudViewer::updateCloud(const float* vertices, const float* normals, size_t count, size_t tuple_size)
{
	if (pointCloud.size() > 0)
	{
		std::shared_ptr<GLPointCloud>& p = pointCloud.at(0);
		p->updateVertices(vertices);
		p->updateNormals(normals);
	}
}



void GLPointCloudViewer::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0, 0, 0, 1);

    // Enable depth buffer
    glEnable(GL_DEPTH_TEST);

    // Enable back face culling
    glEnable(GL_CULL_FACE);
}



void GLPointCloudViewer::renderCloud(QOpenGLShaderProgram* program, GLPointCloud* cloud)
{
	if (program == nullptr || !program->bind())
		return;

	// Calculate model view transformation
	QMatrix4x4 view;
	view.translate(0, 0, distance);
	view.rotate(rotation);

	int projection_matrix_location = program->uniformLocation("projectionMatrix");
	if (projection_matrix_location > -1)
		program->setUniformValue("projectionMatrix", projection);
	else
		std::cerr << "Error: Shader does not have attribute 'projectionMatrix'" << std::endl;


	int view_matrix_location = program->uniformLocation("viewMatrix");
	if (view_matrix_location > -1)
		program->setUniformValue("viewMatrix", view);
	else
		std::cerr << "Error: Shader does not have attribute 'viewMatrix'" << std::endl;


	int model_matrix_location = program->uniformLocation("modelMatrix");
	if (model_matrix_location > -1)
		program->setUniformValue("modelMatrix", cloud->transform());
	else
		std::cerr << "Error: Shader does not have attribute 'modelMatrix'" << std::endl;

	cloud->render(program);
}



void GLPointCloudViewer::paintGL()
{
	if (pointCloud.size() < 1)
		return;

    // Clear color and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // Draw geometry
	if (currentCloud < 0 || currentCloud > pointCloud.size() - 1)
	{
		for (auto cloud : pointCloud)
			renderCloud(cloud->getShaderProgram().get(), cloud.get());
	}
	else
	{
		auto cloud = pointCloud[currentCloud];
		renderCloud(cloud->getShaderProgram().get(), cloud.get());
	}

	
}
