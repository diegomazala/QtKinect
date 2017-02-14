
#include "GLModelViewer.h"

#include <QMouseEvent>
#include <QTimer>
#include <QDir>
#include <math.h>
#include <iostream>
#include <time.h>


GLModelViewer::GLModelViewer(QWidget *parent) :
	QOpenGLTrackballWidget(parent)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);

}


GLModelViewer::~GLModelViewer()
{
	// Make sure the context is current when deleting the texture
	// and the buffers.
	makeCurrent();
	for (auto m : models)
		m->cleanupGL();
	doneCurrent();
}


void GLModelViewer::keyReleaseEvent(QKeyEvent *e)
{
	QOpenGLTrackballWidget::keyReleaseEvent(e);
}


void GLModelViewer::addModel(const std::shared_ptr<GLModel>& model)
{
	models.push_back(model);
}

std::shared_ptr<GLModel> GLModelViewer::getModel(int index)
{
	if (index > -1 && index < models.size())
		return models.at(index);
	else
		return nullptr;
}


void GLModelViewer::updateModel(int index, const float* vertices, const float* normals, size_t count, size_t tuple_size)
{
	if (index > -1 && models.size() > index)
	{
		std::shared_ptr<GLModel>& p = models.at(index);
		p->updateVertices(vertices);
		p->updateNormals(normals);
	}
}



void GLModelViewer::initializeGL()
{
	initializeOpenGLFunctions();

	glClearColor(0, 0, 0, 1);

	// Enable depth buffer
	glEnable(GL_DEPTH_TEST);

	// Enable back face culling
	glEnable(GL_CULL_FACE);
}



void GLModelViewer::renderModel(QOpenGLShaderProgram* program, GLModel* cloud)
{
	if (program == nullptr || !program->bind())
		return;

	// Calculate model view transformation
	QMatrix4x4 view;
	view.translate(position);
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



void GLModelViewer::paintGL()
{
	if (models.size() < 1)
		return;

	// Clear color and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Draw geometry
	for (auto m : models)
		renderModel(m->getShaderProgram().get(), m.get());

}
