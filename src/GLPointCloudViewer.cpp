
#include "GLPointCloudViewer.h"

#include <QMouseEvent>
#include <QTimer>
#include <QDir>
#include <math.h>
#include <iostream>
#include <time.h>

static QVector3D colours[7] = {
	QVector3D(1, 0, 0), QVector3D(0, 1, 0), QVector3D(0, 0, 1),
	QVector3D(1, 1, 0), QVector3D(1, 0, 1), QVector3D(0, 1, 1),
	QVector3D(1, 1, 1) };



GLPointCloudViewer::GLPointCloudViewer(QWidget *parent) :
	QOpenGLTrackballWidget(parent)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);

	srand(time(NULL));

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

    initShaders();

    // Enable depth buffer
    glEnable(GL_DEPTH_TEST);

    // Enable back face culling
    glEnable(GL_CULL_FACE);

	//glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	for (auto cloud : pointCloud)
		cloud->initGL();
}


void GLPointCloudViewer::initShaders()
{
	// look for shader dir 
	QDir dir;
	std::string shader_dir("resources/shaders/");
	for (int i = 0; i < 5; ++i)
	{
		if (!dir.exists(shader_dir.c_str()))
			shader_dir.insert(0, "../");
		else
			break;
	}

	QString vertexShaderFileName = shader_dir.c_str() + QString("color.vert");
	QString fragmentShaderFileName = shader_dir.c_str() + QString("color.frag");

	shaderProgram.reset(new QOpenGLShaderProgram);

    // Compile vertex shader
	if (!shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, vertexShaderFileName))
        close();

    // Compile fragment shader
	if (!shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, fragmentShaderFileName))
        close();

    // Link shader pipeline
	if (!shaderProgram->link())
        close();

    // Bind shader pipeline for use
	if (!shaderProgram->bind())
        close();
}



void GLPointCloudViewer::setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program)
{
	shaderProgram->release();
	shaderProgram = shader_program;
}





void GLPointCloudViewer::paintGL()
{
	if (!shaderProgram->bind())
		return;

    // Clear color and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Calculate model view transformation
    QMatrix4x4 view;
	view.translate(0, 0, distance);
	view.rotate(rotation);
	
	QMatrix4x4 model;
	model.rotate(180, 0, 1, 0);
	

	int projection_matrix_location = shaderProgram->uniformLocation("projectionMatrix");
	if (projection_matrix_location > -1)
		shaderProgram->setUniformValue("projectionMatrix", projection);
	else
		std::cerr << "Error: Shader does not have attribute 'projectionMatrix'" << std::endl;


	int view_matrix_location = shaderProgram->uniformLocation("viewMatrix");
	if (view_matrix_location > -1)
		shaderProgram->setUniformValue("viewMatrix", view);
	else
		std::cerr << "Error: Shader does not have attribute 'viewMatrix'" << std::endl;


	int model_matrix_location = shaderProgram->uniformLocation("modelMatrix");
	if (model_matrix_location > -1)
		shaderProgram->setUniformValue("modelMatrix", model);
	else
		std::cerr << "Error: Shader does not have attribute 'modelMatrix'" << std::endl;



    // Draw geometry
	if (currentCloud < 0 || currentCloud > pointCloud.size() - 1)
		for (auto cloud : pointCloud)
			cloud->render(shaderProgram.get());
	else
		pointCloud[currentCloud]->render(shaderProgram.get());
	
}
