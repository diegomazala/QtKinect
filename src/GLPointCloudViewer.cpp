
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

void GLPointCloudViewer::addPointCloud(const std::shared_ptr<GLModel>& point_cloud)
{
	pointCloud.push_back(point_cloud);
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


	int colorLocation = shaderProgram->uniformLocation("color");
	if (colorLocation > -1)
		shaderProgram->setUniformValue(colorLocation, colours[rand() % 7]);
}

void GLPointCloudViewer::setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program)
{
	shaderProgram->release();
	shaderProgram = shader_program;
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
	for (auto cloud : pointCloud)
		cloud->render(shaderProgram.get());
	
}
