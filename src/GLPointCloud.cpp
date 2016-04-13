
#include "GLPointCloud.h"
#include <iostream>
#include <QVector2D>
#include <QVector3D>

#include <Eigen/Dense>
#include "Projection.h"

GLPointCloud::GLPointCloud() : GLModel()
{
}

GLPointCloud::~GLPointCloud()
{
	cleanupGL();
}

GLuint GLPointCloud::vertexBufferId() const
{
	return vertexBuf.bufferId();
}

void GLPointCloud::initGL()
{
	initializeOpenGLFunctions();

	// Generate VBOs
	vertexBuf.create();
}


void GLPointCloud::cleanupGL()
{
	vertexBuf.destroy();
}

void GLPointCloud::setVertices(const float* vertices, uint count, uint tuple_size)
{
	vertexCount = count;
	tupleSize = tuple_size;
	stride = sizeof(float) * tuple_size;

	vertexBuf.bind();
	vertexBuf.allocate(vertices, static_cast<float>(count * stride));
}



void GLPointCloud::render(QOpenGLShaderProgram *program)
{
	if (!vertexBuf.isCreated())
		return;

    vertexBuf.bind();
	int vertexLocation = program->attributeLocation("in_position");
	program->enableAttributeArray(vertexLocation);
	program->setAttributeBuffer(vertexLocation, GL_FLOAT, 0, tupleSize, stride);

	program->setUniformValue("color", color);

    // Draw geometry 
	glDrawArrays(GL_POINTS, 0, static_cast<float>(vertexCount * tupleSize));

	vertexBuf.release();
}

