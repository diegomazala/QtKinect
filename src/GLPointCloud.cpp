
#include "GLPointCloud.h"
#include <iostream>
#include <QVector2D>
#include <QVector3D>

#include <Eigen/Dense>
#include "Projection.h"

GLPointCloud::GLPointCloud() : GLModel(), color(1, 1, 1)
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

GLuint GLPointCloud::normalBufferId() const
{
	return normalBuf.bufferId();
}

void GLPointCloud::initGL()
{
	initializeOpenGLFunctions();

	// Generate VBOs
	vertexBuf.create();
	vertexBuf.setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);

	// Generate VBOs
	normalBuf.create();
	normalBuf.setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
}


void GLPointCloud::cleanupGL()
{
	vertexBuf.destroy();
	normalBuf.destroy();
}

void GLPointCloud::setVertices(const float* vertices, uint count, uint tuple_size)
{
	vertexCount = count;
	vertexTupleSize = tuple_size;
	vertexStride = sizeof(float) * tuple_size;

	vertexBuf.bind();
	vertexBuf.allocate(vertices, static_cast<float>(count * vertexStride));
}


void GLPointCloud::setNormals(const float* normals, uint count, uint tuple_size)
{
	normalCount = count;
	normalTupleSize = tuple_size;
	normalStride = sizeof(float) * tuple_size;

	normalBuf.bind();
	normalBuf.allocate(normals, static_cast<float>(count * normalStride));
}



void GLPointCloud::render(QOpenGLShaderProgram *program)
{
	if (!vertexBuf.isCreated())
		return;

    vertexBuf.bind();
	//int vertexLocation = program->attributeLocation("in_position");
	program->setAttributeBuffer("in_position", GL_FLOAT, 0, vertexTupleSize, vertexStride);
	program->enableAttributeArray("in_position");
	
	normalBuf.bind();
	program->setAttributeBuffer("in_normal", GL_FLOAT, 0, normalTupleSize, normalStride);
	program->enableAttributeArray("in_normal");


	program->setUniformValue("color", color);

    // Draw geometry 
	glDrawArrays(GL_POINTS, 0, static_cast<float>(vertexCount * vertexTupleSize));

	vertexBuf.release();
}

