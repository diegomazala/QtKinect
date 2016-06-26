
#include "GLPointCloud.h"
#include <iostream>
#include <QVector2D>
#include <QVector3D>

#include <Eigen/Dense>
#include "Projection.h"

GLPointCloud::GLPointCloud() : shaderProgram(nullptr), color(1, 1, 1)
{
	transformMatrix.setToIdentity();
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
	vertexBuf.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);

	// Generate VBOs
	normalBuf.create();
	normalBuf.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
}


void GLPointCloud::cleanupGL()
{
	vertexBuf.destroy();
	normalBuf.destroy();
}


QMatrix4x4& GLPointCloud::transform()
{
	return transformMatrix;
}

void GLPointCloud::setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program)
{
	shaderProgram = shader_program;
}

std::shared_ptr<QOpenGLShaderProgram> GLPointCloud::getShaderProgram()
{
	return shaderProgram;
}


void GLPointCloud::setVertices(const float* vertices, uint count, uint tuple_size)
{
	vertexCount = count;
	vertexTupleSize = tuple_size;
	vertexStride = sizeof(float) * tuple_size;

	vertexBuf.bind();
	vertexBuf.allocate(vertices, static_cast<float>(vertexCount * vertexStride));
	vertexBuf.release();
}


void GLPointCloud::updateVertices(const float* vertices)
{
	if (!vertices)
		return;

	vertexBuf.bind();
	vertexBuf.write(0, vertices, static_cast<float>(vertexCount * vertexStride));
	vertexBuf.release();
}


void GLPointCloud::setNormals(const float* normals, uint count, uint tuple_size)
{
	normalCount = count;
	normalTupleSize = tuple_size;
	normalStride = sizeof(float) * tuple_size;

	normalBuf.bind();
	normalBuf.allocate(normals, static_cast<float>(normalCount * normalStride));
	normalBuf.release();
}


void GLPointCloud::updateNormals(const float* normals)
{
	if (!normals)
		return;

	normalBuf.bind();
	normalBuf.allocate(normals, static_cast<float>(normalCount * normalStride));
	normalBuf.release();
}


void GLPointCloud::render(QOpenGLShaderProgram *program)
{
	program->bind();

	if (!vertexBuf.isCreated())
		return;


	if (vertexBuf.bind())
	{
		shaderProgram->setAttributeBuffer("in_position", GL_FLOAT, 0, vertexTupleSize, vertexStride);
		shaderProgram->enableAttributeArray("in_position");
	}
	
	if (normalBuf.bind())
	{
		shaderProgram->setAttributeBuffer("in_normal", GL_FLOAT, 0, normalTupleSize, normalStride);
		shaderProgram->enableAttributeArray("in_normal");
	}


	shaderProgram->setUniformValue("color", color);

    // Draw geometry 
	glDrawArrays(GL_POINTS, 0, static_cast<float>(vertexCount * vertexTupleSize));


	vertexBuf.release();
	normalBuf.release();

	program->release();
}

