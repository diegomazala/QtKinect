
#include "GLModel.h"
#include <iostream>
#include <QVector2D>
#include <QVector3D>

#include <Eigen/Dense>
#include "Projection.h"

GLModel::GLModel() : shaderProgram(nullptr), color(1, 1, 1)
{
	transformMatrix.setToIdentity();
}

GLModel::~GLModel()
{
	cleanupGL();
}

GLuint GLModel::vertexBufferId() const
{
	return vertexBuf.bufferId();
}

GLuint GLModel::normalBufferId() const
{
	return normalBuf.bufferId();
}

GLuint GLModel::colorBufferId() const
{
	return colorBuf.bufferId();
}

void GLModel::initGL()
{
	initializeOpenGLFunctions();

	// Generate VBOs
	vertexBuf.create();
	vertexBuf.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);

	// Generate VBOs
	normalBuf.create();
	normalBuf.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);

	colorBuf.create();
	colorBuf.setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
}


void GLModel::cleanupGL()
{
	vertexBuf.destroy();
	normalBuf.destroy();
	colorBuf.destroy();
}


QMatrix4x4& GLModel::transform()
{
	return transformMatrix;
}

void GLModel::setShaderProgram(const std::shared_ptr<QOpenGLShaderProgram>& shader_program)
{
	shaderProgram = shader_program;
}

std::shared_ptr<QOpenGLShaderProgram> GLModel::getShaderProgram()
{
	return shaderProgram;
}


void GLModel::setVertices(const float* vertices, uint count, uint tuple_size)
{
	vertexCount = count;
	vertexTupleSize = tuple_size;
	vertexStride = sizeof(float) * tuple_size;

	vertexBuf.bind();
	vertexBuf.allocate(vertices, static_cast<float>(vertexCount * vertexStride));
}


void GLModel::updateVertices(const float* vertices)
{
	if (!vertices)
		return;

	vertexBuf.bind();
	vertexBuf.write(0, vertices, static_cast<float>(vertexCount * vertexStride));
}


void GLModel::setColors(const float* colors, uint count, uint tuple_size)
{
	colorCount = count;
	colorTupleSize = tuple_size;
	colorStride = sizeof(float) * tuple_size;

	colorBuf.bind();
	colorBuf.allocate(colors, static_cast<float>(colorCount * colorStride));
}


void GLModel::updateColors(const float* colors)
{
	if (!colors)
		return;

	colorBuf.bind();
	colorBuf.allocate(colors, static_cast<float>(colorCount * colorStride));
}

void GLModel::setNormals(const float* normals, uint count, uint tuple_size)
{
	normalCount = count;
	normalTupleSize = tuple_size;
	normalStride = sizeof(float) * tuple_size;

	normalBuf.bind();
	normalBuf.allocate(normals, static_cast<float>(normalCount * normalStride));
}


void GLModel::updateNormals(const float* normals)
{
	if (!normals)
		return;

	normalBuf.bind();
	normalBuf.allocate(normals, static_cast<float>(normalCount * normalStride));
}


void GLModel::render(QOpenGLShaderProgram *program)
{
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

	if (colorBuf.bind())
	{
		shaderProgram->setAttributeBuffer("in_color", GL_FLOAT, 0, colorTupleSize, colorStride);
		shaderProgram->enableAttributeArray("in_color");
	}

	// Draw geometry 
	glDrawArrays(GL_POINTS, 0, static_cast<float>(vertexCount * vertexTupleSize));

	vertexBuf.release();
	normalBuf.release();
}

