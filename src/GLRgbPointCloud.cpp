
#include "GLRgbPointCloud.h"
#include <iostream>
#include <QVector2D>
#include <QVector3D>

#include <Eigen/Dense>
#include "Projection.h"

GLRgbPointCloud::GLRgbPointCloud() : GLModel()
{
}

GLRgbPointCloud::~GLRgbPointCloud()
{
	cleanupGL();
}

GLuint GLRgbPointCloud::vertexBufferId() const
{
	return vertexBuf.bufferId();
}

void GLRgbPointCloud::initGL()
{
	initializeOpenGLFunctions();

	// Generate VBOs
	vertexBuf.create();
	colorBuf.create();
}


void GLRgbPointCloud::cleanupGL()
{
	vertexBuf.destroy();
	colorBuf.destroy();
}

void GLRgbPointCloud::setVertices(const float* vertices, uint count, uint tuple_size)
{
	vertexCount = count;
	vertexTupleSize = tuple_size;
	vertexStride = sizeof(float) * tuple_size;

	vertexBuf.bind();
	vertexBuf.allocate(vertices, static_cast<float>(vertexCount * vertexStride));
	vertexBuf.release();
}


void GLRgbPointCloud::setColors(const float* colors, uint count, uint tuple_size)
{
	colorCount = count;
	colorTupleSize = tuple_size;
	colorStride = sizeof(float) * tuple_size;

	colorBuf.bind();
	colorBuf.allocate(colors, static_cast<float>(colorCount * colorStride));
	colorBuf.release();
}



void GLRgbPointCloud::render(QOpenGLShaderProgram *program)
{
	if (!vertexBuf.isCreated())
		return;

    
	int vertexLocation = program->attributeLocation("in_position");
	if (vertexLocation > -1 && vertexBuf.bind())
	{
		program->enableAttributeArray(vertexLocation);
		program->setAttributeBuffer(vertexLocation, GL_FLOAT, 0, vertexTupleSize, vertexStride);
	}
	
	
	int colorLocation = program->attributeLocation("in_color");
	if (colorLocation > -1 && colorBuf.bind())
	{
		program->enableAttributeArray(colorLocation);
		program->setAttributeBuffer(colorLocation, GL_FLOAT, 0, colorTupleSize, colorStride);
	}


    // Draw geometry 
	glDrawArrays(GL_POINTS, 0, static_cast<int>(vertexCount * vertexTupleSize));

	vertexBuf.release();
}

