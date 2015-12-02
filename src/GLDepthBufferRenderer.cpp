
#include "GLDepthBufferRenderer.h"
#include "QKinectGrabber.h"
#include "QKinectPlayerCtrl.h"
#include <QVector2D>
#include <QVector3D>

#include <iostream>

struct VertexData
{
    QVector3D position;
    QVector2D texCoord;
};


GLDepthBufferRenderer::GLDepthBufferRenderer()
    : indexBuf(QOpenGLBuffer::IndexBuffer),
	depthBufferCloud(),
	kinectCtrl(nullptr)
{
    
}

GLDepthBufferRenderer::~GLDepthBufferRenderer()
{

}

void GLDepthBufferRenderer::initGL()
{
	initializeOpenGLFunctions();

	// Generate 2 VBOs
	arrayBuf.create();
	indexBuf.create();

	// Initializes cube geometry and transfers it to VBOs
	initCubeGeometry();
}

void GLDepthBufferRenderer::cleanupGL()
{
	arrayBuf.destroy();
	indexBuf.destroy();
}


void GLDepthBufferRenderer::setController(QKinectPlayerCtrl& kinect_ctrl)
{
	this->kinectCtrl = &kinect_ctrl;
}

void GLDepthBufferRenderer::setKinectReader(QKinectGrabber& kinect_reader)
{
	kinectReader = kinectReader;
}


void GLDepthBufferRenderer::updatePoints()
{
	if (this->kinectCtrl == nullptr)
		return;

	
	const int point_count = (int)kinectCtrl->mDepthBuffer.size();
	const int depth = kinectCtrl->mDepthBuffer.maxDistance() - kinectCtrl->mDepthBuffer.minDistance();

	if (point_count < 1)
		return;

	std::vector<float> vertices;
	int i = 0;
	for (int y = 0; y < kinectCtrl->mDepthBuffer.height(); ++y)
	{
		for (int x = 0; x < kinectCtrl->mDepthBuffer.width(); ++x)
		{
#if 0
			float d = static_cast<float>(kinectCtrl->mDepthBuffer.buffer[i++]) / float(depth);

			if (d < 0.1)
				continue;

			float vx = float(x - kinectCtrl->mDepthBuffer.width * 0.5f) / float(kinectCtrl->mDepthBuffer.width);
			float vy = float(y - kinectCtrl->mDepthBuffer.height * 0.5f) / float(kinectCtrl->mDepthBuffer.height);

			vertices.push_back(vx * 2.f);
			vertices.push_back(-vy * 2.f);
			vertices.push_back(-d);
#else

			const unsigned short d = (kinectCtrl->mDepthBuffer.buffer[i++]);
			
			if (d < kinectCtrl->mDepthBuffer.minDistance() || d > kinectCtrl->mDepthBuffer.maxDistance())
				continue;

			float vx = float(x - kinectCtrl->mDepthBuffer.width() * 0.5f);
			float vy = float(y - kinectCtrl->mDepthBuffer.height() * 0.5f);

			vertices.push_back(vx);
			vertices.push_back(-vy);
			vertices.push_back(-d);
#endif
		}
	}

	arrayBuf.bind();
	arrayBuf.allocate(vertices.data(), vertices.size() * sizeof(float));
}


void GLDepthBufferRenderer::setDepthBuffer(const std::vector<unsigned short>& depthBuffer, unsigned short width, unsigned short height)
{
	float vertexdata[] = { -1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, -1.0f };
}


void GLDepthBufferRenderer::initCubeGeometry()
{
	float vertexdata[] = { -1.0f, -1.0f,  1.0f,
							1.0f, -1.0f,  1.0f,
						   -1.0f,  1.0f,  1.0f,
							1.0f,  1.0f,  1.0f,
						   -1.0f, -1.0f, -1.0f,
							1.0f, -1.0f, -1.0f,
						   -1.0f,  1.0f, -1.0f,
							1.0f,  1.0f, -1.0f };

	// Transfer vertex data to VBO 0
	arrayBuf.bind();
	arrayBuf.allocate(vertexdata, 24 * sizeof(float));
}


void GLDepthBufferRenderer::render(QOpenGLShaderProgram *program)
{
	updatePoints();

	// Tell OpenGL programmable pipeline how to locate vertex position data
	int vertexLocation = program->attributeLocation("a_position");
	program->enableAttributeArray(vertexLocation);
		
	program->setAttributeBuffer(vertexLocation, GL_FLOAT, 0, 3, 0);
	const int point_count = arrayBuf.size() / sizeof(float) / 3;

	glDrawArrays(GL_POINTS, 0, point_count);
}

