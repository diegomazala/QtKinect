
#include "GLQuad.h"
#include <QOpenGLTexture>
#include <iostream>



GLQuad::GLQuad() : 
	GLModel()
	, vertexBuf(nullptr)
	, texCoordBuf(nullptr)
{
}


GLQuad::~GLQuad()
{
	cleanupGL();
}


GLuint GLQuad::vertexBufferId() const
{
	return vertexBuf->bufferId();
}


void GLQuad::initGL()
{
	initializeOpenGLFunctions();

	initShaders();


	float verts[] = {
		-1.0f, -1.0f, 0.f, 1.f,
		1.0f, -1.0f, 0.f, 1.f,
		1.0f, 1.0f, 0.f, 1.f,
		-1.0f, 1.0f, 0.f, 1.f };

	float uv[] = { 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f };
		

	vertexBuf = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	vertexBuf->setUsagePattern(QOpenGLBuffer::StaticDraw);
	vertexBuf->create();
	vertexBuf->bind();
	vertexBuf->allocate(verts, sizeof(verts));
	vertexBuf->release();

	texCoordBuf = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	texCoordBuf->setUsagePattern(QOpenGLBuffer::StaticDraw);
	texCoordBuf->create();
	texCoordBuf->bind();
	texCoordBuf->allocate(uv, sizeof(uv));
	texCoordBuf->release();


	std::vector<GLubyte> image;
	int w = 512,
		h = 512;
	for (GLuint i = 0; i < w; i++)
	{
		for (GLuint j = 0; j < h; j++)
		{
			GLuint c = ((((i & 0x8) == 0) ^ ((j & 0x8)) == 0)) * 255;
			image.push_back((GLubyte)c);
			image.push_back((GLubyte)c);
			image.push_back((GLubyte)c);
			image.push_back(255);
		}
	}

	pixelBuf = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
	pixelBuf->setUsagePattern(QOpenGLBuffer::StreamDraw);
	pixelBuf->create();
	pixelBuf->bind();
	pixelBuf->allocate(image.data(), w * h * sizeof(GLubyte) * 4);
	//pixelBuf->allocate(q_image.bits(), q_image.width() * q_image.height() * sizeof(GLubyte) * 4);
	pixelBuf->release();

	//pixelBuf->allocate(sizeof(image));
	//void *vid = pixelBuf->map(QOpenGLBuffer::WriteOnly);
	//memcpy(vid, data, sizeof(image));
	//pixelBuf->unmap();


	texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	texture->create();
	texture->bind();
	texture->setMinificationFilter(QOpenGLTexture::Nearest);
	texture->setMagnificationFilter(QOpenGLTexture::Nearest);
	texture->setWrapMode(QOpenGLTexture::ClampToEdge);
	texture->setSize(w, h);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	texture->setFormat(QOpenGLTexture::TextureFormat::RGBA8_UNorm);
	texture->allocateStorage(QOpenGLTexture::PixelFormat::RGBA,	QOpenGLTexture::PixelType::UInt8);
	//texture->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::UInt8, image.data());
	texture->release();
}


void GLQuad::cleanupGL()
{
	pixelBuf->destroy();
	vertexBuf->destroy();
	texCoordBuf->destroy();
}



void GLQuad::initShaders()
{
	const GLchar* vertexShader =
	{
		"#version 430\n"\
		"layout(location = 0) in vec4 in_Position;\n"\
		"layout(location = 1) in vec2 in_TexCoord;\n"\
		"out vec2 v_TexCoord;\n"\
		"void main(void)\n"\
		"{\n"\
		"  gl_Position = in_Position;\n"\
		"  v_TexCoord = in_TexCoord;\n"\
		"}\n"
	};


	const GLchar* fragmentShader =
	{
		"#version 430\n"\
		"in vec2 v_TexCoord;\n"\
		"out vec4 out_Color;\n"\
		"uniform sampler2D in_Texture;\n"\
		"void main(void)\n"\
		"{\n"\
		"  out_Color = texture(in_Texture, vec2(v_TexCoord.s, 1 - v_TexCoord.t));\n"\
		"}\n"
	};

	// Compile vertex shader
	if (!program.addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader))
		qDebug() << "Error: Could not load vertex shader";

	// Compile fragment shader
	if (!program.addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShader))
		qDebug() << "Error: Could not load fragment shader";


	// Link shader pipeline
	if (!program.link())
		qDebug() << "Error: Could not link GLQuad program";

	// Bind shader pipeline for use
	if (!program.bind())
		qDebug() << "Error: Could not bind GLQuad program";

	program.release();
}




void GLQuad::render()
{
	//texture->bind();

	//// copy from pbo to texture
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	//pixelBuf->bind();
	//{
	//	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 512, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	//	texture->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::UInt8, (void*)nullptr);
	//}
	//pixelBuf->release();

	program.bind();

	vertexBuf->bind();
	program.setAttributeBuffer(0, GL_FLOAT, 0, 4, 0);
	program.enableAttributeArray(0);

	texCoordBuf->bind();
	program.setAttributeBuffer(1, GL_FLOAT, 0, 2, 0);
	program.enableAttributeArray(1);

	

	
	// Draw geometry 
	glDrawArrays(GL_QUADS, 0, 4);

	program.release();

	vertexBuf->release();
	texCoordBuf->release();

	//texture->release();
}


void GLQuad::render(QOpenGLShaderProgram *program_ptr)
{
	render();
}

