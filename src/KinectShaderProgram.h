
#ifndef __KINECT_SHADER_PROGRAM_H__
#define __KINECT_SHADER_PROGRAM_H__

#include <QOpenGLShaderProgram>
#include <QDir>
#include <QString>
#include <iostream>
class KinectShaderProgram : public QOpenGLShaderProgram
{
	Q_OBJECT
public:
	explicit KinectShaderProgram(QObject *parent = 0) :QOpenGLShaderProgram(parent){};
	virtual ~KinectShaderProgram(){};

	bool build(const QString& vertex_shader_file, const QString& fragment_shader_file)
	{
		// look for shader directory
		QString shader_dir = findShadersDir("resources/shaders/");
		// try another folder
		if (shader_dir.isEmpty())	
			shader_dir = findShadersDir("shaders/");

		QString vertexShaderFileName = shader_dir + vertex_shader_file;
		QString fragmentShaderFileName = shader_dir + fragment_shader_file;

		if (!QFile(vertexShaderFileName).exists() ||
			!QFile(fragmentShaderFileName).exists())
			return false;

		// Compile vertex shader
		if (!addShaderFromSourceFile(QOpenGLShader::Vertex, vertexShaderFileName))
			return false;

		// Compile fragment shader
		if (!addShaderFromSourceFile(QOpenGLShader::Fragment, fragmentShaderFileName))
			return false;

		// Link shader pipeline
		if (!link())
			return false;

		return true;
	}


	//
	// return shader directory if found
	// 
	QString findShadersDir(const QString& folder)
	{
		const int max_iterations = 5;
		// look for shader dir 
		QDir dir;

		std::string shader_dir("resources/shaders/");
		for (int i = 0; i < max_iterations; ++i)
		{
			if (!dir.exists(shader_dir.c_str()))
				shader_dir.insert(0, "../");
			else
				return shader_dir.c_str();
		}
		return QString();
	}

};

#endif // __KINECT_SHADER_PROGRAM_H__