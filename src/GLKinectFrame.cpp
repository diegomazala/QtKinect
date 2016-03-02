
#include "GLKinectFrame.h"
#include "KinectFrame.h"
#include <iostream>
#include <QVector2D>
#include <QVector3D>

#include <Eigen/Dense>
#include "Projection.h"
#include "KinectFrame.h"



GLKinectFrame::GLKinectFrame()
{
}


GLKinectFrame::~GLKinectFrame()
{
}



void GLKinectFrame::initGL()
{
	initializeOpenGLFunctions();

	// Generate VBOs
	vertexBuf.create();
	colorBuf.create();
	normalBuf.create();

	//KinectFrame frame;
	//QKinectIO::loadFrame("../../data/room.knt", frame);
	//setFrame(&frame);
}


void GLKinectFrame::cleanupGL()
{
	vertexBuf.destroy();
	colorBuf.destroy();
	normalBuf.destroy();
}



void GLKinectFrame::setFrame(KinectFrame* frame)
{
	try
	{
		const int color_map_width = 1920;
		const int color_map_height = 1080;
		const int depth_map_width = 512;
		const int depth_map_height = 424;
		const float fovy = 70.0f;
		const float aspect_ratio = static_cast<float>(depth_map_width) / static_cast<float>(depth_map_height);
		const float near_plane = 0.1f;
		const float far_plane = 10240.0f;

		std::vector<Eigen::Vector3f> vertices;
		std::vector<Eigen::Vector3f> normals;
		std::vector<Eigen::Vector3f> colors;

		const float depth_to_color_width = color_map_width / depth_map_width;
		const float depth_to_color_height = color_map_height / depth_map_height;

		for (int x = 1; x < depth_map_width - 1; ++x)
		{
			for (int y = 1; y < depth_map_height - 1; ++y)
			{
				const float depth = static_cast<float>(frame->depth[y * depth_map_width + x]) / 100.f;
				const Eigen::Vector3f vert_uv = window_coord_to_3d(Eigen::Vector2f(x, y), depth, fovy, aspect_ratio, near_plane, far_plane, depth_map_width, depth_map_height);
				const Eigen::Vector3f vert_u1v = window_coord_to_3d(Eigen::Vector2f(x + 1, y), depth, fovy, aspect_ratio, near_plane, far_plane, depth_map_width, depth_map_height);
				const Eigen::Vector3f vert_uv1 = window_coord_to_3d(Eigen::Vector2f(x, y + 1), depth, fovy, aspect_ratio, near_plane, far_plane, depth_map_width, depth_map_height);

				float x_color = x * depth_to_color_width;
				float y_color = y * depth_to_color_height;

				if (!vert_uv.isZero() && !vert_u1v.isZero() && !vert_uv1.isZero())
				{
					const Eigen::Vector3f n1 = vert_u1v - vert_uv;
					const Eigen::Vector3f n2 = vert_uv1 - vert_uv;
					const Eigen::Vector3f n = n1.cross(n2).normalized();

					vertices.push_back(vert_uv);
					normals.push_back(n);

					const uchar r = static_cast<uchar>(frame->color[4 * y_color * color_map_width + x_color + 0]);
					const uchar g = static_cast<uchar>(frame->color[4 * y_color * color_map_width + x_color + 1]);
					const uchar b = static_cast<uchar>(frame->color[4 * y_color * color_map_width + x_color + 2]);

					//colors.push_back((n * 0.5f + Eigen::Vector3f(0.5, 0.5, 0.5)) * 255.0f);
					//colors.push_back(Eigen::Vector3f(0, 1, 0));

					colors.push_back(Eigen::Vector3f(static_cast<float>(r) / 255.f, static_cast<float>(g) / 255.f, static_cast<float>(b) / 255.f));
				}
			}
		}

		vertexBuf.bind();
		vertexBuf.allocate(&vertices[0][0], vertices.size() * sizeof(Eigen::Vector3f));

		colorBuf.bind();
		colorBuf.allocate(&colors[0][0], colors.size() * sizeof(Eigen::Vector3f));

		normalBuf.bind();
		normalBuf.allocate(&normals[0][0], normals.size() * sizeof(Eigen::Vector3f));
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << std::endl;
	}
}




void GLKinectFrame::render(QOpenGLShaderProgram *program)
{
    vertexBuf.bind();
    int vertexLocation = program->attributeLocation("in_position");
    program->enableAttributeArray(vertexLocation);
	program->setAttributeBuffer(vertexLocation, GL_FLOAT, 0, 3, sizeof(Eigen::Vector3f));

	colorBuf.bind();
	int colorLocation = program->attributeLocation("in_color");
	program->enableAttributeArray(colorLocation);
	program->setAttributeBuffer(colorLocation, GL_FLOAT, 0, 3, sizeof(Eigen::Vector3f));

	normalBuf.bind();
	int normalLocation = program->attributeLocation("in_normal");
	program->enableAttributeArray(normalLocation);
	program->setAttributeBuffer(normalLocation, GL_FLOAT, 0, 3, sizeof(Eigen::Vector3f));
 
    // Draw geometry 
	glDrawArrays(GL_POINTS, 0, vertexBuf.size() / sizeof(Eigen::Vector3f));
}

