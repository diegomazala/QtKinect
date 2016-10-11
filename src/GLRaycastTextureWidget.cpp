
#include "GLRaycastTextureWidget.h"
#include "GLModel.h"
#include "GLQuad.h"
#include <QMouseEvent>
#include <QTimer>
#include <math.h>
#include "Timer.h"
#include "Eigen/Dense"
#include "Projection.h"
#include "KinectFrame.h"
#include "KinectSpecs.h"
#include "Volumetric_helper.h"
#include "KinectFusionKernels/KinectFusionKernels.h"

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <helper_cuda_gl.h>

static Timer ttimer;
struct cudaGraphicsResource *cuda_pixel_buffer; // CUDA Graphics Resource (to transfer PBO)

static ushort g_width = KINECT_V2_DEPTH_WIDTH;
static ushort g_height = KINECT_V2_DEPTH_HEIGHT;

GLRaycastTextureWidget::GLRaycastTextureWidget(QWidget *parent) :
	QOpenGLTrackballWidget(parent)
	, pixelBuf(nullptr)
	, texture(nullptr)
	, raycastImage(g_width, g_height, QImage::Format::Format_RGBA8888)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);

	weelSpeed = 0.01f;
	position.setZ(-32);
}


GLRaycastTextureWidget::~GLRaycastTextureWidget()
{
	cleanup();

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}


void GLRaycastTextureWidget::initializeGL()
{
    initializeOpenGLFunctions();

	std::stringstream info;
	info << "OpenGl information: VENDOR:       " << (const char*)glGetString(GL_VENDOR) << std::endl
		<< "                    RENDERER:     " << (const char*)glGetString(GL_RENDERER) << std::endl
		<< "                    VERSION:      " << (const char*)glGetString(GL_VERSION) << std::endl
		<< "                    GLSL VERSION: " << (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	std::cout << std::endl << info.str() << std::endl << std::endl;

    glClearColor(0, 0, 0, 1);

    initShaders();

    // Enable depth buffer
    glEnable(GL_DEPTH_TEST);

    // Enable back face culling
    glEnable(GL_CULL_FACE);

	quad = new GLQuad();
	quad->initGL();

	initPixelBuffer();
}


void GLRaycastTextureWidget::initShaders()
{
    // Compile vertex shader
	if (!program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/resources/shaders/vertices.vert"))
        close();

    // Compile fragment shader
	if (!program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/resources/shaders/vertices.frag"))
        close();

    // Link shader pipeline
    if (!program.link())
        close();

    // Bind shader pipeline for use
    if (!program.bind())
        close();
}





void GLRaycastTextureWidget::resizeGL(int w, int h)
{
	const float fovy = 45.0f;
	const float aspect_ratio = (float)w / (float) h;
	const float near_plane = 0.1f;
	const float far_plane = 10240.0f;

	glViewport(0, 0, w, h);

    // Reset projection
    projection.setToIdentity();

    // Set perspective projection
	projection.perspective(fovy, aspect_ratio, near_plane, far_plane);
}



void GLRaycastTextureWidget::paintGL()
{
	if (!texture)
		return;

	cudaRender();

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	
	texture->bind();

	// copy from pbo to texture
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	pixelBuf->bind();
	texture->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::UInt8, (void*)nullptr);
	pixelBuf->release();


	// Clear color and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	quad->render();


	texture->release();
}





void GLRaycastTextureWidget::cudaRender()
{
	int vol_size = voxel_count * voxel_size;

	if (!pixelBuf || !pixelBuf->isCreated())
		return;

	// map PBO to get CUDA device pointer
	uint *image_dev_ptr;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pixel_buffer, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&image_dev_ptr, &num_bytes, cuda_pixel_buffer));
	
	// clear image
	//checkCudaErrors(cudaMemset(image_dev_ptr, 0, g_width * g_height * 4));

	QMatrix4x4 view_matrix;
	
	view_matrix.translate(position);
	view_matrix.rotate(rotation);

	QMatrix4x4 view_matrix_inv = view_matrix.transposed().inverted();

#if 0
	// call cuda kernel
	raycast_box(
		image_dev_ptr, 
		g_width, //KINECT_V2_DEPTH_WIDTH, 
		g_height, //KINECT_V2_DEPTH_HEIGHT, 
		fov, //KINECT_V2_FOVY, 
		(float)g_width / (float)g_height, //KINECT_V2_DEPTH_ASPECT_RATIO, 
		view_matrix_inv.data(),
		make_ushort3(vol_size, vol_size, vol_size));
#else

	//ttimer.start();
	knt_cuda_raycast_gl(
		image_dev_ptr,
		g_width, //KINECT_V2_DEPTH_WIDTH, 
		g_height, //KINECT_V2_DEPTH_HEIGHT, 
		fov, //KINECT_V2_FOVY, 
		(float)g_width / (float)g_height, //KINECT_V2_DEPTH_ASPECT_RATIO, 
		view_matrix_inv.data());
	//ttimer.print_interval("Raycast             : ");

#endif





	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("kernel failed");

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pixel_buffer, 0));
}




void GLRaycastTextureWidget::initPixelBuffer()
{
	if (pixelBuf)
	{
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pixel_buffer));

		pixelBuf->destroy();
		texture->destroy();
		delete pixelBuf;
		delete texture;
	}

	std::vector<GLubyte> image;
	for (GLuint i = 0; i < g_width; i++)
	{
		for (GLuint j = 0; j < g_height; j++)
		{
			GLuint c = ((((i & 0x4) == 0) ^ ((j & 0x4)) == 0)) * 255;
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
	pixelBuf->allocate(image.data(), g_width * g_height * sizeof(GLubyte) * 4);
	pixelBuf->release();

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pixel_buffer, pixelBuf->bufferId(), cudaGraphicsMapFlagsWriteDiscard));

	texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	texture->create();
	texture->bind();
	texture->setMinificationFilter(QOpenGLTexture::Nearest);
	texture->setMagnificationFilter(QOpenGLTexture::Nearest);
	texture->setWrapMode(QOpenGLTexture::ClampToEdge);
	texture->setSize(g_width, g_height);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	texture->setFormat(QOpenGLTexture::TextureFormat::RGBA8_UNorm);
	texture->allocateStorage(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::UInt8);
	//texture->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::UInt8, image.data());
	texture->release();

}


void GLRaycastTextureWidget::setup(
	const std::string& filepath,
	ushort vx_count,
	ushort vx_size)
{
	int vol_size = vx_count * vx_size;
	float half_vol_size = vol_size * 0.5f;

	position.setX( -vol_size * 0.5f);
	position.setY( -vol_size * 0.5f);
	position.setZ( -vol_size * 0.5f );

	this->voxel_count = vx_count;
	this->voxel_size = vx_size;
	this->fov = KINECT_V2_FOVY;

	Eigen::Vector3i voxel_size(vx_size, vx_size, vx_size);
	Eigen::Vector3i volume_size(vol_size, vol_size, vol_size);
	Eigen::Vector3i voxel_count(vx_count, vx_count, vx_count);
	int total_voxels = voxel_count.x() * voxel_count.y() * voxel_count.z();


	std::cout << std::fixed
		<< "Voxel Count  : " << voxel_count.transpose() << std::endl
		<< "Voxel Size   : " << voxel_size.transpose() << std::endl
		<< "Volume Size  : " << volume_size.transpose() << std::endl
		<< "Total Voxels : " << total_voxels << std::endl
		<< std::endl;

	ttimer.start();
	KinectFrame knt(filepath);
	ttimer.print_interval("Importing knt frame : ");

	Eigen::Affine3f grid_affine = Eigen::Affine3f::Identity();
	grid_affine.translate(Eigen::Vector3f(0, 0, half_vol_size));
	grid_affine.scale(Eigen::Vector3f(1, 1, 1));	// z is negative inside of screen
	Eigen::Matrix4f grid_matrix = grid_affine.matrix();

	float knt_near_plane = 0.1f;
	float knt_far_plane = 10240.0f;
	Eigen::Matrix4f projection = perspective_matrix<float>(KINECT_V2_FOVY, KINECT_V2_DEPTH_ASPECT_RATIO, knt_near_plane, knt_far_plane);
	Eigen::Matrix4f projection_inverse = projection.inverse();
	Eigen::Matrix4f view_matrix = Eigen::Matrix4f::Identity();

	std::vector<float4> vertices(knt.depth.size(), make_float4(0, 0, 0, 1));
	std::vector<float4> normals(knt.depth.size(), make_float4(0, 0, 1, 1));
	std::vector<Eigen::Vector2f> grid_voxels_params(total_voxels);

	// 
	// setup image parameters
	unsigned short image_width = g_width;
	unsigned short image_height = image_width / aspect_ratio;

	knt_cuda_setup(
		vx_count, vx_size,
		grid_matrix.data(),
		projection.data(),
		projection_inverse.data(),
		*grid_voxels_params.data()->data(),
		g_width,
		g_height,
		KINECT_V2_DEPTH_MIN,
		KINECT_V2_DEPTH_MAX,
		vertices.data()[0],
		normals.data()[0],
		image_width,
		image_height
		);

	ttimer.start();
	knt_cuda_allocate();
	knt_cuda_init_grid();
	ttimer.print_interval("Allocating gpu      : ");


#if 0
	//
	// use this with vx_count = 3 and vx_size = 1 to debug
	// 
	std::vector<Eigen::Vector2f> tsdf(total_voxels, Eigen::Vector2f::Ones());

	for (int z = 0; z < vx_count; ++z)
	{
		for (int x = 0; x < vx_count; ++x)
		{
			tsdf.at(vx_count * vx_count + x)[0] = -1.f;
		}
	}

	tsdf.at(0)[0] = tsdf.at(1)[0] = tsdf.at(2)[0] = 
	tsdf.at(9)[0] = tsdf.at(10)[0] = tsdf.at(11)[0] =
	tsdf.at(18)[0] = tsdf.at(19)[0] = tsdf.at(20)[0] = -1.f;
	//tsdf.at(13)[0] = 
	//tsdf.at(22)[0] = 
	//tsdf.at(18)[0] = 
	//tsdf.at(26)[0] = -1.0f;
	knt_cuda_grid_sample_test(tsdf.data()->data(), tsdf.size());
#else
	ttimer.start();
	knt_cuda_copy_host_to_device();
	knt_cuda_copy_depth_buffer_to_device(knt.depth.data());
	ttimer.print_interval("Copy host to device : ");

	ttimer.start();
	knt_cuda_normal_estimation();
	ttimer.print_interval("Normal estimation   : ");

	ttimer.start();
	knt_cuda_update_grid(view_matrix.data());
	ttimer.print_interval("Update grid         : ");
#endif
}


void GLRaycastTextureWidget::cleanup()
{
	ttimer.start();
	knt_cuda_free();
	ttimer.print_interval("Cleanup gpu         : ");
}

