
#include "VolumeRenderWidget.h"

#if 0
#include "GLModel.h"
#include "GLQuad.h"
#include <QMouseEvent>
#include <QTimer>
#include <math.h>

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include "KinectFusionKernels/KinectFusionKernels.h"




static VolumeType *h_volume = nullptr;
static const char *volumeFilename = "Bucky.raw";
static std::string volume_file_path = "";
static cudaExtent volumeSize = make_cudaExtent(32, 32, 32);


uint g_width = 512, g_height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *cuda_timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float fpsResult = 0.f;
int g_Index = 0;
unsigned int frameCount = 0;

int *pArgc;
char **pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif



extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);



static void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&cuda_timer) / 1000.f);
		sprintf(fps, "Volume Render: %3.1f fps", ifps);

		fpsCount = 0;

		fpsLimit = (int)MAX(1.f, ifps);
		sdkResetTimer(&cuda_timer);

		fpsResult = ifps;
	}

}

// Load raw data from disk
static void *loadRawFile(const char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");

	if (!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return 0;
	}

	void *data = malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

#if defined(_MSC_VER_)
	printf("Read '%s', %Iu bytes\n", filename, read);
#else
	printf("Read '%s', %zu bytes\n", filename, read);
#endif

	return data;
}




VolumeRenderWidget::VolumeRenderWidget(QWidget *parent) :
	QOpenGLTrackballWidget(parent)
	, pixelBuf(nullptr)
	, texture(nullptr)
	, cudaInitialized(false)
{
	QTimer* updateTimer = new QTimer(this);
	connect(updateTimer, SIGNAL(timeout()), this, SLOT(update()));
	updateTimer->start(33);

	weelSpeed = 0.01f;
	distance = -4;
}


VolumeRenderWidget::~VolumeRenderWidget()
{
	// cleanup
	// 
	sdkDeleteTimer(&cuda_timer);

	volume_render_cleanup();

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}


void VolumeRenderWidget::setup(const std::string& buffer_file_path, const size_t x, const size_t y, const size_t z)
{
	volumeSize = make_cudaExtent(x, y, z);
	volume_file_path = buffer_file_path;
}


void VolumeRenderWidget::initializeGL()
{
    initializeOpenGLFunctions();

	std::stringstream info;

	info << "OpenGl information: VENDOR:       " << (const char*)glGetString(GL_VENDOR) << std::endl
		<< "                    RENDERER:     " << (const char*)glGetString(GL_RENDERER) << std::endl
		<< "                    VERSION:      " << (const char*)glGetString(GL_VERSION) << std::endl
		<< "                    GLSL VERSION: " << (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	std::cout << std::endl << info.str() << std::endl << std::endl;

	//GL_KHR_debug extension must be available in the context in order to access the messages logged by OpenGL
	if (!context()->hasExtension(QByteArrayLiteral("GL_KHR_debug")))
	{
		qDebug() << "Error: Could not start Opengl logger";
	}
	else
	{
		logger = new QOpenGLDebugLogger(this);
		logger->initialize(); // initializes in the current context
		//connect(logger, &QOpenGLDebugLogger::messageLogged, this, &LogHandler::handleLoggedMessage);
		//logger->startLogging();
	}


	



    glClearColor(0, 0, 0, 1);


	quad = new GLQuad();
	quad->initGL();

	QList<QOpenGLDebugMessage> messages = logger->loggedMessages();
	foreach(const QOpenGLDebugMessage &message, messages)
		qDebug() << "Init GL: " << message << "\n";





	cudaInit();

	initPixelBuffer();
}



void VolumeRenderWidget::cudaInit()
{
	cudaInitialized = false;
	int devID = gpuGetMaxGflopsDeviceId();
	if (devID < 0)
		std::cerr << "Error: No CUDA capable devices found" << std::endl;
	else
		checkCudaErrors(cudaGLSetGLDevice(devID));

	cudaInitialized = true;

	size_t size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(VolumeType);
	VolumeType *h_volume = (VolumeType*)loadRawFile(volume_file_path.c_str(), size);

	if (!h_volume)
		std::cerr << "Error: Could not load volume buffer: " << volume_file_path << std::endl;
	else
	{
		volume_render_setup(h_volume, volumeSize.width, volumeSize.height, volumeSize.depth);
		free(h_volume);
	}

	sdkCreateTimer(&cuda_timer);

	// calculate new grid size
	gridSize = dim3(iDivUp(g_width, blockSize.x), iDivUp(g_height, blockSize.y));
}


void VolumeRenderWidget::initPixelBuffer()
{
#if 0
	if (pbo)
	{
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	// create pixel buffer object for display
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, g_width * g_height*sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, g_width, g_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

#else

	if (pixelBuf)
	{
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

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
	pixelBuf->allocate(image.data(), g_width * g_height * sizeof(GLubyte) * 4);
	//pixelBuf->allocate(q_image.bits(), q_image.width() * q_image.height() * sizeof(GLubyte) * 4);
	pixelBuf->release();

	//pixelBuf->allocate(sizeof(image));
	//void *vid = pixelBuf->map(QOpenGLBuffer::WriteOnly);
	//memcpy(vid, data, sizeof(image));
	//pixelBuf->unmap();

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pixelBuf->bufferId(), cudaGraphicsMapFlagsWriteDiscard));

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
#endif
}






void VolumeRenderWidget::resizeGL(int w, int h)
{
	glViewport(0, 0, w, h);
}

void VolumeRenderWidget::cudaRender()
{
	if (!pixelBuf || !pixelBuf->isCreated())
		return;

	// render image using CUDA
	copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	checkCudaErrors(cudaMemset(d_output, 0, g_width*g_height * 4));

	// call CUDA kernel, writing results to PBO
	render_kernel(gridSize, blockSize, d_output, g_width, g_height, density, brightness, transferOffset, transferScale);

	getLastCudaError("kernel failed");

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void VolumeRenderWidget::paintGL()
{
	if (!texture)
		return;

	sdkStartTimer(&cuda_timer);


	QMatrix4x4 view_matrix;
	view_matrix.translate(0, 0, -distance);
	view_matrix.rotate(rotation);

#if 1
	memcpy(invViewMatrix, view_matrix.transposed().data(), 12 * sizeof(float));
#else
	invViewMatrix[0] = 1.f;
	invViewMatrix[1] = 0.f;
	invViewMatrix[2] = 0.f;
	invViewMatrix[3] = 0.f;
	invViewMatrix[4] = 0.f;
	invViewMatrix[5] = 1.f;
	invViewMatrix[6] = 0.f;
	invViewMatrix[7] = 0.f;
	invViewMatrix[8] = 0.f;
	invViewMatrix[9] = 0.f;
	invViewMatrix[10] = 1.f;
	invViewMatrix[11] = 4.f;
#endif
	cudaRender();

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	texture->bind();

	// copy from pbo to texture
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	pixelBuf->bind();
	{
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 512, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		texture->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::UInt8, (void*)nullptr);
	}
	pixelBuf->release();
	
	// Clear color and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	quad->render();


	texture->release();

	sdkStopTimer(&cuda_timer);
	computeFPS();
}



void VolumeRenderWidget::keyPressEvent(QKeyEvent* e)
{
	if (e->key() == Qt::Key_Q || e->key() == Qt::Key_Escape)
		close();
}
#endif