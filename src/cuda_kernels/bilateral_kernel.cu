/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include "cuda_kernels.h"
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions

__constant__ float cGaussian[64];   //gaussian array in device side
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
texture<float4, 2, cudaReadModeElementType> float4Tex;
texture<float, 2> floatTex;
texture<uchar, 2> ucharTex;
texture<ushort, 2> ushortTex;


uint *dImage = NULL;   //original image
uint *dTemp = NULL;   //temp array for iterations
size_t pitch;

/*
Perform a simple bilateral filter.

Bilateral filter is a nonlinear filter that is a mixture of range
filter and domain filter, the previous one preserves crisp edges and
the latter one filters noise. The intensity value at each pixel in
an image is replaced by a weighted average of intensity values from
nearby pixels.

The weight factor is calculated by the product of domain filter
component(using the gaussian distribution as a spatial distance) as
well as range filter component(Euclidean distance between center pixel
and the current neighbor pixel). Because this process is nonlinear,
the sample just uses a simple pixel by pixel step.

Texture fetches automatically clamp to edge of image. 1D gaussian array
is mapped to a 1D texture instead of using shared memory, which may
cause severe bank conflict.

Threads are y-pass(column-pass), because the output is coalesced.

Parameters
od - pointer to output data in global memory
d_f - pointer to the 1D gaussian array
e_d - euclidean delta
w  - image width
h  - image height
r  - filter radius
*/

//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d)
{

	float mod = (b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z);

	return __expf(-mod / (2.f * d * d));
}

__device__ float euclideanLen(float a, float b, float d)
{
	 float mod = (b - a) * (b - a);

	return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ uint rgbaFloatToInt(float value)
{
	uint v = __saturatef(fabs(value));   // clamp to [0.0, 1.0]
	return uint(value * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c)
{
	float4 rgba;
	rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
	rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
	rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
	rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
	return rgba;
}

//column pass using coalesced global memory reads
__global__ void
d_bilateral_filter(uint *od, int w, int h, float e_d, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}


	float sum = 0.0f;
	float factor;
	float4 t = { 0.f, 0.f, 0.f, 0.f };
	float4 center = tex2D(rgbaTex, x, y);

	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			float4 curPix = tex2D(rgbaTex, x + j, y + i);

			factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
				euclideanLen(curPix, center, e_d);             //range factor

			t += factor * curPix;
			sum += factor;
		}
	}

	od[y * w + x] = rgbaFloatToInt(t / sum);
}


__global__ void
d_bilateral_filter_uchar(uchar *out, int w, int h, float e_d, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}


	float sum = 0.0f;
	float factor;

	float t = 0;

	float center = ((float)tex2D(ucharTex, x, y)) / 255.0f;

	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			float curPix = ((float)tex2D(ucharTex, x + j, y + i)) / 255.0f;

			factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
				euclideanLen(curPix, center, e_d);             //range factor

			t += factor * curPix;
			sum += factor;
		}
	}

	float pixel_float = (t / sum) * 255.f;
	out[y * w + x] = (uchar)pixel_float;
}



__global__ void
d_bilateral_filter_ushort(ushort *out, int w, int h, ushort max_depth, float e_d, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}


//	ushort pixel = tex2D(ushortTex, x, y);
//	out[y * w + x] = pixel;
//	return;


	float sum = 0.0f;
	float factor;
	float t = 0;
	float center = ((float)tex2D(ushortTex, x, y)) / (float)max_depth;

	
	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			float curPix = ((float)tex2D(ushortTex, x + j, y + i)) / (float)max_depth;

			factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
				euclideanLen(curPix, center, e_d);             //range factor

			t += factor * curPix;
			sum += factor;
		}
	}

	out[y * w + x] = (ushort)(t / sum * max_depth);

	

	//float pixel_float = (t / sum) * max_depth;
	//out[y * w + x] = (ushort)pixel_float;
}







__global__ void
d_bilateral_filter_normal_estimate_ushort(float *out, int w, int h, ushort max_depth, float e_d, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}


	//ushort pixel = tex2D(ushortTex, x, y);
	//out[y * w + x] = (float)pixel;
	//return;

	float sum = 0.0f;
	float factor;
	float t = 0;
	float center = ((float)tex2D(ushortTex, x, y)) / (float)max_depth;


	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			float curPix = ((float)tex2D(ushortTex, x + j, y + i)) / (float)max_depth;

			factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
				euclideanLen(curPix, center, e_d);             //range factor

			t += factor * curPix;
			sum += factor;
		}
	}

	out[y * w + x] = (t / sum * max_depth);
	
}




__global__ void
d_bilateral_filter_normal_estimate_float(float *out, int w, int h, ushort max_depth, float e_d, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}


	//ushort pixel = tex2D(ushortTex, x, y);
	//out[y * w + x] = (float)pixel;
	//return;

	float sum = 0.0f;
	float factor;
	float t = 0;
	float center = ((float)tex2D(floatTex, x, y)) / (float)max_depth;


	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			float curPix = ((float)tex2D(floatTex, x + j, y + i)) / (float)max_depth;

			factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
				euclideanLen(curPix, center, e_d);             //range factor

			t += factor * curPix;
			sum += factor;
		}
	}

	out[y * w + x] = (t / sum * max_depth);

}





__global__ void
d_bilateral_filter_normal_estimate_ushort_float4(float4 *out, int w, int h, ushort max_depth, float e_d, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}

	float sum = 0.0f;
	float factor;
	float t = 0;
	float center = ((float)tex2D(ushortTex, x, y)) / (float)max_depth;


	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			float curPix = ((float)tex2D(ushortTex, x + j, y + i)) / (float)max_depth;

			factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
				euclideanLen(curPix, center, e_d);             //range factor

			t += factor * curPix;
			sum += factor;
		}
	}

	out[y * w + x].x = (t / sum * max_depth);
	out[y * w + x].y = (t / sum * max_depth);
	out[y * w + x].z = (t / sum * max_depth);
	out[y * w + x].w = (float)max_depth;
}


__global__ void
d_bilateral_filter_normal_estimate_float4(float4 *out, int w, int h, ushort max_depth, float e_d, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}


	float sum = 0.0f;
	float factor;
	float4 t = { 0.f, 0.f, 0.f, 0.f };
	float4 center = tex2D(float4Tex, x, y);
	center /= (float)max_depth;


	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			float4 curPix = tex2D(float4Tex, x + j, y + i);
			curPix /= (float)max_depth;

			factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
				euclideanLen(curPix, center, e_d);             //range factor

			t += factor * curPix;
			sum += factor;
		}
	}

	out[y * w + x] = t / sum * max_depth;

	//out[y * w + x].x = (t.x / sum * max_depth);
	//out[y * w + x].y = (t.y / sum * max_depth);
	//out[y * w + x].z = (t.z / sum * max_depth);
	//out[y * w + x].w = (t.w / sum * max_depth);
}




__global__ void
d_passthrough_texture_ushort(ushort* pImage, int w, int h)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}

	ushort pixel = tex2D(ushortTex, x, y);
	pImage[y * w + x] = pixel;

	return;
}


__global__ void
d_passthrough_texture_uchar(uchar* pImage, int w, int h)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}

	uchar pixel = tex2D(ucharTex, x, y);
	pImage[y * w + x] = 255 - pixel;

	return;
}





extern "C"
void initTexture(int width, int height, uint *hImage)
{
	// copy image data to array
	checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(uint)*width, height));
	checkCudaErrors(cudaMallocPitch(&dTemp, &pitch, sizeof(uint)*width, height));
	checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint)*width,
		sizeof(uint)*width, height, cudaMemcpyHostToDevice));
}

extern "C"
void freeTextures()
{
	checkCudaErrors(cudaFree(dImage));
	checkCudaErrors(cudaFree(dTemp));
}

/*
Because a 2D gaussian mask is symmetry in row and column,
here only generate a 1D mask, and use the product by row
and column index later.

1D gaussian distribution :
g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

parameters:
og - output gaussian array in global memory
delta - the 2nd parameter 'd' in the above function
radius - half of the filter size
(total filter size = 2 * radius + 1)
*/
extern "C"
void updateGaussian(float delta, int radius)
{
	float  fGaussian[64];

	for (int i = 0; i < 2 * radius + 1; ++i)
	{
		float x = i - radius;
		fGaussian[i] = expf(-(x*x) / (2 * delta*delta));
	}

	checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2 * radius + 1)));
}

/*
Perform 2D bilateral filter on image using CUDA

Parameters:
d_dest - pointer to destination image in device memory
width  - image width
height - image height
e_d    - euclidean delta
radius - filter radius
iterations - number of iterations
*/

// RGBA version
extern "C"
double bilateralFilterRGBA(uint *dDest,
int width, int height,
float e_d, int radius, int iterations,
StopWatchInterface *timer)
{
	// var for kernel computation timing
	double dKernelTime;

	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, width, height, pitch));

	for (int i = 0; i<iterations; i++)
	{
		// sync host and start kernel computation timer
		dKernelTime = 0.0;
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);

		dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
		dim3 blockSize(16, 16);
		d_bilateral_filter << < gridSize, blockSize >> >(
			dDest, width, height, e_d, radius);

		// sync host and stop computation timer
		checkCudaErrors(cudaDeviceSynchronize());
		dKernelTime += sdkGetTimerValue(&timer);

		if (iterations > 1)
		{
			// copy result back from global memory to array
			checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
				sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
		}
	}

	return ((dKernelTime / 1000.) / (double)iterations);
}



// Gray version
extern "C"
double bilateralFilterGray(
	uchar *dOutputImage,
	uchar *dInputImage,
	int width, int height, size_t pitch,
	float e_d, int radius, int iterations,
	StopWatchInterface *timer)
{

	uchar* dTempImage = nullptr;
	checkCudaErrors(cudaMallocPitch(&dTempImage, &pitch, sizeof(uchar) * width, height));


	// var for kernel computation timing
	double dKernelTime = 0.0;

	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
	checkCudaErrors(cudaBindTexture2D(0, ucharTex, dInputImage, desc, width, height, pitch));


	const dim3 threads_per_block(16, 16);
	dim3 num_blocks;
	num_blocks.x = (width + threads_per_block.x - 1) / threads_per_block.x;
	num_blocks.y = (height + threads_per_block.y - 1) / threads_per_block.y;


	//cudaMemcpy2D(
	//	dOutputImage,
	//	sizeof(uchar) * width,
	//	dInputImage,
	//	pitch,
	//	sizeof(uchar) * width,
	//	height,
	//	cudaMemcpyDeviceToDevice);


	for (int i = 0; i<iterations; i++)
	{
		// sync host and start kernel computation timer
		dKernelTime = 0.0;
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);

		d_bilateral_filter_uchar <<<  num_blocks, threads_per_block >>>(dOutputImage, width, height, e_d, radius);

		// sync host and stop computation timer
		checkCudaErrors(cudaDeviceSynchronize());
		dKernelTime += sdkGetTimerValue(&timer);

		if (iterations > 1)
		{
			// copy result back from global memory to array
			checkCudaErrors(cudaMemcpy2D(
				dTempImage, 
				pitch, 
				dOutputImage, 
				sizeof(uchar) * width,
				sizeof(uchar) * width, 
				height, 
				cudaMemcpyDeviceToDevice));

			checkCudaErrors(cudaBindTexture2D(0, ucharTex, dTempImage, desc, width, height, pitch));
		}
	}


	return ((dKernelTime / 1000.) / (double)iterations);
}


// Gray version
extern "C"
double bilateralFilter_ushort(
ushort *dOutputImage,
ushort *dInputImage,
int width, int height, size_t pitch,
ushort max_depth,
float e_d, int radius, int iterations,
StopWatchInterface *timer)
{

	ushort* dTempImage = nullptr;
	checkCudaErrors(cudaMallocPitch(&dTempImage, &pitch, sizeof(ushort) * width, height));


	// var for kernel computation timing
	double dKernelTime = 0.0;

	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
	checkCudaErrors(cudaBindTexture2D(0, ushortTex, dInputImage, desc, width, height, pitch));


	const dim3 threads_per_block(32, 32);
	dim3 num_blocks;
	num_blocks.x = (width + threads_per_block.x - 1) / threads_per_block.x;
	num_blocks.y = (height + threads_per_block.y - 1) / threads_per_block.y;


	//cudaMemcpy2D(
	//	dOutputImage,
	//	sizeof(uchar) * width,
	//	dInputImage,
	//	pitch,
	//	sizeof(uchar) * width,
	//	height,
	//	cudaMemcpyDeviceToDevice);


	for (int i = 0; i<iterations; i++)
	{
		// sync host and start kernel computation timer
		dKernelTime = 0.0;
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);

		d_bilateral_filter_ushort << <  num_blocks, threads_per_block >> >(dOutputImage, width, height, max_depth, e_d, radius);

		// sync host and stop computation timer
		checkCudaErrors(cudaDeviceSynchronize());
		dKernelTime += sdkGetTimerValue(&timer);

		if (iterations > 1)
		{
			// copy result back from global memory to array
			checkCudaErrors(cudaMemcpy2D(
				dTempImage,
				pitch,
				dOutputImage,
				sizeof(ushort) * width,
				sizeof(ushort) * width,
				height,
				cudaMemcpyDeviceToDevice));

			checkCudaErrors(cudaBindTexture2D(0, ushortTex, dTempImage, desc, width, height, pitch));
		}
	}


	return ((dKernelTime / 1000.) / (double)iterations);
}


extern "C"
double bilateralFilter_normal_estimate(
float *dOutputImage,
ushort *dInputImage,
int width,
int height,
size_t in_pitch,
size_t out_pitch,
ushort max_depth,
float e_d,
int radius,
int iterations,
StopWatchInterface *timer)
{
	float* dTempImage = nullptr;
	checkCudaErrors(cudaMallocPitch(&dTempImage, &out_pitch, sizeof(float) * width, height));


	// var for kernel computation timing
	double dKernelTime = 0.0;

	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
	checkCudaErrors(cudaBindTexture2D(0, ushortTex, dInputImage, desc, width, height, in_pitch));


	const dim3 threads_per_block(32, 32);
	dim3 num_blocks;
	num_blocks.x = (width + threads_per_block.x - 1) / threads_per_block.x;
	num_blocks.y = (height + threads_per_block.y - 1) / threads_per_block.y;


	//cudaMemcpy2D(
	//	dOutputImage,
	//	sizeof(uchar) * width,
	//	dInputImage,
	//	pitch,
	//	sizeof(uchar) * width,
	//	height,
	//	cudaMemcpyDeviceToDevice);


	// sync host and start kernel computation timer
	dKernelTime = 0.0;
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&timer);

	d_bilateral_filter_normal_estimate_ushort << <  num_blocks, threads_per_block >> >(dOutputImage, width, height, max_depth, e_d, radius);

	// sync host and stop computation timer
	checkCudaErrors(cudaDeviceSynchronize());
	dKernelTime += sdkGetTimerValue(&timer);

	if (iterations > 1)
	{
		// copy result back from global memory to array
		checkCudaErrors(cudaMemcpy2D(
			dTempImage,
			out_pitch,
			dOutputImage,
			sizeof(float) * width,
			sizeof(float) * width,
			height,
			cudaMemcpyDeviceToDevice));

		cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float>();
		checkCudaErrors(cudaBindTexture2D(0, floatTex, dTempImage, desc_float, width, height, out_pitch));
	}


	for (int i = 1; i < iterations; i++)
	{
		// sync host and start kernel computation timer
		dKernelTime = 0.0;
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);

		d_bilateral_filter_normal_estimate_float << <  num_blocks, threads_per_block >> >(dOutputImage, width, height, max_depth, e_d, radius);

		// sync host and stop computation timer
		checkCudaErrors(cudaDeviceSynchronize());
		dKernelTime += sdkGetTimerValue(&timer);

		// copy result back from global memory to array
		checkCudaErrors(cudaMemcpy2D(
			dTempImage,
			out_pitch,
			dOutputImage,
			sizeof(float) * width,
			sizeof(float) * width,
			height,
			cudaMemcpyDeviceToDevice));

		cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float>();
		checkCudaErrors(cudaBindTexture2D(0, floatTex, dTempImage, desc_float, width, height, out_pitch));
	}


	return ((dKernelTime / 1000.) / (double)iterations);
}


extern "C"
double bilateralFilter_normal_estimate_float4(
float4 *dOutputImage,
ushort *dInputImage,
int width,
int height,
size_t in_pitch,
size_t out_pitch,
ushort max_depth,
float e_d,
int radius,
int iterations,
StopWatchInterface *timer)
{
	float4* dTempImage = nullptr;
	checkCudaErrors(cudaMallocPitch(&dTempImage, &out_pitch, sizeof(float4) * width, height));


	// var for kernel computation timing
	double dKernelTime = 0.0;

	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
	checkCudaErrors(cudaBindTexture2D(0, ushortTex, dInputImage, desc, width, height, in_pitch));


	const dim3 threads_per_block(32, 32);
	dim3 num_blocks;
	num_blocks.x = (width + threads_per_block.x - 1) / threads_per_block.x;
	num_blocks.y = (height + threads_per_block.y - 1) / threads_per_block.y;



	// sync host and start kernel computation timer
	dKernelTime = 0.0;
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&timer);

	d_bilateral_filter_normal_estimate_ushort_float4 << <  num_blocks, threads_per_block >> >(dOutputImage, width, height, max_depth, e_d, radius);

	// sync host and stop computation timer
	checkCudaErrors(cudaDeviceSynchronize());
	dKernelTime += sdkGetTimerValue(&timer);


	if (iterations > 1)
	{
		// copy result back from global memory to array
		checkCudaErrors(cudaMemcpy2D(
			dTempImage,
			out_pitch,
			dOutputImage,
			sizeof(float4) * width,
			sizeof(float4) * width,
			height,
			cudaMemcpyDeviceToDevice));

		cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture2D(0, float4Tex, dTempImage, desc_float, width, height, out_pitch));
	}


	for (int i = 1; i < iterations; i++)
	{
		// sync host and start kernel computation timer
		dKernelTime = 0.0;
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);

		d_bilateral_filter_normal_estimate_float4 << <  num_blocks, threads_per_block >> >(dOutputImage, width, height, max_depth, e_d, radius);

		// sync host and stop computation timer
		checkCudaErrors(cudaDeviceSynchronize());
		dKernelTime += sdkGetTimerValue(&timer);

		// copy result back from global memory to array
		checkCudaErrors(cudaMemcpy2D(
			dTempImage,
			out_pitch,
			dOutputImage,
			sizeof(float4) * width,
			sizeof(float4) * width,
			height,
			cudaMemcpyDeviceToDevice));

		cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture2D(0, float4Tex, dTempImage, desc_float, width, height, out_pitch));
	}
	


	checkCudaErrors(cudaFree(dTempImage));

	return ((dKernelTime / 1000.) / (double)iterations);
}


void passthrough_texture_ushort(ushort* dOutputImage, ushort* dInputImage, int width, int height, size_t pitch)
{
	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort>();
	checkCudaErrors(cudaBindTexture2D(0, ushortTex, dInputImage, desc, width, height, pitch));

	const dim3 threads_per_block(16, 16);
	dim3 num_blocks;
	num_blocks.x = (width + threads_per_block.x - 1) / threads_per_block.x;
	num_blocks.y = (height + threads_per_block.y - 1) / threads_per_block.y;

	d_passthrough_texture_ushort << <  num_blocks, threads_per_block >> >(dOutputImage, width, height);
}