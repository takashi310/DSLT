/*
 * DSLT Demo
 *
 * Copyright (C) 2014 Kyoto University
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License or any
 * later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 
#include <cstdio>
#include <cstdlib>
#include <string>
#include <assert.h>
#define DLL_CONVOLUTIONSEPARABLE
#include "convolutionSeparable_common.h"
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization



 __declspec (dllexport) void CuDeviceProp(void)
{
 	printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		return;
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		char msg[256];
		sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
			(float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
		printf("%s", msg);

		printf("  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}
#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
		}
#endif

		printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
			deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
			deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#ifdef WIN32
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}

	// csv masterlog info
	// *****************************
	// exe and CUDA driver name
	printf("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
	char cTemp[16];

	// driver version
	sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
	sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
	sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
	sProfileString +=  cTemp;

	// Runtime version
	sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
	sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
	sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
	sProfileString +=  cTemp;

	// Device count
	sProfileString += ", NumDevs = ";
#ifdef WIN32
	sprintf_s(cTemp, 10, "%d", deviceCount);
#else
	sprintf(cTemp, "%d", deviceCount);
#endif
	sProfileString += cTemp;

	// Print Out all device Names
	for (dev = 0; dev < deviceCount; ++dev)
	{
#ifdef _WIN32
		sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
		sprintf(cTemp, ", Device%d = ", dev);
#endif
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		sProfileString += cTemp;
		sProfileString += deviceProp.name;
	}

	sProfileString += "\n";
	printf("%s", sProfileString.c_str());
}

__global__ void binarizeKernel(
    float *d_Dst,
	float *d_Gau,
    float *d_Src,
    int imageW,
	int pitchY,
	int pitchZ,
	float constC
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW){
		d_Dst[id] = (d_Src[id] > d_Gau[id] - constC) ? BINALIZE_UPPER_VAL : BINALIZE_LOWER_VAL;
		//d_Dst[id] = d_Gau[id];
	}
}

extern "C" __declspec (dllexport) void binarizeGPU(
    float *d_Dst,
	float *d_Gau,
    float *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	float constC
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    binarizeKernel<<<blocks, threads>>>(
        d_Dst,
		d_Gau,
        d_Src,
		imageW,
		imageW,
		imageW * imageH,
		constC
    );
    getLastCudaError("binarizeKernel() execution failed\n");
}

__global__ void thresholdKernel(
    float *d_Dst,
	float *d_Src,
    int imageW,
	int pitchY,
	int pitchZ,
	float th
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW){
		d_Dst[id] = (th > d_Src[id]) ? BINALIZE_LOWER_VAL : BINALIZE_UPPER_VAL;
	}
}

extern "C" __declspec (dllexport) void thresholdGPU(
    float *d_Dst,
	float *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	float th
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    thresholdKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src,
		imageW,
		imageW,
		imageW * imageH,
		th
    );
    getLastCudaError("binarizeKernel() execution failed\n");
}


__global__ void pixelwiseOrKernel(
    float *d_Dst,
	float *d_Src1,
    float *d_Src2,
    int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW){
		d_Dst[id] = (d_Src1[id] >= BINALIZE_UPPER_VAL || d_Src2[id] >= BINALIZE_UPPER_VAL) ? BINALIZE_UPPER_VAL : BINALIZE_LOWER_VAL;
		//d_Dst[id] = d_Gau[id];
	}
}

extern "C" __declspec (dllexport) void pixelwiseOrGPU(
    float *d_Dst,
	float *d_Src1,
    float *d_Src2,
	int imageW,
    int imageH,
	int imageZ
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    pixelwiseOrKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src1,
        d_Src2,
		imageW,
		imageW,
		imageW * imageH
    );
    getLastCudaError("pixelwiseOrKernel() execution failed\n");
}


__global__ void pixelwiseAndKernel(
    float *d_Dst,
	float *d_Src1,
    float *d_Src2,
    int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW){
		d_Dst[id] = (d_Src1[id] > (BINALIZE_LOWER_VAL + 0.001f) && d_Src2[id] > (BINALIZE_LOWER_VAL + 0.001f)) ? BINALIZE_UPPER_VAL : BINALIZE_LOWER_VAL;
		//d_Dst[id] = d_Gau[id];
	}
}

extern "C" __declspec (dllexport) void pixelwiseAndGPU(
    float *d_Dst,
	float *d_Src1,
    float *d_Src2,
	int imageW,
    int imageH,
	int imageZ
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    pixelwiseAndKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src1,
        d_Src2,
		imageW,
		imageW,
		imageW * imageH
    );
    getLastCudaError("pixelwiseOrKernel() execution failed\n");
}

template <typename T> __global__ void fillValKernel(
    T *d_Dst,
	int imageW,
	int pitchY,
	int pitchZ,
	T val
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW)d_Dst[id] = val;
}

extern "C" __declspec (dllexport) void fillIntGPU(
    int *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	int val
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    fillValKernel<int><<<blocks, threads>>>(
        d_Dst,
		imageW,
		imageW,
		imageW * imageH,
		val
    );
    getLastCudaError("fillIntKernel() execution failed\n");
}

extern "C" __declspec (dllexport) void fillFloatGPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	float val
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    fillValKernel<float><<<blocks, threads>>>(
        d_Dst,
		imageW,
		imageW,
		imageW * imageH,
		val
    );
    getLastCudaError("fillFloatKernel() execution failed\n");
}

__global__ void fillFloatKernelROI(
    float *d_Dst,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	float val
){
	const int id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Dst[id] = val;
}

extern "C" __declspec (dllexport) void fillFloatROIGPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d,
	float val
){

    dim3 blocks(roi_w/BINALIZE_BLOCKDIM + 1, roi_h, roi_d);
    dim3 threads(BINALIZE_BLOCKDIM);

    fillFloatKernelROI<<<blocks, threads>>>(
        d_Dst,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w,
		val
    );
    getLastCudaError("fillFloatKernelROI() execution failed\n");
}

__global__ void addFloatKernel(
    float *d_Dst,
	float *d_Src1,
	float *d_Src2,
	int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW)d_Dst[id] = d_Src1[id] + d_Src2[id];
}

extern "C" __declspec (dllexport) void addFloatGPU(
    float *d_Dst,
	float *d_Src1,
	float *d_Src2,
	int imageW,
    int imageH,
	int imageZ
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    addFloatKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src1,
		d_Src2,
		imageW,
		imageW,
		imageW * imageH
    );
    getLastCudaError("addFloatKernel() execution failed\n");
}

__global__ void subtractFloatKernelROI(
    float *d_Dst,
	float *d_Src1,
	float *d_Src2,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w
){
	const int id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Dst[id] = (d_Src1[id] - d_Src2[id] > 0.0f) ? d_Src1[id] - d_Src2[id] : 0.0f;
}

extern "C" __declspec (dllexport) void subtractFloatROIGPU(
    float *d_Dst,
	float *d_Src1,
	float *d_Src2,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){

    dim3 blocks(roi_w/BINALIZE_BLOCKDIM + 1, roi_h, roi_d);
    dim3 threads(BINALIZE_BLOCKDIM);

    subtractFloatKernelROI<<<blocks, threads>>>(
        d_Dst,
		d_Src1,
		d_Src2,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w
    );
    getLastCudaError("fillFloatKernelROI() execution failed\n");
}

__global__ void divFloatKernel(
    float *d_Dst,
	float *d_Numer,
	float *d_Denom,
	int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW){
		if(d_Denom[id] != 0.0f)d_Dst[id] = d_Numer[id] / d_Denom[id];
		else d_Dst[id] = 2.0f;
	}
}

extern "C" __declspec (dllexport) void divFloatGPU(
    float *d_Dst,
	float *d_Numer,
	float *d_Denom,
	int imageW,
    int imageH,
	int imageZ
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    divFloatKernel<<<blocks, threads>>>(
        d_Dst,
		d_Numer,
		d_Denom,
		imageW,
		imageW,
		imageW * imageH
    );
    getLastCudaError("divFloatKernel() execution failed\n");
}

__global__ void addFloatValueKernel(
    float *d_Dst,
	int imageW,
	int pitchY,
	int pitchZ,
	float val
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW)d_Dst[id] += val;
}

extern "C" __declspec (dllexport) void addFloatValueGPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	float val
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    addFloatValueKernel<<<blocks, threads>>>(
        d_Dst,
		imageW,
		imageW,
		imageW * imageH,
		val
    );
    getLastCudaError("fillFloatKernel() execution failed\n");
}

template <typename T> __global__ void setMinValKernel(
    T *d_Dst,
	int imageW,
	int pitchY,
	int pitchZ,
	T val
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW)d_Dst[id] = (d_Dst[id] < val) ? val : d_Dst[id];
}

extern "C" __declspec (dllexport) void setMinValIntGPU(
    int *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	int val
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    setMinValKernel<int><<<blocks, threads>>>(
        d_Dst,
		imageW,
		imageW,
		imageW * imageH,
		val
    );
    getLastCudaError("setMinValKernel() execution failed\n");
}

extern "C" __declspec (dllexport) void setMinValFloatGPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	float val
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    setMinValKernel<float><<<blocks, threads>>>(
        d_Dst,
		imageW,
		imageW,
		imageW * imageH,
		val
    );
    getLastCudaError("setMinValKernel() execution failed\n");
}

__global__ void cropFloatKernel(
    float *d_Dst,
	float *d_Src,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_pitchY,
	int roi_pitchZ
){
	const int src_id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	const int dst_id = blockIdx.z*roi_pitchZ + blockIdx.y*roi_pitchY + blockIdx.x*BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Dst[dst_id] = d_Src[src_id];
}

extern "C" __declspec (dllexport) void cropFloatGPU(
    float *d_Dst,
	float *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){
	assert(roi_x + roi_w < imageW && roi_y + roi_h < imageH && roi_z + roi_d < imageZ);

    dim3 blocks((roi_w%BINALIZE_BLOCKDIM > 0) ? roi_w/BINALIZE_BLOCKDIM + 1 : roi_w/BINALIZE_BLOCKDIM, roi_h, roi_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    cropFloatKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w,
		roi_w,
		roi_w * roi_h
    );
    getLastCudaError("cropFloatKernel() execution failed\n");
}

__global__ void cropIntKernel(
    int *d_Dst,
	int *d_Src,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_pitchY,
	int roi_pitchZ
){
	const int src_id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	const int dst_id = blockIdx.z*roi_pitchZ + blockIdx.y*roi_pitchY + blockIdx.x*BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Dst[dst_id] = d_Src[src_id];
}

extern "C" __declspec (dllexport) void cropIntGPU(
    int *d_Dst,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){
	assert(roi_x + roi_w < imageW && roi_y + roi_h < imageH && roi_z + roi_d < imageZ);

    dim3 blocks((roi_w%BINALIZE_BLOCKDIM > 0) ? roi_w/BINALIZE_BLOCKDIM + 1 : roi_w/BINALIZE_BLOCKDIM, roi_h, roi_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    cropIntKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w,
		roi_w,
		roi_w * roi_h
    );
    getLastCudaError("fillFloatKernelROI() execution failed\n");
}

__global__ void copySingleSegmentIntOffsetKernel(
    int *d_Dst,
	int *d_Src,
	int seg_id,
	int dst_w,
	int dst_h,
	int dst_d,
	int dst_pitchY,
	int dst_pitchZ,
	int x_offset,
	int y_offset,
	int z_offset,
	int src_w,
	int src_pitchY,
	int src_pitchZ
){
	const int src_id = blockIdx.z*src_pitchZ + blockIdx.y*src_pitchY + blockIdx.x*BINALIZE_BLOCKDIM + threadIdx.x;
	const int dst_x = x_offset + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	const int dst_y = blockIdx.y + y_offset;
	const int dst_z = blockIdx.z + z_offset;
	const int dst_id = dst_z * dst_pitchZ + dst_y * dst_pitchY + dst_x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < src_w && dst_x < dst_w && dst_y < dst_h && dst_z < dst_d)
		if((d_Src[src_id] == seg_id))d_Dst[dst_id] = d_Src[src_id];
}

extern "C" __declspec (dllexport) void copySingleSegmentIntOffsetGPU(
    int *d_Dst,
	int *d_Src,
	int seg_id,
	int dst_w,
    int dst_h,
	int dst_d,
	int x_offset,
	int y_offset,
	int z_offset,
	int src_w,
	int src_h,
	int src_d
){
	dim3 blocks((src_w%BINALIZE_BLOCKDIM > 0) ? src_w/BINALIZE_BLOCKDIM + 1 : src_w/BINALIZE_BLOCKDIM, src_h, src_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    copySingleSegmentIntOffsetKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src,
		seg_id,
		dst_w,
		dst_h,
		dst_d,
		dst_w,
		dst_w*dst_h,
		x_offset,
		y_offset,
		z_offset,
		src_w,
		src_w,
		src_w * src_h
    );
    getLastCudaError("copySingleSegmentIntOffsetKernel() execution failed\n");
}

__global__ void copySingleSegmentROIKernel(
    float *d_Dst,
	int *d_Seg,
	int seg_id,
	float val,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w
){
	const int id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Dst[id] = (d_Seg[id] == seg_id) ? val : d_Dst[id];
}

extern "C" __declspec (dllexport) void copySingleSegmentROIGPU(
    float *d_Dst,
	int *d_Seg,
	int seg_id,
	float val,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){
	assert(roi_x + roi_w < imageW && roi_y + roi_h < imageH && roi_z + roi_d < imageZ);

    dim3 blocks((roi_w%BINALIZE_BLOCKDIM > 0) ? roi_w/BINALIZE_BLOCKDIM + 1 : roi_w/BINALIZE_BLOCKDIM, roi_h, roi_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    copySingleSegmentROIKernel<<<blocks, threads>>>(
        d_Dst,
		d_Seg,
		seg_id,
		val,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w
    );
    getLastCudaError("fillFloatKernelROI() execution failed\n");
}

__global__ void overwriteSingleSegmentROIKernel(
    int *d_Seg,
	int seg_id,
	float *d_Mask,
	float th_val,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w
){
	const int id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Seg[id] = (d_Mask[id] >= th_val) ? seg_id : d_Seg[id];
}

extern "C" __declspec (dllexport) void overwriteSingleSegmentROIGPU(
    int *d_Seg,
	int seg_id,
	float *d_Mask,
	float th_val,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){
	assert(roi_x + roi_w < imageW && roi_y + roi_h < imageH && roi_z + roi_d < imageZ);

    dim3 blocks((roi_w%BINALIZE_BLOCKDIM > 0) ? roi_w/BINALIZE_BLOCKDIM + 1 : roi_w/BINALIZE_BLOCKDIM, roi_h, roi_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    overwriteSingleSegmentROIKernel<<<blocks, threads>>>(
		d_Seg,
		seg_id,
		d_Mask,
		th_val,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w
    );
    getLastCudaError("overwriteSingleSegmentROIKernel() execution failed\n");
}

template <typename T> __global__ void replacementROIKernel(
    T *d_Dst,
	T src_val,
	T dst_val,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w
){
	const int id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Dst[id] = (d_Dst[id] == src_val) ? dst_val : d_Dst[id];
}

extern "C" __declspec (dllexport) void replacementIntROIGPU(
    int *d_Dst,
	int src_val,
	int dst_val,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){
	assert(roi_x + roi_w < imageW && roi_y + roi_h < imageH && roi_z + roi_d < imageZ);

    dim3 blocks((roi_w%BINALIZE_BLOCKDIM > 0) ? roi_w/BINALIZE_BLOCKDIM + 1 : roi_w/BINALIZE_BLOCKDIM, roi_h, roi_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    replacementROIKernel<<<blocks, threads>>>(
        d_Dst,
		src_val,
		dst_val,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w
    );
    getLastCudaError("replacementIntROIGPU() execution failed\n");
}

extern "C" __declspec (dllexport) void replacementFloatROIGPU(
    float *d_Dst,
	float src_val,
	float dst_val,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){
	assert(roi_x + roi_w < imageW && roi_y + roi_h < imageH && roi_z + roi_d < imageZ);

    dim3 blocks((roi_w%BINALIZE_BLOCKDIM > 0) ? roi_w/BINALIZE_BLOCKDIM + 1 : roi_w/BINALIZE_BLOCKDIM, roi_h, roi_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    replacementROIKernel<<<blocks, threads>>>(
        d_Dst,
		src_val,
		dst_val,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w
    );
    getLastCudaError("replacementIntROIGPU() execution failed\n");
}

__global__ void setValToSegmentedRegionROIKernel(
    float *d_Dst,
	int *d_Seg,
	float val,
	int imageW,
	int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w
){
	const int id = (blockIdx.z + roi_z) * pitchZ + (blockIdx.y + roi_y) * pitchY + roi_x + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < roi_w)d_Dst[id] = (d_Seg[id] >= 0) ? val : d_Dst[id];
}

extern "C" __declspec (dllexport) void setValToSegmentedRegionROIGPU(
    float *d_Dst,
	int *d_Seg,
	float val,
	int imageW,
    int imageH,
	int imageZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_w,
	int roi_h,
	int roi_d
){
	assert(roi_x + roi_w < imageW && roi_y + roi_h < imageH && roi_z + roi_d < imageZ);

    dim3 blocks((roi_w%BINALIZE_BLOCKDIM > 0) ? roi_w/BINALIZE_BLOCKDIM + 1 : roi_w/BINALIZE_BLOCKDIM, roi_h, roi_d);
	dim3 threads(BINALIZE_BLOCKDIM);

    setValToSegmentedRegionROIKernel<<<blocks, threads>>>(
        d_Dst,
		d_Seg,
		val,
		imageW,
		imageW,
		imageW * imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_w
    );
    getLastCudaError("setValToSegmentedRegionROIKernel() execution failed\n");
}


template <typename T> __global__ void notEqualIntKernel(
    float *d_Dst,
	T *d_Src1,
	T *d_Src2,
	int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW)d_Dst[id] = (d_Src1[id] != d_Src2[id]) ? 1.0f : 0.0f;
}

extern "C" __declspec (dllexport) void notEqualIntGPU(
    float *d_Dst,
	int *d_Src1,
	int *d_Src2,
	int imageW,
    int imageH,
	int imageZ
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    notEqualIntKernel<int><<<blocks, threads>>>(
        d_Dst,
		d_Src1,
		d_Src2,
		imageW,
		imageW,
		imageW * imageH
    );
    getLastCudaError("notEqualIntKernel() execution failed\n");
}

/*
#define   SRC_BLOCKDIM 1024
#define   HMP_BLOCKDIM  64
#define   ROWS_BLOCKDIM_Z 1
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void depthMapKernel(
    float *d_Dst,
	float *d_hmp,
	int imageW,
    int imageH,
	int imageZ
){
	__shared__ float s_Data[2][SRC_BLOCKDIM];

	const int srcZ = (blockIdx.x * SRC_BLOCKDIM + threadIdx.x)/(imageW*imageH);
	const int srcY = ((blockIdx.x * SRC_BLOCKDIM + threadIdx.x)%(imageW*imageH))/imageW;
	const int srcX = ((blockIdx.x * SRC_BLOCKDIM + threadIdx.x)%(imageW*imageH))%imageW;
	const int hmpId = blockIdx.y * HMP_BLOCKDIM + (threadIdx.x % HMP_BLOCKDIM);

	s_Data[0][threadIdx.x] = d_hmp[blockIdx.y * HMP_BLOCKDIM + (threadIdx.x % HMP_BLOCKDIM)];
	s_Data[1][threadIdx.x] = FLT_MAX;
	
	__syncthreads();
	
	#pragma unroll
    for(int i = 0; i < HMP_BLOCKDIM; i++){
		float d;
		//int hmpX = (blockIdx.y * HMP_BLOCKDIM + (threadIdx.x % HMP_BLOCKDIM) + i) % imageW;
		//int hmpY = (blockIdx.y * HMP_BLOCKDIM + (threadIdx.x % HMP_BLOCKDIM) + i) / imageW;
		d = (srcX - ((hmpId + i)%imageW))*(srcX - ((hmpId + i)%imageW)) + (srcY - ((hmpId + i)/imageW))*(srcY - ((hmpId + i)/imageW)) + (srcZ - s_Data[0][(threadIdx.x + i) % SRC_BLOCKDIM])*(srcZ - s_Data[0][(threadIdx.x + i) % SRC_BLOCKDIM]);
		if(s_Data[1][threadIdx.x] > d)s_Data[1][threadIdx.x] = d;
	}
	
	if(d_Dst[blockIdx.x * SRC_BLOCKDIM + threadIdx.x] > s_Data[1][threadIdx.x])d_Dst[blockIdx.x * SRC_BLOCKDIM + threadIdx.x] = s_Data[1][threadIdx.x];
	//d_Dst[blockIdx.x * SRC_BLOCKDIM + threadIdx.x] = 1.0f;
}

extern "C" __declspec (dllexport) void depthMapGPU(
    float *d_dmp,
	float *d_hmp,
	int imageW,
    int imageH,
	int imageZ
){

	assert( SRC_BLOCKDIM % HMP_BLOCKDIM == 0 && SRC_BLOCKDIM > HMP_BLOCKDIM);
	assert( (imageW*imageH*imageZ) % SRC_BLOCKDIM == 0);
	assert( (imageW*imageH) % HMP_BLOCKDIM == 0);

    dim3 blocks((imageW*imageH*imageZ)/SRC_BLOCKDIM, (imageW*imageH)/HMP_BLOCKDIM);
    dim3 threads(SRC_BLOCKDIM);

    depthMapKernel<<<blocks, threads>>>(
        d_dmp,
		d_hmp,
		imageW,
		imageH,
		imageZ
	);
    getLastCudaError("depthMapKernel() execution failed\n");
}

*/

#define   SHARED_MEMORY_SIZE_FLOAT 128

__global__ void depthMapKernel(
    float *d_Dst,
	float *d_hmp,
	int imageW,
	int imageH,
	int offset
){
	__shared__ float s_Data[2][SHARED_MEMORY_SIZE_FLOAT];

	const int hmpX = (offset + threadIdx.x) % imageW;
	const int hmpY = (offset + threadIdx.x) / imageW;

	s_Data[0][threadIdx.x] = (offset + threadIdx.x < imageW*imageH) ? d_hmp[offset + threadIdx.x] : FLT_MAX;
	__syncthreads();

	s_Data[1][threadIdx.x] =(blockIdx.x - hmpX)*(blockIdx.x - hmpX) + (blockIdx.y - hmpY)*(blockIdx.y - hmpY) + (blockIdx.z - s_Data[0][threadIdx.x])*(blockIdx.z - s_Data[0][threadIdx.x]);
	__syncthreads();

	if(threadIdx.x == 0){
		float dmin = s_Data[1][0];
		#pragma unroll
		for(int i = 1; i < SHARED_MEMORY_SIZE_FLOAT; i++)if(dmin > s_Data[1][i])dmin = s_Data[1][i];
		if(d_Dst[blockIdx.z*imageH*imageW + blockIdx.y*imageW + blockIdx.x] > dmin)d_Dst[blockIdx.z*imageH*imageW + blockIdx.y*imageW + blockIdx.x] = dmin;
	}
}

__global__ void sqrtKernel(
    float *d_Dst,
	int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW){
		d_Dst[id] = sqrt(d_Dst[id]);
	}
}

extern "C" __declspec (dllexport) void depthMapGPU(
    float *d_dmp,
	float *d_hmp,
	int imageW,
    int imageH,
	int imageZ
){
	/*
	assert( SRC_BLOCKDIM % HMP_BLOCKDIM == 0 && SRC_BLOCKDIM > HMP_BLOCKDIM);
	assert( (imageW*imageH*imageZ) % SRC_BLOCKDIM == 0);
	assert( (imageW*imageH) % HMP_BLOCKDIM == 0);
	*/
    dim3 blocks(imageW, imageH, imageZ);
    dim3 threads(SHARED_MEMORY_SIZE_FLOAT);
	for(int i = 0; i < (imageW*imageH/SHARED_MEMORY_SIZE_FLOAT+1); i++){
		
		depthMapKernel<<<blocks, threads>>>(
			d_dmp,
			d_hmp,
			imageW,
			imageH,
			i*SHARED_MEMORY_SIZE_FLOAT
		);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	sqrtKernel<<<blocks, threads>>>(
        d_dmp,
		imageW,
		imageW,
		imageW * imageH
	);

    getLastCudaError("depthMapKernel() execution failed\n");
}
