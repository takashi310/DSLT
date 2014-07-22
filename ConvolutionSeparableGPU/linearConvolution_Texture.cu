#include <assert.h>
#include <stdio.h>
#define DLL_CONVOLUTIONSEPARABLE
#include "convolutionSeparable_common.h"
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <helper_math.h>

#define F_PI 3.141592653589793238462f

#define LIST_LENGTH_MAXIMUM 512
#define LCT_BLOCKDIM 32

texture<float, 3, cudaReadModeElementType> tex;  // 3D texture

__constant__ float c_PointListAndWeight[LIST_LENGTH_MAXIMUM*4];

cudaArray *d_volumeArray = 0;
unsigned int _kernel_length = 0;


//h_PointListÇÕx,y,z,x,y,z,x,y,z,...ÇÃï¿Ç—ÇÃàÍéüå≥îzóÒ
extern "C" __declspec (dllexport) void setPointList_LCT(float *h_PointListAndWeight, int list_length){
	float tmpList[LIST_LENGTH_MAXIMUM*4];

	assert( list_length <= LIST_LENGTH_MAXIMUM );
	assert( list_length > 0 );
	_kernel_length = list_length;
	
	for(int i = 0; i < LIST_LENGTH_MAXIMUM; i++){
		if(i < list_length){
			tmpList[i*4]      = h_PointListAndWeight[i*4];
			tmpList[i*4 + 1]  = h_PointListAndWeight[i*4 + 1];
			tmpList[i*4 + 2]  = h_PointListAndWeight[i*4 + 2];
			tmpList[i*4 + 3]  = h_PointListAndWeight[i*4 + 3];
		}
		else{
			tmpList[i*4]      = 0.0f;
			tmpList[i*4 + 1]  = 0.0f;
			tmpList[i*4 + 2]  = 0.0f;
			tmpList[i*4 + 3]  = 0.0f;
		}
	}
	checkCudaErrors( cudaMemcpyToSymbol(c_PointListAndWeight, tmpList, LIST_LENGTH_MAXIMUM * 4 * sizeof(float)) );

}

extern "C" __declspec (dllexport) void destructCuda_LCT(){
	if(d_volumeArray)checkCudaErrors(cudaFreeArray(d_volumeArray));
	d_volumeArray = 0;
}

extern "C" __declspec (dllexport) void initCuda_LCT(const float *h_volume, int imageW, int imageH, int imageZ){

	destructCuda_LCT();

	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_volume, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex.normalized = false;                      // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(&tex, d_volumeArray, &channelDesc));
}


template<unsigned int length> __global__ void linearConvolutionMinKernel_Texture(
    float *d_Dst,
    int imageW,
	int pitchY,
	int pitchZ,
	float *d_ArgMin_lati,
	float lati,
	float *d_ArgMin_longi,
	float longi
){
	
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * LCT_BLOCKDIM + threadIdx.x;
	const int idx = blockIdx.x * LCT_BLOCKDIM + threadIdx.x;
	float sum = 0.0f;
	if(idx < imageW){
		#pragma unroll
		for(int i = 0; i < length; i++){
			sum += tex3D(tex, (float)(idx + c_PointListAndWeight[i*4]) + 0.5f,  (float)(blockIdx.y + c_PointListAndWeight[i*4 + 1]) + 0.5f,  (float)(blockIdx.z + c_PointListAndWeight[i*4 + 2]) + 0.5f)  * c_PointListAndWeight[i*4 + 3];
		}
	}
	__syncthreads();
	if(idx < imageW && d_Dst[id] > sum){
		d_Dst[id] = sum;
		if(d_ArgMin_lati  != NULL)d_ArgMin_lati[id]  = lati;
		if(d_ArgMin_longi != NULL)d_ArgMin_longi[id] = longi;
	}
	
/*
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * LCT_BLOCKDIM + threadIdx.x;
	const int idx = blockIdx.x * LCT_BLOCKDIM + threadIdx.x;
	float sum = 0.0f, left = 0.0f, right = 0.0f;
	if(idx < imageW){
		#pragma unroll
		for(int i = 0; i < length-1; i += 2){
			left += tex3D(tex, (float)(idx + c_PointListAndWeight[i*4]) + 0.5f,  (float)(blockIdx.y + c_PointListAndWeight[i*4 + 1]) + 0.5f,  (float)(blockIdx.z + c_PointListAndWeight[i*4 + 2]) + 0.5f)  * c_PointListAndWeight[i*4 + 3];
		}
		sum += left;
		#pragma unroll
		for(int i = 1; i < length-1; i += 2){
			right += tex3D(tex, (float)(idx + c_PointListAndWeight[i*4]) + 0.5f,  (float)(blockIdx.y + c_PointListAndWeight[i*4 + 1]) + 0.5f,  (float)(blockIdx.z + c_PointListAndWeight[i*4 + 2]) + 0.5f)  * c_PointListAndWeight[i*4 + 3];
		}
		sum += right;
		sum += tex3D(tex, (float)(idx + c_PointListAndWeight[(length-1)*4]) + 0.5f,  (float)(blockIdx.y + c_PointListAndWeight[(length-1)*4 + 1]) + 0.5f,  (float)(blockIdx.z + c_PointListAndWeight[(length-1)*4 + 2]) + 0.5f)  * c_PointListAndWeight[(length-1)*4 + 3];
	}
	__syncthreads();
	if(idx < imageW && (d_ArgMin_longi[id] > left || d_ArgMin_longi[id] > right)){
		d_Dst[id] = sum;
		if(d_ArgMin_lati  != NULL)d_ArgMin_lati[id]  = lati;
		if(d_ArgMin_longi != NULL)d_ArgMin_longi[id] = (left < right) ? left : right;
	}
*/
}



template<unsigned int blockSize> __global__ void linearConvolutionMinKernel_Texture_v2(
    float *d_Dst,
	int k_length,
    int imageW,
	int pitchY,
	int pitchZ,
	float *d_ArgMin_lati,
	float lati,
	float *d_ArgMin_longi,
	float longi
){
	__shared__ float sdata[blockSize];//threadÇ∆ìØÇ∂êî(blockSize)

	const unsigned int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x;
	const unsigned int tid = threadIdx.x;
	unsigned int i = tid;
	sdata[tid] = 0;
	while(i < k_length){
		sdata[tid] += tex3D(tex, (float)(blockIdx.x + c_PointListAndWeight[i*4]) + 0.5f,
								 (float)(blockIdx.y + c_PointListAndWeight[i*4 + 1]) + 0.5f,
								 (float)(blockIdx.z + c_PointListAndWeight[i*4 + 2]) + 0.5f) * c_PointListAndWeight[i*4 + 3];
		i += blockSize;
	}
	__syncthreads();

	if(blockSize >= 256){ if(tid < 128){ sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128){ if(tid < 64) { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }
	if(blockSize >= 64) { if(tid < 32) { sdata[tid] += sdata[tid + 32];  } __syncthreads(); }
	if(blockSize >= 32) { if(tid < 16) { sdata[tid] += sdata[tid + 16];  } __syncthreads(); }
	if(blockSize >= 16) { if(tid < 8)  { sdata[tid] += sdata[tid + 8];   } __syncthreads(); }
	if(blockSize >= 8)  { if(tid < 4)  { sdata[tid] += sdata[tid + 4];   } __syncthreads(); }
	if(blockSize >= 4)  { if(tid < 2)  { sdata[tid] += sdata[tid + 2];   } __syncthreads(); }
	if(blockSize >= 2)  { if(tid < 1)  { sdata[tid] += sdata[tid + 1];   } __syncthreads(); }

	if(tid == 0){
		if(d_Dst[id] > sdata[0]){
			d_Dst[id] = sdata[0];
			if(d_ArgMin_lati  != NULL)d_ArgMin_lati[id]  = lati;
			if(d_ArgMin_longi != NULL)d_ArgMin_longi[id] = longi;
		}
	}
}

/*
__global__ void linearConvolutionKernel_Texture(
    float *d_Dst,
    int imageW,
	int pitchY,
	int pitchZ
){
	__shared__ float sums[LCT_BLOCKDIM][2];

	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * LCT_BLOCKDIM + threadIdx.x;
	const int idx = blockIdx.x * LCT_BLOCKDIM + threadIdx.x;
	if(idx < imageW){
		sums[threadIdx.x][threadIdx.y] = tex3D(tex, (float)idx, (float)blockIdx.y, (float)blockIdx.z) * c_KernelCenter[0];
		#pragma unroll
		for(int i = 0; i < KERNEL_RADIUS_MAXIMUM; i++){
			sums[threadIdx.x][threadIdx.y] += tex3D(tex,
													(float)(idx + c_PointList[(KERNEL_RADIUS_MAXIMUM*threadIdx.y + i)*3]),
													(float)(blockIdx.y + c_PointList[(KERNEL_RADIUS_MAXIMUM*threadIdx.y + i)*3 + 1]),
													(float)(blockIdx.z + c_PointList[(KERNEL_RADIUS_MAXIMUM*threadIdx.y + i)*3 + 2]) ) * c_Kernel[threadIdx.y*KERNEL_RADIUS_MAXIMUM + i];
		}
		
	}
	__syncthreads();
	if(idx < imageW && threadIdx.y == 0)d_Dst[id] = sums[threadIdx.x][0] + sums[threadIdx.x][1];
}
*/

/*
__global__ void linearConvolutionKernel_Texture(
    float *d_Dst,
    int imageW,
	int pitchY,
	int pitchZ
){
	__shared__ float s_Data[LCT_BLOCKDIM][2][KERNEL_RADIUS_MAXIMUM];
	__shared__ float s_c[LCT_BLOCKDIM];

	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * LCT_BLOCKDIM + threadIdx.x;
	const int idx = blockIdx.x * LCT_BLOCKDIM + threadIdx.x;

	s_c[threadIdx.x] = tex3D(tex, (float)idx, (float)blockIdx.y, (float)blockIdx.z);
	#pragma unroll
	for(int i = 0; i < KERNEL_RADIUS_MAXIMUM; i++){
		s_Data[threadIdx.x][0][i] = tex3D(tex, (float)(idx + c_PointListLeft[i*3]),  (float)(blockIdx.y + c_PointListLeft[i*3 + 1]),  (float)(blockIdx.z + c_PointListLeft[i*3 + 2]));
		s_Data[threadIdx.x][1][i] = tex3D(tex, (float)(idx + c_PointListRight[i*3]), (float)(blockIdx.y + c_PointListRight[i*3 + 1]), (float)(blockIdx.z + c_PointListRight[i*3 + 2]));
	}
	__syncthreads();
	if(idx < imageW){
		float sum = s_c[threadIdx.x] * c_KernelCenter[0];
		#pragma unroll
		for(int i = 0; i < KERNEL_RADIUS_MAXIMUM; i++){
			sum += s_Data[threadIdx.x][0][i]  * c_KernelLeft[i];
			sum += s_Data[threadIdx.x][1][i] * c_KernelRight[i];
		}
		d_Dst[id] = sum;
	}
}
*/

extern "C" __declspec (dllexport) void linearConvolutionMinGPU_Texture(
    float *d_Dst,
    int imageW,
    int imageH,
	int imageZ,
	float *d_ArgMin_lati,
	float lati,
	float *d_ArgMin_longi,
	float longi
){
/*	dim3 blocks(imageW, imageH, imageZ);
    dim3 threads(LCT_BLOCKDIM);

	linearConvolutionMinKernel_Texture_v2<LCT_BLOCKDIM><<<blocks, threads>>>(d_Dst, _kernel_length, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi);
*/

	dim3 blocks(imageW/LCT_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(LCT_BLOCKDIM);
	switch(_kernel_length){
		case 1:  linearConvolutionMinKernel_Texture<1><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 2:  linearConvolutionMinKernel_Texture<2><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 3:  linearConvolutionMinKernel_Texture<3><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 4:  linearConvolutionMinKernel_Texture<4><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 5:  linearConvolutionMinKernel_Texture<5><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 6:  linearConvolutionMinKernel_Texture<6><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 7:  linearConvolutionMinKernel_Texture<7><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 8:  linearConvolutionMinKernel_Texture<8><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 9:  linearConvolutionMinKernel_Texture<9><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 10: linearConvolutionMinKernel_Texture<10><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 11: linearConvolutionMinKernel_Texture<11><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 12: linearConvolutionMinKernel_Texture<12><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 13: linearConvolutionMinKernel_Texture<13><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 14: linearConvolutionMinKernel_Texture<14><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 15: linearConvolutionMinKernel_Texture<15><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 16: linearConvolutionMinKernel_Texture<16><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 17: linearConvolutionMinKernel_Texture<17><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 18: linearConvolutionMinKernel_Texture<18><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 19: linearConvolutionMinKernel_Texture<19><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 20: linearConvolutionMinKernel_Texture<20><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 21: linearConvolutionMinKernel_Texture<21><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 22: linearConvolutionMinKernel_Texture<22><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 23: linearConvolutionMinKernel_Texture<23><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 24: linearConvolutionMinKernel_Texture<24><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 25: linearConvolutionMinKernel_Texture<25><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 26: linearConvolutionMinKernel_Texture<26><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 27: linearConvolutionMinKernel_Texture<27><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 28: linearConvolutionMinKernel_Texture<28><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 29: linearConvolutionMinKernel_Texture<29><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 30: linearConvolutionMinKernel_Texture<30><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 31: linearConvolutionMinKernel_Texture<31><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 32: linearConvolutionMinKernel_Texture<32><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 33: linearConvolutionMinKernel_Texture<33><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 34: linearConvolutionMinKernel_Texture<34><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 35: linearConvolutionMinKernel_Texture<35><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 36: linearConvolutionMinKernel_Texture<36><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 37: linearConvolutionMinKernel_Texture<37><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 38: linearConvolutionMinKernel_Texture<38><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 39: linearConvolutionMinKernel_Texture<39><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 40: linearConvolutionMinKernel_Texture<40><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 41: linearConvolutionMinKernel_Texture<41><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 42: linearConvolutionMinKernel_Texture<42><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 43: linearConvolutionMinKernel_Texture<43><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 44: linearConvolutionMinKernel_Texture<44><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 45: linearConvolutionMinKernel_Texture<45><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 46: linearConvolutionMinKernel_Texture<46><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 47: linearConvolutionMinKernel_Texture<47><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 48: linearConvolutionMinKernel_Texture<48><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 49: linearConvolutionMinKernel_Texture<49><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 50: linearConvolutionMinKernel_Texture<50><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 51: linearConvolutionMinKernel_Texture<51><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 52: linearConvolutionMinKernel_Texture<52><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 53: linearConvolutionMinKernel_Texture<53><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 54: linearConvolutionMinKernel_Texture<54><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 55: linearConvolutionMinKernel_Texture<55><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 56: linearConvolutionMinKernel_Texture<56><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 57: linearConvolutionMinKernel_Texture<57><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 58: linearConvolutionMinKernel_Texture<58><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 59: linearConvolutionMinKernel_Texture<59><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 60: linearConvolutionMinKernel_Texture<60><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 61: linearConvolutionMinKernel_Texture<61><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 62: linearConvolutionMinKernel_Texture<62><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 63: linearConvolutionMinKernel_Texture<63><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 64: linearConvolutionMinKernel_Texture<64><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 65: linearConvolutionMinKernel_Texture<65><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 66: linearConvolutionMinKernel_Texture<66><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 67: linearConvolutionMinKernel_Texture<67><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 68: linearConvolutionMinKernel_Texture<68><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case 69: linearConvolutionMinKernel_Texture<69><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	70	: linearConvolutionMinKernel_Texture<	70	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	71	: linearConvolutionMinKernel_Texture<	71	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	72	: linearConvolutionMinKernel_Texture<	72	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	73	: linearConvolutionMinKernel_Texture<	73	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	74	: linearConvolutionMinKernel_Texture<	74	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	75	: linearConvolutionMinKernel_Texture<	75	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	76	: linearConvolutionMinKernel_Texture<	76	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	77	: linearConvolutionMinKernel_Texture<	77	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	78	: linearConvolutionMinKernel_Texture<	78	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	79	: linearConvolutionMinKernel_Texture<	79	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	80	: linearConvolutionMinKernel_Texture<	80	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	81	: linearConvolutionMinKernel_Texture<	81	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	82	: linearConvolutionMinKernel_Texture<	82	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	83	: linearConvolutionMinKernel_Texture<	83	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	84	: linearConvolutionMinKernel_Texture<	84	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	85	: linearConvolutionMinKernel_Texture<	85	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	86	: linearConvolutionMinKernel_Texture<	86	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	87	: linearConvolutionMinKernel_Texture<	87	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	88	: linearConvolutionMinKernel_Texture<	88	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	89	: linearConvolutionMinKernel_Texture<	89	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	90	: linearConvolutionMinKernel_Texture<	90	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	91	: linearConvolutionMinKernel_Texture<	91	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	92	: linearConvolutionMinKernel_Texture<	92	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	93	: linearConvolutionMinKernel_Texture<	93	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	94	: linearConvolutionMinKernel_Texture<	94	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	95	: linearConvolutionMinKernel_Texture<	95	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	96	: linearConvolutionMinKernel_Texture<	96	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	97	: linearConvolutionMinKernel_Texture<	97	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	98	: linearConvolutionMinKernel_Texture<	98	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	99	: linearConvolutionMinKernel_Texture<	99	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	100	: linearConvolutionMinKernel_Texture<	100	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	101	: linearConvolutionMinKernel_Texture<	101	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	102	: linearConvolutionMinKernel_Texture<	102	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	103	: linearConvolutionMinKernel_Texture<	103	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	104	: linearConvolutionMinKernel_Texture<	104	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	105	: linearConvolutionMinKernel_Texture<	105	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	106	: linearConvolutionMinKernel_Texture<	106	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	107	: linearConvolutionMinKernel_Texture<	107	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	108	: linearConvolutionMinKernel_Texture<	108	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	109	: linearConvolutionMinKernel_Texture<	109	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	110	: linearConvolutionMinKernel_Texture<	110	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	111	: linearConvolutionMinKernel_Texture<	111	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	112	: linearConvolutionMinKernel_Texture<	112	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	113	: linearConvolutionMinKernel_Texture<	113	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	114	: linearConvolutionMinKernel_Texture<	114	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	115	: linearConvolutionMinKernel_Texture<	115	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	116	: linearConvolutionMinKernel_Texture<	116	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	117	: linearConvolutionMinKernel_Texture<	117	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	118	: linearConvolutionMinKernel_Texture<	118	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	119	: linearConvolutionMinKernel_Texture<	119	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	120	: linearConvolutionMinKernel_Texture<	120	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	121	: linearConvolutionMinKernel_Texture<	121	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	122	: linearConvolutionMinKernel_Texture<	122	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	123	: linearConvolutionMinKernel_Texture<	123	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	124	: linearConvolutionMinKernel_Texture<	124	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	125	: linearConvolutionMinKernel_Texture<	125	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	126	: linearConvolutionMinKernel_Texture<	126	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	127	: linearConvolutionMinKernel_Texture<	127	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	128	: linearConvolutionMinKernel_Texture<	128	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	129	: linearConvolutionMinKernel_Texture<	129	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	130	: linearConvolutionMinKernel_Texture<	130	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	131	: linearConvolutionMinKernel_Texture<	131	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	132	: linearConvolutionMinKernel_Texture<	132	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	133	: linearConvolutionMinKernel_Texture<	133	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	134	: linearConvolutionMinKernel_Texture<	134	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	135	: linearConvolutionMinKernel_Texture<	135	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	136	: linearConvolutionMinKernel_Texture<	136	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	137	: linearConvolutionMinKernel_Texture<	137	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	138	: linearConvolutionMinKernel_Texture<	138	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	139	: linearConvolutionMinKernel_Texture<	139	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	140	: linearConvolutionMinKernel_Texture<	140	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	141	: linearConvolutionMinKernel_Texture<	141	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	142	: linearConvolutionMinKernel_Texture<	142	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	143	: linearConvolutionMinKernel_Texture<	143	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	144	: linearConvolutionMinKernel_Texture<	144	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	145	: linearConvolutionMinKernel_Texture<	145	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	146	: linearConvolutionMinKernel_Texture<	146	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	147	: linearConvolutionMinKernel_Texture<	147	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	148	: linearConvolutionMinKernel_Texture<	148	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	149	: linearConvolutionMinKernel_Texture<	149	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	150	: linearConvolutionMinKernel_Texture<	150	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	151	: linearConvolutionMinKernel_Texture<	151	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	152	: linearConvolutionMinKernel_Texture<	152	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	153	: linearConvolutionMinKernel_Texture<	153	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	154	: linearConvolutionMinKernel_Texture<	154	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	155	: linearConvolutionMinKernel_Texture<	155	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	156	: linearConvolutionMinKernel_Texture<	156	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	157	: linearConvolutionMinKernel_Texture<	157	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	158	: linearConvolutionMinKernel_Texture<	158	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	159	: linearConvolutionMinKernel_Texture<	159	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	160	: linearConvolutionMinKernel_Texture<	160	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	161	: linearConvolutionMinKernel_Texture<	161	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	162	: linearConvolutionMinKernel_Texture<	162	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	163	: linearConvolutionMinKernel_Texture<	163	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	164	: linearConvolutionMinKernel_Texture<	164	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	165	: linearConvolutionMinKernel_Texture<	165	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	166	: linearConvolutionMinKernel_Texture<	166	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	167	: linearConvolutionMinKernel_Texture<	167	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	168	: linearConvolutionMinKernel_Texture<	168	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	169	: linearConvolutionMinKernel_Texture<	169	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	170	: linearConvolutionMinKernel_Texture<	170	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	171	: linearConvolutionMinKernel_Texture<	171	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	172	: linearConvolutionMinKernel_Texture<	172	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	173	: linearConvolutionMinKernel_Texture<	173	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	174	: linearConvolutionMinKernel_Texture<	174	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	175	: linearConvolutionMinKernel_Texture<	175	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	176	: linearConvolutionMinKernel_Texture<	176	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	177	: linearConvolutionMinKernel_Texture<	177	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	178	: linearConvolutionMinKernel_Texture<	178	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	179	: linearConvolutionMinKernel_Texture<	179	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	180	: linearConvolutionMinKernel_Texture<	180	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	181	: linearConvolutionMinKernel_Texture<	181	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	182	: linearConvolutionMinKernel_Texture<	182	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	183	: linearConvolutionMinKernel_Texture<	183	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	184	: linearConvolutionMinKernel_Texture<	184	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	185	: linearConvolutionMinKernel_Texture<	185	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	186	: linearConvolutionMinKernel_Texture<	186	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	187	: linearConvolutionMinKernel_Texture<	187	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	188	: linearConvolutionMinKernel_Texture<	188	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	189	: linearConvolutionMinKernel_Texture<	189	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	190	: linearConvolutionMinKernel_Texture<	190	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	191	: linearConvolutionMinKernel_Texture<	191	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	192	: linearConvolutionMinKernel_Texture<	192	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	193	: linearConvolutionMinKernel_Texture<	193	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	194	: linearConvolutionMinKernel_Texture<	194	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	195	: linearConvolutionMinKernel_Texture<	195	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	196	: linearConvolutionMinKernel_Texture<	196	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	197	: linearConvolutionMinKernel_Texture<	197	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	198	: linearConvolutionMinKernel_Texture<	198	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	199	: linearConvolutionMinKernel_Texture<	199	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	200	: linearConvolutionMinKernel_Texture<	200	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	201	: linearConvolutionMinKernel_Texture<	201	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	202	: linearConvolutionMinKernel_Texture<	202	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	203	: linearConvolutionMinKernel_Texture<	203	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	204	: linearConvolutionMinKernel_Texture<	204	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	205	: linearConvolutionMinKernel_Texture<	205	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	206	: linearConvolutionMinKernel_Texture<	206	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	207	: linearConvolutionMinKernel_Texture<	207	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	208	: linearConvolutionMinKernel_Texture<	208	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	209	: linearConvolutionMinKernel_Texture<	209	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	210	: linearConvolutionMinKernel_Texture<	210	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	211	: linearConvolutionMinKernel_Texture<	211	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	212	: linearConvolutionMinKernel_Texture<	212	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	213	: linearConvolutionMinKernel_Texture<	213	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	214	: linearConvolutionMinKernel_Texture<	214	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	215	: linearConvolutionMinKernel_Texture<	215	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	216	: linearConvolutionMinKernel_Texture<	216	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	217	: linearConvolutionMinKernel_Texture<	217	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	218	: linearConvolutionMinKernel_Texture<	218	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	219	: linearConvolutionMinKernel_Texture<	219	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	220	: linearConvolutionMinKernel_Texture<	220	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	221	: linearConvolutionMinKernel_Texture<	221	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	222	: linearConvolutionMinKernel_Texture<	222	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	223	: linearConvolutionMinKernel_Texture<	223	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	224	: linearConvolutionMinKernel_Texture<	224	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	225	: linearConvolutionMinKernel_Texture<	225	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	226	: linearConvolutionMinKernel_Texture<	226	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	227	: linearConvolutionMinKernel_Texture<	227	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	228	: linearConvolutionMinKernel_Texture<	228	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	229	: linearConvolutionMinKernel_Texture<	229	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	230	: linearConvolutionMinKernel_Texture<	230	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	231	: linearConvolutionMinKernel_Texture<	231	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	232	: linearConvolutionMinKernel_Texture<	232	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	233	: linearConvolutionMinKernel_Texture<	233	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	234	: linearConvolutionMinKernel_Texture<	234	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	235	: linearConvolutionMinKernel_Texture<	235	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	236	: linearConvolutionMinKernel_Texture<	236	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	237	: linearConvolutionMinKernel_Texture<	237	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	238	: linearConvolutionMinKernel_Texture<	238	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	239	: linearConvolutionMinKernel_Texture<	239	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	240	: linearConvolutionMinKernel_Texture<	240	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	241	: linearConvolutionMinKernel_Texture<	241	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	242	: linearConvolutionMinKernel_Texture<	242	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	243	: linearConvolutionMinKernel_Texture<	243	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	244	: linearConvolutionMinKernel_Texture<	244	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	245	: linearConvolutionMinKernel_Texture<	245	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	246	: linearConvolutionMinKernel_Texture<	246	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	247	: linearConvolutionMinKernel_Texture<	247	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	248	: linearConvolutionMinKernel_Texture<	248	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	249	: linearConvolutionMinKernel_Texture<	249	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	250	: linearConvolutionMinKernel_Texture<	250	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	251	: linearConvolutionMinKernel_Texture<	251	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	252	: linearConvolutionMinKernel_Texture<	252	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	253	: linearConvolutionMinKernel_Texture<	253	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	254	: linearConvolutionMinKernel_Texture<	254	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	255	: linearConvolutionMinKernel_Texture<	255	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
		case	256	: linearConvolutionMinKernel_Texture<	256	><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi); break;
	}
	if(_kernel_length > 256){
		if(_kernel_length <= 512)linearConvolutionMinKernel_Texture<512><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi);
		else if(_kernel_length <= 1024)linearConvolutionMinKernel_Texture<1024><<<blocks, threads>>>(d_Dst, imageW, imageW, imageW*imageH, d_ArgMin_lati, lati, d_ArgMin_longi, longi);
	}
	
    getLastCudaError("linearConvolutionGPU_Texture() execution failed\n");

}

extern "C" __declspec (dllexport) void lineSegmentConvolutionMinGPU(
    float *d_Dst,
    int imageW,
    int imageH,
	int imageZ,
	int blocksize,
	float *h_kernel,
	float *d_ArgMin_lati,
	float lati,
	float *d_ArgMin_longi,
	float longi
){
	float *plist;
	float dx, dy, dz;
	
	plist = (float *)malloc(sizeof(float)*(blocksize*2 + 1)*4);

	dx = cos(longi) * cos(lati);
	dy = sin(longi) * cos(lati);
	dz = sin(lati);
	for(int i = 0; i < blocksize*2 + 1; i++){
		plist[i*4]       = dx*(i-blocksize);
		plist[i*4 + 1]   = dy*(i-blocksize);
		plist[i*4 + 2]   = dz*(i-blocksize);
		plist[i*4 + 3]   = h_kernel[i];
	}
	
	setPointList_LCT(plist, blocksize*2 + 1);
	checkCudaErrors( cudaDeviceSynchronize() );
	linearConvolutionMinGPU_Texture(d_Dst, imageW, imageH, imageZ, d_ArgMin_lati, lati, d_ArgMin_longi, longi);
	
	free(plist);
}

extern "C" __declspec (dllexport) void CircleConvolutionMinGPU(
    float *d_Dst,
    int imageW,
    int imageH,
	int imageZ,
	int radius,
	float *h_kernel,
	float *d_ArgMin_lati,
	float lati,
	float *d_ArgMin_longi,
	float longi
){
	
	float *list;
	list = (float *)malloc(sizeof(float)*(radius*2 + 1)*(radius*2 + 1)*4);
	
	const double rot[3][3] = {{ cos(longi)*cos(lati),  sin(longi), cos(longi)*sin(lati)  },
							  { -sin(longi)*cos(lati), cos(longi), -sin(longi)*sin(lati) },
							  { sin(lati),			   0,		    cos(lati)			 }};
	
	int count = 0;
	double k_sum = 0.0;
	for(int z = -radius; z <= radius; z++){
		for(int y = -radius; y <= radius; y++){
			if( z*z + y*y <= radius*radius){
				double tmpx, tmpy, tmpz;

				tmpx = 0.0;
				tmpy = y;
				tmpz = z;
				
				list[count*4]     = rot[0][0]*tmpx + rot[0][1]*tmpy + rot[0][2]*tmpz;
				list[count*4 + 1] = rot[1][0]*tmpx + rot[1][1]*tmpy + rot[1][2]*tmpz;
				list[count*4 + 2] = rot[2][0]*tmpx + rot[2][1]*tmpy + rot[2][2]*tmpz;
				list[count*4 + 3] = h_kernel[z + radius] * h_kernel[y + radius];
				k_sum += list[count*4 + 3];
				count++;
			}
		}
	}
	for(int i = 0; i < count; i++)list[i*4 + 3] /= k_sum;

	setPointList_LCT(list, (radius*2 + 1)*(radius*2 + 1));
	checkCudaErrors( cudaDeviceSynchronize() );
	linearConvolutionMinGPU_Texture(d_Dst, imageW, imageH, imageZ, d_ArgMin_lati, lati, d_ArgMin_longi, longi);
	
	free(list);
}

__global__ void calcCorrectionTerms_DSADTH_Kernel(
    float *d_Dst,
	float *d_ArgMin_lati,
	float constC_XY,
	float constC_Z,
	int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW)
		d_Dst[id] = -1.0f * (float)(constC_XY*(1.0f - fabsf(d_ArgMin_lati[id]*2.0f/F_PI)) + constC_Z * fabsf(d_ArgMin_lati[id]*2.0f/F_PI));
}

extern "C" __declspec (dllexport) void calcCorrectionTerms_DSADTH_GPU(
    float *d_Dst,
	float *d_ArgMin_lati,
	float constC_XY,
	float constC_Z,
	int imageW,
    int imageH,
	int imageZ
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    calcCorrectionTerms_DSADTH_Kernel<<<blocks, threads>>>(
        d_Dst,
		d_ArgMin_lati,
		constC_XY,
		constC_Z,
		imageW,
		imageW,
		imageW * imageH
    );
    getLastCudaError("calcCorrectionTerms_DSADTH_Kernel() execution failed\n");
}

__global__ void countUpN_LessThan_Kernel(
    float *d_Dst,
	const float *d_Src,
	const float *d_T,
	float *d_N,
	int imageW,
	int pitchY,
	int pitchZ
){
	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * BINALIZE_BLOCKDIM + threadIdx.x < imageW)if(d_Src[id] > d_T[id]){ d_N[id] += 1.0f; d_Dst[id] += d_T[id]; }
}

extern "C" __declspec (dllexport) void countUpN_LessThan_GPU(
    float *d_Dst,
	const float *d_Src,
	const float *d_T,
	float *d_N,
	int imageW,
    int imageH,
	int imageZ
){

    dim3 blocks(imageW/BINALIZE_BLOCKDIM + 1, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);

    countUpN_LessThan_Kernel<<<blocks, threads>>>(
		d_Dst,
		d_Src,
		d_T,
		d_N,
		imageW,
		imageW,
		imageW * imageH
    );
    getLastCudaError("countUpN_LessThan_Kernel() execution failed\n");
}