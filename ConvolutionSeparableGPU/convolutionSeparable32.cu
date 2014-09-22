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
 

#include <assert.h>
#define DLL_CONVOLUTIONSEPARABLE
#include "convolutionSeparable_common.h"
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization



#define KERNEL_RADIUS 32
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define   ROWS_BLOCKDIM_X 32
#define   ROWS_BLOCKDIM_Y 8
#define   ROWS_BLOCKDIM_Z 1
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_BLOCKDIM_Z 1
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 2

#define   Z_COLUMNS_BLOCKDIM_X 8
#define   Z_COLUMNS_BLOCKDIM_Y 4
#define   Z_COLUMNS_BLOCKDIM_Z 8
#define Z_COLUMNS_RESULT_STEPS 4
#define   Z_COLUMNS_HALO_STEPS 4


////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel32[KERNEL_LENGTH];

extern "C" __declspec (dllexport) void setConvolutionKernel32(float *h_Kernel, int kernel_radius){
	float kernel[KERNEL_LENGTH];

	assert( kernel_radius <= KERNEL_RADIUS );

	for(int i = 0; i < KERNEL_LENGTH; i++){
		kernel[i] = ((i >= KERNEL_RADIUS - kernel_radius) && (KERNEL_LENGTH - KERNEL_RADIUS + kernel_radius > i)) ? h_Kernel[i - KERNEL_RADIUS + kernel_radius] : 0;
	}
	cudaMemcpyToSymbol(c_Kernel32, kernel, KERNEL_LENGTH * sizeof(float));
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////


__global__ void convolutionRowsKernel32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ
){
    __shared__ float s_Data[ROWS_BLOCKDIM_Z][ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = blockIdx.z * ROWS_BLOCKDIM_Z + threadIdx.z;

	if((baseY < imageH) && (baseZ < imageZ)){
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

		//Load main data
		#pragma unroll
		for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : d_Src[imageW - baseX - 1];

		//Load left halo
		#pragma unroll
		for(int i = 0; i < ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : *(d_Src - baseX);

		//Load right halo
		#pragma unroll
		for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : d_Src[imageW - baseX - 1];
	}

    //Compute and store results
	__syncthreads();
	if((baseY < imageH) && (baseZ < imageZ)){
		#pragma unroll
		for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
			float sum = 0;

			#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				sum += c_Kernel32[KERNEL_RADIUS - j] * s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];

			if(imageW - baseX > i * ROWS_BLOCKDIM_X)d_Dst[i * ROWS_BLOCKDIM_X] = sum;
		}
	}
}

extern "C" __declspec (dllexport) void convolutionRowsGPU32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
){
    assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    //assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    //assert( imageH % ROWS_BLOCKDIM_Y == 0 );
	//assert( imageZ % ROWS_BLOCKDIM_Z == 0 );

	int x_blocknum = (imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0) ? imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) : imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) + 1;
	int y_blocknum = (imageH % ROWS_BLOCKDIM_Y == 0) ? imageH / ROWS_BLOCKDIM_Y : imageH / ROWS_BLOCKDIM_Y + 1;

    dim3 blocks(x_blocknum, y_blocknum, imageZ / ROWS_BLOCKDIM_Z);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, ROWS_BLOCKDIM_Z);

    convolutionRowsKernel32<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnsKernel32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ
){
    __shared__ float s_Data[COLUMNS_BLOCKDIM_Z][COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
	const int baseZ = blockIdx.z * COLUMNS_BLOCKDIM_Z + threadIdx.z;
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    
	if((baseX < imageW) && (baseZ < imageZ)){
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

		//Main data
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : d_Src[(imageH - baseY - 1) * pitchY];

		//Upper halo
		#pragma unroll
		for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : *(d_Src - baseY * pitchY);

		//Lower halo
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : d_Src[(imageH - baseY - 1) * pitchY];
	}

    //Compute and store results
    __syncthreads();
	if((baseX < imageW) && (baseZ < imageZ)){
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
			float sum = 0;
			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				sum += c_Kernel32[KERNEL_RADIUS - j] * s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];

			if(imageH - baseY > i * COLUMNS_BLOCKDIM_Y)d_Dst[i * COLUMNS_BLOCKDIM_Y * pitchY] = sum;
		}
	}
}

extern "C" __declspec (dllexport) void convolutionColumnsGPU32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
){
    assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
//  assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
//  assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );
//	assert( imageZ % COLUMNS_BLOCKDIM_Z == 0 );

	int x_blocknum = (imageW % COLUMNS_BLOCKDIM_X == 0) ? imageW / COLUMNS_BLOCKDIM_X : imageW / COLUMNS_BLOCKDIM_X + 1;
	int y_blocknum = (imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0) ? imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) : imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) + 1;

    dim3 blocks(x_blocknum, y_blocknum, imageZ / COLUMNS_BLOCKDIM_Z);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, COLUMNS_BLOCKDIM_Z);

    convolutionColumnsKernel32<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Z Column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionZColumnsKernel32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ
){
    __shared__ float s_Data[Z_COLUMNS_BLOCKDIM_X][Z_COLUMNS_BLOCKDIM_Y][(Z_COLUMNS_RESULT_STEPS + 2 * Z_COLUMNS_HALO_STEPS) * Z_COLUMNS_BLOCKDIM_Z + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * Z_COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * Z_COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = (blockIdx.z * Z_COLUMNS_RESULT_STEPS - Z_COLUMNS_HALO_STEPS) * Z_COLUMNS_BLOCKDIM_Z + threadIdx.z;
    
	if((baseX < imageW) && (baseY < imageH)){
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

		//Main data
		#pragma unroll
		for(int i = Z_COLUMNS_HALO_STEPS; i < Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS; i++)
			s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z] = (imageZ - baseZ > i * Z_COLUMNS_BLOCKDIM_Z) ? d_Src[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] : *(d_Src + (imageZ - baseZ - 1) * pitchZ);

		//Upper halo
		#pragma unroll
		for(int i = 0; i < Z_COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z] = (baseZ >= -i * Z_COLUMNS_BLOCKDIM_Z) ? d_Src[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] : *(d_Src - baseZ * pitchZ);

		//Lower halo
		#pragma unroll
		for(int i = Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS; i < Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS + Z_COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z]= (imageZ - baseZ > i * Z_COLUMNS_BLOCKDIM_Z) ? d_Src[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] : *(d_Src + (imageZ - baseZ - 1) * pitchZ);
	}

    //Compute and store results
    __syncthreads();
	if((baseX < imageW) && (baseY < imageH)){
		#pragma unroll
		for(int i = Z_COLUMNS_HALO_STEPS; i < Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS; i++){
			float sum = 0;
			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				sum += c_Kernel32[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z + j];

			if(imageZ - baseZ > i * Z_COLUMNS_BLOCKDIM_Z)d_Dst[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] = sum;
		}
	}
}

extern "C" __declspec (dllexport) void convolutionZColumnsGPU32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
){
    assert( Z_COLUMNS_BLOCKDIM_Z * Z_COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
    //assert( imageW % Z_COLUMNS_BLOCKDIM_X == 0 );
    //assert( imageH % Z_COLUMNS_BLOCKDIM_Y == 0 );
	//assert( imageZ % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0 );
	
	int x_blocknum = (imageW % Z_COLUMNS_BLOCKDIM_X == 0) ? imageW / Z_COLUMNS_BLOCKDIM_X : imageW / Z_COLUMNS_BLOCKDIM_X + 1;
	int y_blocknum = (imageH % Z_COLUMNS_BLOCKDIM_Y == 0) ? imageH / Z_COLUMNS_BLOCKDIM_Y : imageH / Z_COLUMNS_BLOCKDIM_Y + 1;
	int z_blocknum = (imageZ % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0) ? imageZ / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) : imageZ / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) + 1;
	
	dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
    dim3 threads(Z_COLUMNS_BLOCKDIM_X, Z_COLUMNS_BLOCKDIM_Y, Z_COLUMNS_BLOCKDIM_Z);

    convolutionZColumnsKernel32<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH
    );
    getLastCudaError("convolutionZColumnsKernel() execution failed\n");
}

