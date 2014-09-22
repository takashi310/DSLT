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


#define KERNEL_RADIUS 1
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 16
#define   ROWS_BLOCKDIM_Z 1
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_BLOCKDIM_Z 1
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

#define   Z_COLUMNS_BLOCKDIM_X 8
#define   Z_COLUMNS_BLOCKDIM_Y 4
#define   Z_COLUMNS_BLOCKDIM_Z 8
#define Z_COLUMNS_RESULT_STEPS 2
#define   Z_COLUMNS_HALO_STEPS 1

#define MASK_BLOCKDIM 32

/*
////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel8[KERNEL_LENGTH];

extern "C" __declspec (dllexport) void setConvolutionKernel8(float *h_Kernel, int kernel_radius){
	float kernel[KERNEL_LENGTH];

	assert( kernel_radius <= KERNEL_RADIUS );

	for(int i = 0; i < KERNEL_LENGTH; i++){
		kernel[i] = ((i >= KERNEL_RADIUS - kernel_radius) && (KERNEL_LENGTH - KERNEL_RADIUS + kernel_radius > i)) ? h_Kernel[i - KERNEL_RADIUS + kernel_radius] : 0;
	}
    cudaMemcpyToSymbol(c_Kernel8, kernel, KERNEL_LENGTH * sizeof(float));
}
*/

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void maximumFilterRowsKernel1(
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
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW > baseX + i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0.0f;

		//Load left halo
		#pragma unroll
		for(int i = 0; i < ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0 ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0.0f;

		//Load right halo
		#pragma unroll
		for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW > baseX + i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0.0f;
	}

    //Compute and store results
    __syncthreads();
	if((baseY < imageH) && (baseZ < imageZ)){
	    #pragma unroll
	    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
	        float fmax = 0.0f;
	
	        #pragma unroll
	        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
	            fmax = (fmax < s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j]) ? s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j] : fmax;
	
	        if(imageW > baseX + i * ROWS_BLOCKDIM_X)d_Dst[i * ROWS_BLOCKDIM_X] = fmax;
	    }
	}
}

extern "C" __declspec (dllexport) void maximumFilterRows1GPU(
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

    dim3 blocks(x_blocknum, y_blocknum, imageZ);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1);

    maximumFilterRowsKernel1<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH
    );
    getLastCudaError("maximumFilterRowsKernel1() execution failed\n");
}

__global__ void maximumFilterRowsKernel1_vHGW(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ
){
    __shared__ float s_Data[2][KERNEL_LENGTH];
	
    //Offset to the left halo edge
    const int baseX = blockIdx.x * KERNEL_LENGTH;
    const int baseY = blockIdx.y;
	const int baseZ = blockIdx.z;

    d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
    d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

    //Load data
	if(threadIdx.x == 0){
		//left (reverse)
		#pragma unroll
		for(int i = 0; i < KERNEL_LENGTH; i++)
			s_Data[threadIdx.x][i] = (baseX + KERNEL_RADIUS - i >= 0) ? *(d_Src + KERNEL_RADIUS - i) : 0.0f;
	}
	else{
		//right
		#pragma unroll
		for(int i = 0; i < KERNEL_LENGTH; i++)
			s_Data[threadIdx.x][i] = (baseX + KERNEL_RADIUS + i < imageW) ? d_Src[KERNEL_RADIUS + i] : 0.0f;
	}
	
	#pragma unroll
	for(int i = 1; i < KERNEL_LENGTH; i++)
		s_Data[threadIdx.x][i] = (s_Data[threadIdx.x][i] > s_Data[threadIdx.x][i - 1]) ? s_Data[threadIdx.x][i] : s_Data[threadIdx.x][i - 1];
		
    //Compute and store results
    __syncthreads();
    
	if(threadIdx.x == 0){
		#pragma unroll
		for(int i = 0; i < KERNEL_LENGTH; i += 2)
		    d_Dst[i] = (s_Data[0][KERNEL_LENGTH - 1 - i] > s_Data[1][i]) ? s_Data[0][KERNEL_LENGTH - 1 - i] : s_Data[1][i];
	}
	else{
		#pragma unroll
		for(int i = 1; i < KERNEL_LENGTH; i += 2)
		    d_Dst[i] = (s_Data[0][KERNEL_LENGTH - 1 - i] > s_Data[1][i]) ? s_Data[0][KERNEL_LENGTH - 1 - i] : s_Data[1][i];
	}
   
}

extern "C" __declspec (dllexport) void maximumFilterRows1GPU_vHGW(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
){
    /*assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert( imageH % ROWS_BLOCKDIM_Y == 0 );
	assert( imageZ % ROWS_BLOCKDIM_Z == 0 );
	*/
    dim3 blocks(imageW / KERNEL_LENGTH + 1, imageH, imageZ);
    dim3 threads(2);

    maximumFilterRowsKernel1_vHGW<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH
    );
    getLastCudaError("maximumFilterRowsKernel1() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void maximumFilterColumnsKernel1(
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

	if((baseX < imageW) && (baseZ < imageZ))
	{
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;
		//Main data
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : 0.0f;

		//Upper halo
		#pragma unroll
		for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : 0.0f;

		//Lower halo
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : 0.0f;
	}

	//Compute and store results
	__syncthreads();
	if((baseX < imageW) && (baseZ < imageZ)){
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
			float fmax = 0.0f;

			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				fmax = (fmax < s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j]) ? s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j] : fmax;

			if(imageH - baseY > i * COLUMNS_BLOCKDIM_Y)d_Dst[i * COLUMNS_BLOCKDIM_Y * pitchY] = fmax;
		}
	}
}

extern "C" __declspec (dllexport) void maximumFilterColumns1GPU(
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

    dim3 blocks(x_blocknum, y_blocknum, imageZ);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1);

    maximumFilterColumnsKernel1<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH
    );
    getLastCudaError("maximumFilterColumns1GPU execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Z Column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void maximumFilterZColumnsKernel1(
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
    d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
    d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	if((baseX < imageW) && (baseY < imageH)){ 
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
			float fmax = 0.0f;
			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				fmax = (fmax < s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z + j]) ? s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z + j] : fmax;

			if(imageZ - baseZ > i * Z_COLUMNS_BLOCKDIM_Z)d_Dst[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] = fmax;
		}
	}
}

extern "C" __declspec (dllexport) void maximumFilterZColumns1GPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
){
    assert( Z_COLUMNS_BLOCKDIM_Z * Z_COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
//	assert( imageW % Z_COLUMNS_BLOCKDIM_X == 0 );
//	assert( imageH % Z_COLUMNS_BLOCKDIM_Y == 0 );
//	assert( imageZ % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0 );
	
	int x_blocknum = (imageW % Z_COLUMNS_BLOCKDIM_X == 0) ? imageW / Z_COLUMNS_BLOCKDIM_X : imageW / Z_COLUMNS_BLOCKDIM_X + 1;
	int y_blocknum = (imageH % Z_COLUMNS_BLOCKDIM_Y == 0) ? imageH / Z_COLUMNS_BLOCKDIM_Y : imageH / Z_COLUMNS_BLOCKDIM_Y + 1;
	int z_blocknum = (imageZ % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0) ? imageZ / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) : imageZ / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) + 1;
	
	dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
    dim3 threads(Z_COLUMNS_BLOCKDIM_X, Z_COLUMNS_BLOCKDIM_Y, Z_COLUMNS_BLOCKDIM_Z);

    maximumFilterZColumnsKernel1<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH
    );
    getLastCudaError("maximumFilterZColumnsKernel1() execution failed\n");
}


__global__ void maximumFilterRowsKernel1ROI(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_xx,
	int roi_yy,
	int roi_zz
){
    __shared__ float s_Data[ROWS_BLOCKDIM_Z][ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = roi_x + (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + blockIdx.z * ROWS_BLOCKDIM_Z + threadIdx.z;

	if((baseY < imageH) && (baseZ < imageZ)){
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

		//Load main data
		#pragma unroll
		for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW > baseX + i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0.0f;

		//Load left halo
		#pragma unroll
		for(int i = 0; i < ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0 ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0.0f;

		//Load right halo
		#pragma unroll
		for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW > baseX + i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0.0f;
	}

    //Compute and store results
    __syncthreads();
	if((baseY < roi_yy) && (baseZ < roi_zz)){
	    #pragma unroll
	    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
	        float fmax = 0.0f;
	
	        #pragma unroll
	        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
	            fmax = (fmax < s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j]) ? s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j] : fmax;
	
			if(roi_xx > baseX + i * ROWS_BLOCKDIM_X)d_Dst[i * ROWS_BLOCKDIM_X] = fmax;
	    }
	}
}


extern "C" __declspec (dllexport) void maximumFilterRows1ROIGPU(
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
    assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    //assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    //assert( imageH % ROWS_BLOCKDIM_Y == 0 );
	//assert( imageZ % ROWS_BLOCKDIM_Z == 0 );

	int x_blocknum = (roi_w % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0) ? roi_w / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) : roi_w / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) + 1;
	int y_blocknum = (roi_h % ROWS_BLOCKDIM_Y == 0) ? roi_h / ROWS_BLOCKDIM_Y : roi_h / ROWS_BLOCKDIM_Y + 1;

    dim3 blocks(x_blocknum, y_blocknum, roi_d);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1);

    maximumFilterRowsKernel1ROI<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_x + roi_w,
		roi_y + roi_h,
		roi_z + roi_d
    );
    getLastCudaError("maximumFilterRowsKernel1ROI() execution failed\n");
}

__global__ void maximumFilterColumnsKernel1ROI(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_xx,
	int roi_yy,
	int roi_zz
){
    __shared__ float s_Data[COLUMNS_BLOCKDIM_Z][COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
	const int baseZ = roi_z + blockIdx.z * COLUMNS_BLOCKDIM_Z + threadIdx.z;
    const int baseX = roi_x + blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

	if((baseX < imageW) && (baseZ < imageZ))
	{
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;
		//Main data
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : 0.0f;

		//Upper halo
		#pragma unroll
		for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : 0.0f;

		//Lower halo
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : 0.0f;
	}

	//Compute and store results
	__syncthreads();
	if((baseX < roi_xx) && (baseZ < roi_zz)){
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
			float fmax = 0.0f;

			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				fmax = (fmax < s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j]) ? s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j] : fmax;

			if(roi_yy - baseY > i * COLUMNS_BLOCKDIM_Y)d_Dst[i * COLUMNS_BLOCKDIM_Y * pitchY] = fmax;
		}
	}
}

extern "C" __declspec (dllexport) void maximumFilterColumns1ROIGPU(
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
    assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
//  assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
//  assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );
//	assert( imageZ % COLUMNS_BLOCKDIM_Z == 0 );

    int x_blocknum = (roi_w % COLUMNS_BLOCKDIM_X == 0) ? roi_w / COLUMNS_BLOCKDIM_X : roi_w / COLUMNS_BLOCKDIM_X + 1;
	int y_blocknum = (roi_h % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0) ? roi_h / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) : roi_h / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) + 1;

    dim3 blocks(x_blocknum, y_blocknum, roi_d);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1);

    maximumFilterColumnsKernel1ROI<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_x + roi_w,
		roi_y + roi_h,
		roi_z + roi_d
    );
    getLastCudaError("maximumFilterColumns1ROIGPU execution failed\n");
}


__global__ void maximumFilterZColumnsKernel1ROI(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_xx,
	int roi_yy,
	int roi_zz
){
    __shared__ float s_Data[Z_COLUMNS_BLOCKDIM_X][Z_COLUMNS_BLOCKDIM_Y][(Z_COLUMNS_RESULT_STEPS + 2 * Z_COLUMNS_HALO_STEPS) * Z_COLUMNS_BLOCKDIM_Z + 1];

    //Offset to the upper halo edge
    const int baseX = roi_x + blockIdx.x * Z_COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = roi_y + blockIdx.y * Z_COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = roi_z + (blockIdx.z * Z_COLUMNS_RESULT_STEPS - Z_COLUMNS_HALO_STEPS) * Z_COLUMNS_BLOCKDIM_Z + threadIdx.z;
    d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
    d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	if((baseX < imageW) && (baseY < imageH)){ 
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
	if((baseX < roi_xx) && (baseY < roi_yy)){ 
		#pragma unroll
		for(int i = Z_COLUMNS_HALO_STEPS; i < Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS; i++){
			float fmax = 0.0f;
			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				fmax = (fmax < s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z + j]) ? s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z + j] : fmax;

			if(roi_zz - baseZ > i * Z_COLUMNS_BLOCKDIM_Z)d_Dst[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] = fmax;
		}
	}
}

extern "C" __declspec (dllexport) void maximumFilterZColumns1ROIGPU(
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
    assert( Z_COLUMNS_BLOCKDIM_Z * Z_COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
//	assert( imageW % Z_COLUMNS_BLOCKDIM_X == 0 );
//	assert( imageH % Z_COLUMNS_BLOCKDIM_Y == 0 );
//	assert( imageZ % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0 );
	
	int x_blocknum = (roi_w % Z_COLUMNS_BLOCKDIM_X == 0) ? roi_w / Z_COLUMNS_BLOCKDIM_X : roi_w / Z_COLUMNS_BLOCKDIM_X + 1;
	int y_blocknum = (roi_h % Z_COLUMNS_BLOCKDIM_Y == 0) ? roi_h / Z_COLUMNS_BLOCKDIM_Y : roi_h / Z_COLUMNS_BLOCKDIM_Y + 1;
	int z_blocknum = (roi_d % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0) ? roi_d / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) : roi_d / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) + 1;
	
	dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
    dim3 threads(Z_COLUMNS_BLOCKDIM_X, Z_COLUMNS_BLOCKDIM_Y, Z_COLUMNS_BLOCKDIM_Z);

    maximumFilterZColumnsKernel1ROI<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_x + roi_w,
		roi_y + roi_h,
		roi_z + roi_d
    );
    getLastCudaError("maximumFilterZColumnsKernel1ROIGPU() execution failed\n");
}


__global__ void minimumFilterRowsKernel1ROI(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_xx,
	int roi_yy,
	int roi_zz
){
    __shared__ float s_Data[ROWS_BLOCKDIM_Z][ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = roi_x + (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + blockIdx.z * ROWS_BLOCKDIM_Z + threadIdx.z;

	if((baseY < imageH) && (baseZ < imageZ)){
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

		//Load main data
		#pragma unroll
		for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW > baseX + i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : FLT_MAX;

		//Load left halo
		#pragma unroll
		for(int i = 0; i < ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0 ) ? d_Src[i * ROWS_BLOCKDIM_X] : FLT_MAX;

		//Load right halo
		#pragma unroll
		for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW > baseX + i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : FLT_MAX;
	}

    //Compute and store results
    __syncthreads();
	if((baseY < roi_yy) && (baseZ < roi_zz)){
	    #pragma unroll
	    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
	        float fmin = FLT_MAX;
	
	        #pragma unroll
	        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
	            fmin = (fmin > s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j]) ? s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j] : fmin;
	
			if(roi_xx > baseX + i * ROWS_BLOCKDIM_X)d_Dst[i * ROWS_BLOCKDIM_X] = fmin;
	    }
	}
}


extern "C" __declspec (dllexport) void minimumFilterRows1ROIGPU(
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
    assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    //assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    //assert( imageH % ROWS_BLOCKDIM_Y == 0 );
	//assert( imageZ % ROWS_BLOCKDIM_Z == 0 );

	int x_blocknum = (roi_w % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0) ? roi_w / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) : roi_w / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) + 1;
	int y_blocknum = (roi_h % ROWS_BLOCKDIM_Y == 0) ? roi_h / ROWS_BLOCKDIM_Y : roi_h / ROWS_BLOCKDIM_Y + 1;

    dim3 blocks(x_blocknum, y_blocknum, roi_d);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1);

    minimumFilterRowsKernel1ROI<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_x + roi_w,
		roi_y + roi_h,
		roi_z + roi_d
    );
    getLastCudaError("minimumFilterRowsKernel1ROI() execution failed\n");
}

__global__ void minimumFilterColumnsKernel1ROI(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_xx,
	int roi_yy,
	int roi_zz
){
    __shared__ float s_Data[COLUMNS_BLOCKDIM_Z][COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
	const int baseZ = roi_z + blockIdx.z * COLUMNS_BLOCKDIM_Z + threadIdx.z;
    const int baseX = roi_x + blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

	if((baseX < imageW) && (baseZ < imageZ))
	{
		d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
		d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;
		//Main data
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : FLT_MAX;

		//Upper halo
		#pragma unroll
		for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : FLT_MAX;

		//Lower halo
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchY] : FLT_MAX;
	}

	//Compute and store results
	__syncthreads();
	if((baseX < roi_xx) && (baseZ < roi_zz)){
		#pragma unroll
		for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
			float fmin = FLT_MAX;

			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				fmin = (fmin > s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j]) ? s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j] : fmin;

			if(roi_yy - baseY > i * COLUMNS_BLOCKDIM_Y)d_Dst[i * COLUMNS_BLOCKDIM_Y * pitchY] = fmin;
		}
	}
}

extern "C" __declspec (dllexport) void minimumFilterColumns1ROIGPU(
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
    assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
//  assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
//  assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );
//	assert( imageZ % COLUMNS_BLOCKDIM_Z == 0 );

    int x_blocknum = (roi_w % COLUMNS_BLOCKDIM_X == 0) ? roi_w / COLUMNS_BLOCKDIM_X : roi_w / COLUMNS_BLOCKDIM_X + 1;
	int y_blocknum = (roi_h % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0) ? roi_h / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) : roi_h / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) + 1;

    dim3 blocks(x_blocknum, y_blocknum, roi_d);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1);

    minimumFilterColumnsKernel1ROI<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_x + roi_w,
		roi_y + roi_h,
		roi_z + roi_d
    );
    getLastCudaError("minimumFilterColumns1ROIGPU execution failed\n");
}


__global__ void minimumFilterZColumnsKernel1ROI(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ,
    int pitchY,
	int pitchZ,
	int roi_x,
	int roi_y,
	int roi_z,
	int roi_xx,
	int roi_yy,
	int roi_zz
){
    __shared__ float s_Data[Z_COLUMNS_BLOCKDIM_X][Z_COLUMNS_BLOCKDIM_Y][(Z_COLUMNS_RESULT_STEPS + 2 * Z_COLUMNS_HALO_STEPS) * Z_COLUMNS_BLOCKDIM_Z + 1];

    //Offset to the upper halo edge
    const int baseX = roi_x + blockIdx.x * Z_COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = roi_y + blockIdx.y * Z_COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = roi_z + (blockIdx.z * Z_COLUMNS_RESULT_STEPS - Z_COLUMNS_HALO_STEPS) * Z_COLUMNS_BLOCKDIM_Z + threadIdx.z;
    d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
    d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	if((baseX < imageW) && (baseY < imageH)){ 
		//Main data
		#pragma unroll
		for(int i = Z_COLUMNS_HALO_STEPS; i < Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS; i++)
			s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z] = (imageZ - baseZ > i * Z_COLUMNS_BLOCKDIM_Z) ? d_Src[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] : FLT_MAX;

		//Upper halo
		#pragma unroll
		for(int i = 0; i < Z_COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z] = (baseZ >= -i * Z_COLUMNS_BLOCKDIM_Z) ? d_Src[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] : FLT_MAX;

		//Lower halo
		#pragma unroll
		for(int i = Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS; i < Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS + Z_COLUMNS_HALO_STEPS; i++)
			s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z]= (imageZ - baseZ > i * Z_COLUMNS_BLOCKDIM_Z) ? d_Src[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] : FLT_MAX;
	}

    //Compute and store results
    __syncthreads();
	if((baseX < roi_xx) && (baseY < roi_yy)){ 
		#pragma unroll
		for(int i = Z_COLUMNS_HALO_STEPS; i < Z_COLUMNS_HALO_STEPS + Z_COLUMNS_RESULT_STEPS; i++){
			float fmin = FLT_MAX;
			//#pragma unroll
			for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
				fmin = (fmin > s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z + j]) ? s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * Z_COLUMNS_BLOCKDIM_Z + j] : fmin;

			if(roi_zz - baseZ > i * Z_COLUMNS_BLOCKDIM_Z)d_Dst[i * Z_COLUMNS_BLOCKDIM_Z * pitchZ] = fmin;
		}
	}
}

extern "C" __declspec (dllexport) void minimumFilterZColumns1ROIGPU(
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
    assert( Z_COLUMNS_BLOCKDIM_Z * Z_COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
//	assert( imageW % Z_COLUMNS_BLOCKDIM_X == 0 );
//	assert( imageH % Z_COLUMNS_BLOCKDIM_Y == 0 );
//	assert( imageZ % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0 );
	
	int x_blocknum = (roi_w % Z_COLUMNS_BLOCKDIM_X == 0) ? roi_w / Z_COLUMNS_BLOCKDIM_X : roi_w / Z_COLUMNS_BLOCKDIM_X + 1;
	int y_blocknum = (roi_h % Z_COLUMNS_BLOCKDIM_Y == 0) ? roi_h / Z_COLUMNS_BLOCKDIM_Y : roi_h / Z_COLUMNS_BLOCKDIM_Y + 1;
	int z_blocknum = (roi_d % (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) == 0) ? roi_d / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) : roi_d / (Z_COLUMNS_RESULT_STEPS * Z_COLUMNS_BLOCKDIM_Z) + 1;
	
	dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
    dim3 threads(Z_COLUMNS_BLOCKDIM_X, Z_COLUMNS_BLOCKDIM_Y, Z_COLUMNS_BLOCKDIM_Z);

    minimumFilterZColumnsKernel1ROI<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
		imageZ,
        imageW,
		imageW*imageH,
		roi_x,
		roi_y,
		roi_z,
		roi_x + roi_w,
		roi_y + roi_h,
		roi_z + roi_d
    );
    getLastCudaError("minimumFilterZColumnsKernel1ROIGPU() execution failed\n");
}


__global__ void upperMaskFittingKernel(
    float *d_Dst,
	float *d_Mask,
    float *d_Src,
	float mask_offset,
    int imageW,
    int pitchY,
	int pitchZ
){
//	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x;
//	if(d_Src[id] > d_Mask[id] + mask_offset)d_Dst[id] = d_Mask[id] + mask_offset;
//	else d_Dst[id] = d_Src[id];


	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * MASK_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * MASK_BLOCKDIM + threadIdx.x < imageW){
		if(d_Src[id] > d_Mask[id] + mask_offset)d_Dst[id] = d_Mask[id] + mask_offset;
		else d_Dst[id] = d_Src[id];
	}

}

extern "C" __declspec (dllexport) void upperMaskFittingGPU(
    float *d_Dst,
    float *d_Mask,
	float *d_Src,
	float mask_offset,
    int imageW,
    int imageH,
	int imageZ
){
//	dim3 blocks(imageW, imageH, imageZ);
//  dim3 threads(1);

	dim3 blocks(imageW/MASK_BLOCKDIM + 1, imageH, imageZ);
	dim3 threads(MASK_BLOCKDIM);

	upperMaskFittingKernel<<<blocks, threads>>>(
        d_Dst,
		d_Mask,
        d_Src,
		mask_offset,
        imageW,
        imageW,
		imageH*imageW
    );
	getLastCudaError("maskFittingKernel() execution failed\n");
}

__global__ void lowerMaskFittingKernel(
    float *d_Dst,
	float *d_Mask,
    float *d_Src,
	float mask_offset,
    int imageW,
    int pitchY,
	int pitchZ
){
//	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x;
//	if(d_Src[id] > d_Mask[id] + mask_offset)d_Dst[id] = d_Mask[id] + mask_offset;
//	else d_Dst[id] = d_Src[id];


	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * MASK_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * MASK_BLOCKDIM + threadIdx.x < imageW){
		if(d_Src[id] < d_Mask[id] + mask_offset)d_Dst[id] = d_Mask[id] + mask_offset;
		else d_Dst[id] = d_Src[id];
	}

}

extern "C" __declspec (dllexport) void lowerMaskFittingGPU(
    float *d_Dst,
    float *d_Mask,
	float *d_Src,
	float mask_offset,
    int imageW,
    int imageH,
	int imageZ
){
//	dim3 blocks(imageW, imageH, imageZ);
//  dim3 threads(1);

	dim3 blocks(imageW/MASK_BLOCKDIM + 1, imageH, imageZ);
	dim3 threads(MASK_BLOCKDIM);

	lowerMaskFittingKernel<<<blocks, threads>>>(
        d_Dst,
		d_Mask,
        d_Src,
		mask_offset,
        imageW,
        imageW,
		imageH*imageW
    );
	getLastCudaError("maskFittingKernel() execution failed\n");
}


__global__ void inverseKernel(
    float *d_Dst,
    float *d_Src,
	float maxval,
	float minval,
    int imageW,
    int pitchY,
	int pitchZ
){

	const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * MASK_BLOCKDIM + threadIdx.x;
	if(blockIdx.x * MASK_BLOCKDIM + threadIdx.x < imageW){
		d_Dst[id] = maxval - (d_Src[id] + minval);
	}

}

extern "C" __declspec (dllexport) void inverseGPU(
    float *d_Dst,
	float *d_Src,
	float maxval,
	float minval,
    int imageW,
    int imageH,
	int imageZ
){

	dim3 blocks(imageW/MASK_BLOCKDIM + 1, imageH, imageZ);
	dim3 threads(MASK_BLOCKDIM);

	inverseKernel<<<blocks, threads>>>(
        d_Dst,
		d_Src,
		maxval,
		minval,
        imageW,
        imageW,
		imageH*imageW
    );
	getLastCudaError("inverseKernel() execution failed\n");
}

