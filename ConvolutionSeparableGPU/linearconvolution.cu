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


#define KERNEL_RADIUS 6
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define     K6_BLOCKDIM_X 6
#define     K6_BLOCKDIM_Y 3
#define     K6_BLOCKDIM_Z 2
#define K6_RESULT_STEPS_X 1
#define K6_RESULT_STEPS_Y 1
#define K6_RESULT_STEPS_Z 1
#define   K6_HALO_STEPS_X 1
#define   K6_HALO_STEPS_Y 2
#define   K6_HALO_STEPS_Z 3

#define     K4_BLOCKDIM_X 8
#define     K4_BLOCKDIM_Y 4
#define     K4_BLOCKDIM_Z 4
#define K4_RESULT_STEPS_X 1
#define K4_RESULT_STEPS_Y 1
#define K4_RESULT_STEPS_Z 1
#define   K4_HALO_STEPS_X 1
#define   K4_HALO_STEPS_Y 1
#define   K4_HALO_STEPS_Z 1

#define     K2_BLOCKDIM_X 16
#define     K2_BLOCKDIM_Y 2
#define     K2_BLOCKDIM_Z 2
#define K2_RESULT_STEPS_X 2
#define K2_RESULT_STEPS_Y 1
#define K2_RESULT_STEPS_Z 1
#define   K2_HALO_STEPS_X 1
#define   K2_HALO_STEPS_Y 1
#define   K2_HALO_STEPS_Z 1


////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel6[KERNEL_LENGTH*KERNEL_LENGTH*KERNEL_LENGTH];

int _kernel_radus = 0;

extern "C" __declspec (dllexport) void setSphereMaskKernel6(int radius){
	float kernel[KERNEL_LENGTH*KERNEL_LENGTH*KERNEL_LENGTH];

	assert( radius <= KERNEL_RADIUS );

	_kernel_radus = radius;
	
	for(int z = 0; z < KERNEL_LENGTH; z++){
		for(int y = 0; y < KERNEL_LENGTH; y++){
			for(int x = 0; x < KERNEL_LENGTH; x++){
				if( sqrt( (x - KERNEL_RADIUS)*(x - KERNEL_RADIUS) + (y - KERNEL_RADIUS)*(y - KERNEL_RADIUS) + (z - KERNEL_RADIUS)*(z - KERNEL_RADIUS) ) <= radius )
					kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] = 1.0f;
				else kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] = 0.0f;
				//printf("%.1f ", kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x]);
			}
			//putchar('\n');
		}
		//putchar('\n');
		//putchar('\n');
	}
	/*
	for(int z = 0; z < KERNEL_LENGTH; z++){
		for(int y = 0; y < KERNEL_LENGTH; y++){
			for(int x = 0; x < KERNEL_LENGTH; x++){
				kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] = 0.0f;
			}
		}
	}
	int x,y,z;
*/	//for(x = KERNEL_RADIUS, y = 0, z = KERNEL_LENGTH-1; y < KERNEL_LENGTH; y++,z--)if(x <= KERNEL_RADIUS)kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] = 1.0f;
	//for(z = KERNEL_RADIUS; z < KERNEL_LENGTH; z++)kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + KERNEL_RADIUS*KERNEL_LENGTH + KERNEL_RADIUS] = 1.0f;
	//for(y = 0; y <= KERNEL_RADIUS; y++)kernel[KERNEL_RADIUS*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + KERNEL_RADIUS] = 1.0f;
	//for(x = 0; x <= KERNEL_RADIUS; x++)kernel[KERNEL_RADIUS*KERNEL_LENGTH*KERNEL_LENGTH + KERNEL_RADIUS*KERNEL_LENGTH + x] = 1.0f;
	//for(y = KERNEL_LENGTH - 1; y >= KERNEL_RADIUS; y--)kernel[KERNEL_RADIUS*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + KERNEL_RADIUS] = 1.0f;
    cudaMemcpyToSymbol(c_Kernel6, kernel, KERNEL_LENGTH*KERNEL_LENGTH*KERNEL_LENGTH * sizeof(float));
}

//* d_Src == d_DstÇæÇ∆ê≥ÇµÇ≠ìÆçÏÇµÇ»Ç¢ *//
template<int k_radius> __global__ void maximumSphereFilterKernel6(
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
	__shared__ float s_Data[(K6_RESULT_STEPS_Z + 2 * K6_HALO_STEPS_Z) * K6_BLOCKDIM_Z][(K6_RESULT_STEPS_Y + 2 * K6_HALO_STEPS_Y) * K6_BLOCKDIM_Y][(K6_RESULT_STEPS_X + 2 * K6_HALO_STEPS_X) * K6_BLOCKDIM_X];

    //Offset to the K6_HALO edges
    const int baseX = roi_x + (blockIdx.x * K6_RESULT_STEPS_X - K6_HALO_STEPS_X) * K6_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * K6_RESULT_STEPS_Y - K6_HALO_STEPS_Y) * K6_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + (blockIdx.z * K6_RESULT_STEPS_Z - K6_HALO_STEPS_Z) * K6_BLOCKDIM_Z + threadIdx.z;

	d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
	d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	//Load main data
	#pragma unroll
	for(int i = 0; i < K6_HALO_STEPS_Z + K6_RESULT_STEPS_Z + K6_HALO_STEPS_Z; i++){
		#pragma unroll
		for(int j = 0; j < K6_HALO_STEPS_Y + K6_RESULT_STEPS_Y + K6_HALO_STEPS_Y; j++){
			#pragma unroll
			for(int k = 0; k < K6_HALO_STEPS_X + K6_RESULT_STEPS_X + K6_HALO_STEPS_X; k++){
				if((imageZ - baseZ > i * K6_BLOCKDIM_Z && baseZ + i * K6_BLOCKDIM_Z >= 0) &&
				   (imageH - baseY > j * K6_BLOCKDIM_Y && baseY + j * K6_BLOCKDIM_Y >= 0) &&
				   (imageW - baseX > k * K6_BLOCKDIM_X && baseX + k * K6_BLOCKDIM_X >= 0)   )
				   s_Data[threadIdx.z + i * K6_BLOCKDIM_Z][threadIdx.y + j * K6_BLOCKDIM_Y][threadIdx.x + k * K6_BLOCKDIM_X] = d_Src[i * K6_BLOCKDIM_Z * pitchZ + j * K6_BLOCKDIM_Y * pitchY + k * K6_BLOCKDIM_X];
				else s_Data[threadIdx.z + i * K6_BLOCKDIM_Z][threadIdx.y + j * K6_BLOCKDIM_Y][threadIdx.x + k * K6_BLOCKDIM_X] = 0.0f;
			}
		}
	}

	//Compute and store K6_RESULTs
	__syncthreads();
	
	#pragma unroll
	for(int i = K6_HALO_STEPS_Z; i < K6_HALO_STEPS_Z + K6_RESULT_STEPS_Z; i++){
		#pragma unroll
		for(int j = K6_HALO_STEPS_Y; j < K6_HALO_STEPS_Y + K6_RESULT_STEPS_Y; j++){
			#pragma unroll
			for(int k = K6_HALO_STEPS_X; k < K6_HALO_STEPS_X + K6_RESULT_STEPS_X; k++){
				float max = FLT_MIN;
				
				#pragma unroll
				for(int z = -1 * k_radius; z <= k_radius; z++){
					#pragma unroll
					for(int y = -1 * k_radius; y <= k_radius; y++){
						#pragma unroll
						for(int x = -1 * k_radius; x <= k_radius; x++){
							float val =  c_Kernel6[(KERNEL_RADIUS + z)*KERNEL_LENGTH*KERNEL_LENGTH + (KERNEL_RADIUS + y)*KERNEL_LENGTH + KERNEL_RADIUS + x]
										 * s_Data[threadIdx.z + i * K6_BLOCKDIM_Z + z][threadIdx.y + j * K6_BLOCKDIM_Y + y][threadIdx.x + k * K6_BLOCKDIM_X + x];
							max = (max < val) ? val : max;
						}
					}
				}
				
				//max = s_Data[threadIdx.z + i * K6_BLOCKDIM_Z][threadIdx.y + j * K6_BLOCKDIM_Y - 2][threadIdx.x + k * K6_BLOCKDIM_X];
				if( (roi_xx - baseX > k * K6_BLOCKDIM_X) && 
					(roi_yy - baseY > j * K6_BLOCKDIM_Y) &&
					(roi_zz - baseZ > i * K6_BLOCKDIM_Z)    )d_Dst[i * K6_BLOCKDIM_Z * pitchZ + j * K6_BLOCKDIM_Y * pitchY + k * K6_BLOCKDIM_X] = max;

			}
		}
	}
	

}

template<int k_radius> __global__ void maximumSphereFilterKernel4(
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
	__shared__ float s_Data[(K4_RESULT_STEPS_Z + 2 * K4_HALO_STEPS_Z) * K4_BLOCKDIM_Z][(K4_RESULT_STEPS_Y + 2 * K4_HALO_STEPS_Y) * K4_BLOCKDIM_Y][(K4_RESULT_STEPS_X + 2 * K4_HALO_STEPS_X) * K4_BLOCKDIM_X];

    //Offset to the K4_HALO edges
    const int baseX = roi_x + (blockIdx.x * K4_RESULT_STEPS_X - K4_HALO_STEPS_X) * K4_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * K4_RESULT_STEPS_Y - K4_HALO_STEPS_Y) * K4_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + (blockIdx.z * K4_RESULT_STEPS_Z - K4_HALO_STEPS_Z) * K4_BLOCKDIM_Z + threadIdx.z;

	d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
	d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	//Load main data
	#pragma unroll
	for(int i = 0; i < K4_HALO_STEPS_Z + K4_RESULT_STEPS_Z + K4_HALO_STEPS_Z; i++){
		#pragma unroll
		for(int j = 0; j < K4_HALO_STEPS_Y + K4_RESULT_STEPS_Y + K4_HALO_STEPS_Y; j++){
			#pragma unroll
			for(int k = 0; k < K4_HALO_STEPS_X + K4_RESULT_STEPS_X + K4_HALO_STEPS_X; k++){
				if((imageZ - baseZ > i * K4_BLOCKDIM_Z && baseZ + i * K4_BLOCKDIM_Z >= 0) &&
				   (imageH - baseY > j * K4_BLOCKDIM_Y && baseY + j * K4_BLOCKDIM_Y >= 0) &&
				   (imageW - baseX > k * K4_BLOCKDIM_X && baseX + k * K4_BLOCKDIM_X >= 0)   )
				   s_Data[threadIdx.z + i * K4_BLOCKDIM_Z][threadIdx.y + j * K4_BLOCKDIM_Y][threadIdx.x + k * K4_BLOCKDIM_X] = d_Src[i * K4_BLOCKDIM_Z * pitchZ + j * K4_BLOCKDIM_Y * pitchY + k * K4_BLOCKDIM_X];
				else s_Data[threadIdx.z + i * K4_BLOCKDIM_Z][threadIdx.y + j * K4_BLOCKDIM_Y][threadIdx.x + k * K4_BLOCKDIM_X] = 0.0f;
			}
		}
	}

	//Compute and store K4_RESULTs
	__syncthreads();
	
	#pragma unroll
	for(int i = K4_HALO_STEPS_Z; i < K4_HALO_STEPS_Z + K4_RESULT_STEPS_Z; i++){
		#pragma unroll
		for(int j = K4_HALO_STEPS_Y; j < K4_HALO_STEPS_Y + K4_RESULT_STEPS_Y; j++){
			#pragma unroll
			for(int k = K4_HALO_STEPS_X; k < K4_HALO_STEPS_X + K4_RESULT_STEPS_X; k++){
				float max = FLT_MIN;
				
				#pragma unroll
				for(int z = -1 * k_radius; z <= k_radius; z++){
					#pragma unroll
					for(int y = -1 * k_radius; y <= k_radius; y++){
						#pragma unroll
						for(int x = -1 * k_radius; x <= k_radius; x++){
							float val =  c_Kernel6[(KERNEL_RADIUS + z)*KERNEL_LENGTH*KERNEL_LENGTH + (KERNEL_RADIUS + y)*KERNEL_LENGTH + KERNEL_RADIUS + x]
										 * s_Data[threadIdx.z + i * K4_BLOCKDIM_Z + z][threadIdx.y + j * K4_BLOCKDIM_Y + y][threadIdx.x + k * K4_BLOCKDIM_X + x];
							max = (max < val) ? val : max;
						}
					}
				}
				
				//max = s_Data[threadIdx.z + i * K4_BLOCKDIM_Z][threadIdx.y + j * K4_BLOCKDIM_Y - 2][threadIdx.x + k * K4_BLOCKDIM_X];
				if( (roi_xx - baseX > k * K4_BLOCKDIM_X) && 
					(roi_yy - baseY > j * K4_BLOCKDIM_Y) &&
					(roi_zz - baseZ > i * K4_BLOCKDIM_Z)    )d_Dst[i * K4_BLOCKDIM_Z * pitchZ + j * K4_BLOCKDIM_Y * pitchY + k * K4_BLOCKDIM_X] = max;

			}
		}
	}
	

}

template<int k_radius> __global__ void maximumSphereFilterKernel2(
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
	__shared__ float s_Data[(K2_RESULT_STEPS_Z + 2 * K2_HALO_STEPS_Z) * K2_BLOCKDIM_Z][(K2_RESULT_STEPS_Y + 2 * K2_HALO_STEPS_Y) * K2_BLOCKDIM_Y][(K2_RESULT_STEPS_X + 2 * K2_HALO_STEPS_X) * K2_BLOCKDIM_X];

    //Offset to the K2_HALO edges
    const int baseX = roi_x + (blockIdx.x * K2_RESULT_STEPS_X - K2_HALO_STEPS_X) * K2_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * K2_RESULT_STEPS_Y - K2_HALO_STEPS_Y) * K2_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + (blockIdx.z * K2_RESULT_STEPS_Z - K2_HALO_STEPS_Z) * K2_BLOCKDIM_Z + threadIdx.z;

	d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
	d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	//Load main data
	#pragma unroll
	for(int i = 0; i < K2_HALO_STEPS_Z + K2_RESULT_STEPS_Z + K2_HALO_STEPS_Z; i++){
		#pragma unroll
		for(int j = 0; j < K2_HALO_STEPS_Y + K2_RESULT_STEPS_Y + K2_HALO_STEPS_Y; j++){
			#pragma unroll
			for(int k = 0; k < K2_HALO_STEPS_X + K2_RESULT_STEPS_X + K2_HALO_STEPS_X; k++){
				if((imageZ - baseZ > i * K2_BLOCKDIM_Z && baseZ + i * K2_BLOCKDIM_Z >= 0) &&
				   (imageH - baseY > j * K2_BLOCKDIM_Y && baseY + j * K2_BLOCKDIM_Y >= 0) &&
				   (imageW - baseX > k * K2_BLOCKDIM_X && baseX + k * K2_BLOCKDIM_X >= 0)   )
				   s_Data[threadIdx.z + i * K2_BLOCKDIM_Z][threadIdx.y + j * K2_BLOCKDIM_Y][threadIdx.x + k * K2_BLOCKDIM_X] = d_Src[i * K2_BLOCKDIM_Z * pitchZ + j * K2_BLOCKDIM_Y * pitchY + k * K2_BLOCKDIM_X];
				else s_Data[threadIdx.z + i * K2_BLOCKDIM_Z][threadIdx.y + j * K2_BLOCKDIM_Y][threadIdx.x + k * K2_BLOCKDIM_X] = 0.0f;
			}
		}
	}

	//Compute and store K2_RESULTs
	__syncthreads();
	
	#pragma unroll
	for(int i = K2_HALO_STEPS_Z; i < K2_HALO_STEPS_Z + K2_RESULT_STEPS_Z; i++){
		#pragma unroll
		for(int j = K2_HALO_STEPS_Y; j < K2_HALO_STEPS_Y + K2_RESULT_STEPS_Y; j++){
			#pragma unroll
			for(int k = K2_HALO_STEPS_X; k < K2_HALO_STEPS_X + K2_RESULT_STEPS_X; k++){
				float max = FLT_MIN;
				
				#pragma unroll
				for(int z = -1 * k_radius; z <= k_radius; z++){
					#pragma unroll
					for(int y = -1 * k_radius; y <= k_radius; y++){
						#pragma unroll
						for(int x = -1 * k_radius; x <= k_radius; x++){
							float val =  c_Kernel6[(KERNEL_RADIUS + z)*KERNEL_LENGTH*KERNEL_LENGTH + (KERNEL_RADIUS + y)*KERNEL_LENGTH + KERNEL_RADIUS + x]
										 * s_Data[threadIdx.z + i * K2_BLOCKDIM_Z + z][threadIdx.y + j * K2_BLOCKDIM_Y + y][threadIdx.x + k * K2_BLOCKDIM_X + x];
							max = (max < val) ? val : max;
						}
					}
				}
				
				//max = s_Data[threadIdx.z + i * K2_BLOCKDIM_Z][threadIdx.y + j * K2_BLOCKDIM_Y - 2][threadIdx.x + k * K2_BLOCKDIM_X];
				if( (roi_xx - baseX > k * K2_BLOCKDIM_X) && 
					(roi_yy - baseY > j * K2_BLOCKDIM_Y) &&
					(roi_zz - baseZ > i * K2_BLOCKDIM_Z)    )d_Dst[i * K2_BLOCKDIM_Z * pitchZ + j * K2_BLOCKDIM_Y * pitchY + k * K2_BLOCKDIM_X] = max;

			}
		}
	}
	

}

extern "C" __declspec (dllexport) void maximumSphereFilterGPU6(
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
	if(_kernel_radus <= 2){
		assert( K2_BLOCKDIM_X * K2_HALO_STEPS_X >= KERNEL_RADIUS );
		assert( K2_BLOCKDIM_Y * K2_HALO_STEPS_Y >= KERNEL_RADIUS );
		assert( K2_BLOCKDIM_Z * K2_HALO_STEPS_Z >= KERNEL_RADIUS );

		int x_blocknum = (roi_w % (K2_RESULT_STEPS_X * K2_BLOCKDIM_X) == 0) ? roi_w / (K2_RESULT_STEPS_X * K2_BLOCKDIM_X) : roi_w / (K2_RESULT_STEPS_X * K2_BLOCKDIM_X) + 1;
		int y_blocknum = (roi_h % (K2_RESULT_STEPS_Y * K2_BLOCKDIM_Y) == 0) ? roi_h / (K2_RESULT_STEPS_Y * K2_BLOCKDIM_Y) : roi_h / (K2_RESULT_STEPS_Y * K2_BLOCKDIM_Y) + 1;
		int z_blocknum = (roi_d % (K2_RESULT_STEPS_Z * K2_BLOCKDIM_Z) == 0) ? roi_d / (K2_RESULT_STEPS_Z * K2_BLOCKDIM_Z) : roi_d / (K2_RESULT_STEPS_Z * K2_BLOCKDIM_Z) + 1;

		dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
		dim3 threads(K2_BLOCKDIM_X, K2_BLOCKDIM_Y, K2_BLOCKDIM_Z);

		if(_kernel_radus == 1)
			maximumSphereFilterKernel2<1><<<blocks, threads>>>(
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
		else
			maximumSphereFilterKernel2<2><<<blocks, threads>>>(
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
	}
	else if(_kernel_radus <= 4){
		assert( K4_BLOCKDIM_X * K4_HALO_STEPS_X >= KERNEL_RADIUS );
		assert( K4_BLOCKDIM_Y * K4_HALO_STEPS_Y >= KERNEL_RADIUS );
		assert( K4_BLOCKDIM_Z * K4_HALO_STEPS_Z >= KERNEL_RADIUS );

		int x_blocknum = (roi_w % (K4_RESULT_STEPS_X * K4_BLOCKDIM_X) == 0) ? roi_w / (K4_RESULT_STEPS_X * K4_BLOCKDIM_X) : roi_w / (K4_RESULT_STEPS_X * K4_BLOCKDIM_X) + 1;
		int y_blocknum = (roi_h % (K4_RESULT_STEPS_Y * K4_BLOCKDIM_Y) == 0) ? roi_h / (K4_RESULT_STEPS_Y * K4_BLOCKDIM_Y) : roi_h / (K4_RESULT_STEPS_Y * K4_BLOCKDIM_Y) + 1;
		int z_blocknum = (roi_d % (K4_RESULT_STEPS_Z * K4_BLOCKDIM_Z) == 0) ? roi_d / (K4_RESULT_STEPS_Z * K4_BLOCKDIM_Z) : roi_d / (K4_RESULT_STEPS_Z * K4_BLOCKDIM_Z) + 1;

		dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
		dim3 threads(K4_BLOCKDIM_X, K4_BLOCKDIM_Y, K4_BLOCKDIM_Z);

		if(_kernel_radus == 3)
			maximumSphereFilterKernel4<3><<<blocks, threads>>>(
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
		else
			maximumSphereFilterKernel4<4><<<blocks, threads>>>(
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
	}
	else if(_kernel_radus <= 6){
		assert( K6_BLOCKDIM_X * K6_HALO_STEPS_X >= KERNEL_RADIUS );
		assert( K6_BLOCKDIM_Y * K6_HALO_STEPS_Y >= KERNEL_RADIUS );
		assert( K6_BLOCKDIM_Z * K6_HALO_STEPS_Z >= KERNEL_RADIUS );

		int x_blocknum = (roi_w % (K6_RESULT_STEPS_X * K6_BLOCKDIM_X) == 0) ? roi_w / (K6_RESULT_STEPS_X * K6_BLOCKDIM_X) : roi_w / (K6_RESULT_STEPS_X * K6_BLOCKDIM_X) + 1;
		int y_blocknum = (roi_h % (K6_RESULT_STEPS_Y * K6_BLOCKDIM_Y) == 0) ? roi_h / (K6_RESULT_STEPS_Y * K6_BLOCKDIM_Y) : roi_h / (K6_RESULT_STEPS_Y * K6_BLOCKDIM_Y) + 1;
		int z_blocknum = (roi_d % (K6_RESULT_STEPS_Z * K6_BLOCKDIM_Z) == 0) ? roi_d / (K6_RESULT_STEPS_Z * K6_BLOCKDIM_Z) : roi_d / (K6_RESULT_STEPS_Z * K6_BLOCKDIM_Z) + 1;

		dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
		dim3 threads(K6_BLOCKDIM_X, K6_BLOCKDIM_Y, K6_BLOCKDIM_Z);

		if(_kernel_radus == 5)
			maximumSphereFilterKernel6<5><<<blocks, threads>>>(
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
		else
			maximumSphereFilterKernel6<6><<<blocks, threads>>>(
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
	}
    getLastCudaError("maximumSphereFilterKernel6() execution failed\n");
}

/*
#define SPK_BLOCKDIM 128

__global__ void maximumSphereFilterKernel6(
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
	int roi_w,
	int roi_h,
	int roi_d
){
    __shared__ float s_Data[KERNEL_LENGTH*KERNEL_LENGTH];

    //Offset to the K6_HALO edges
	const int idx = roi_x + blockIdx.x - KERNEL_RADIUS;
    const int idy = roi_y + blockIdx.y - KERNEL_RADIUS + threadIdx.x;
	const int idz = roi_z + blockIdx.z - KERNEL_RADIUS + threadIdx.y;
	const int id = idz * pitchZ + idy * pitchY + idx;
	const int smem_size = KERNEL_LENGTH*KERNEL_LENGTH;
	const int tid = threadIdx.x + threadIdx.y * KERNEL_LENGTH;

	//Load main data
	s_Data[tid] = FLT_MIN;
	#pragma unroll
	for(int i = 0; i < KERNEL_LENGTH; i++){
		if(id + i >= 0 && id + i < imageW*imageH*imageZ){
			float tmp = d_Src[id + i]*c_Kernel6[tid*KERNEL_LENGTH + i];
			if(s_Data[tid] < tmp)s_Data[tid] = tmp;
		}
	}

	//Compute and store K6_RESULTs
	__syncthreads();
	
	if(tid < 2048 && tid + 2048 < smem_size) { if(s_Data[tid + 2048] > s_Data[tid])s_Data[tid] = s_Data[tid + 2048]; __syncthreads(); }
	if(tid < 1024 && tid + 1024 < smem_size) { if(s_Data[tid + 1024] > s_Data[tid])s_Data[tid] = s_Data[tid + 1024]; __syncthreads(); }
	if(tid < 512 && tid + 512 < smem_size) { if(s_Data[tid + 512] > s_Data[tid])s_Data[tid] = s_Data[tid + 512]; __syncthreads(); }
	if(tid < 256 && tid + 256 < smem_size) { if(s_Data[tid + 256] > s_Data[tid])s_Data[tid] = s_Data[tid + 256]; __syncthreads(); }
	if(tid < 128 && tid + 128 < smem_size) { if(s_Data[tid + 128] > s_Data[tid])s_Data[tid] = s_Data[tid + 128]; __syncthreads(); }
	if(tid < 64 && tid + 64 < smem_size) { if(s_Data[tid + 64] > s_Data[tid])s_Data[tid] = s_Data[tid + 64]; __syncthreads(); }
	if(tid < 32 && tid + 32 < smem_size) { if(s_Data[tid + 32] > s_Data[tid])s_Data[tid] = s_Data[tid + 32]; __syncthreads(); }
	if(tid < 16 && tid + 16 < smem_size) { if(s_Data[tid + 16] > s_Data[tid])s_Data[tid] = s_Data[tid + 16]; __syncthreads(); }
	if(tid < 8 && tid + 8 < smem_size) { if(s_Data[tid + 8] > s_Data[tid])s_Data[tid] = s_Data[tid + 8]; __syncthreads(); }
	if(tid < 4 && tid + 4 < smem_size) { if(s_Data[tid + 4] > s_Data[tid])s_Data[tid] = s_Data[tid + 4]; __syncthreads(); }
	if(tid < 2 && tid + 2 < smem_size) { if(s_Data[tid + 2] > s_Data[tid])s_Data[tid] = s_Data[tid + 2]; __syncthreads(); }
	if(tid < 1 && tid + 1 < smem_size) { if(s_Data[tid + 1] > s_Data[tid])s_Data[tid] = s_Data[tid + 1]; __syncthreads(); }
	
	if(tid == 0)d_Dst[(idz + KERNEL_RADIUS) * pitchZ + (idy + KERNEL_RADIUS) * pitchY + idx + KERNEL_RADIUS] = s_Data[0];
}

extern "C" __declspec (dllexport) void maximumSphereFilterGPU6(
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
    int x_blocknum = roi_w;
	int y_blocknum = roi_h;
	int z_blocknum = roi_d;

    dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
	dim3 threads(KERNEL_LENGTH, KERNEL_LENGTH);
	
    maximumSphereFilterKernel6<<<blocks, threads>>>(
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
		roi_w,
		roi_h,
		roi_d
    );
	
    getLastCudaError("maximumSphereFilterKernel6() execution failed\n");
}
*/

template<int k_radius> __global__ void minimumSphereFilterKernel6(
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
	__shared__ float s_Data[(K6_RESULT_STEPS_Z + 2 * K6_HALO_STEPS_Z) * K6_BLOCKDIM_Z][(K6_RESULT_STEPS_Y + 2 * K6_HALO_STEPS_Y) * K6_BLOCKDIM_Y][(K6_RESULT_STEPS_X + 2 * K6_HALO_STEPS_X) * K6_BLOCKDIM_X];

    //Offset to the K6_HALO edges
    const int baseX = roi_x + (blockIdx.x * K6_RESULT_STEPS_X - K6_HALO_STEPS_X) * K6_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * K6_RESULT_STEPS_Y - K6_HALO_STEPS_Y) * K6_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + (blockIdx.z * K6_RESULT_STEPS_Z - K6_HALO_STEPS_Z) * K6_BLOCKDIM_Z + threadIdx.z;

	d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
	d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	//Load main data
	#pragma unroll
	for(int i = 0; i < K6_HALO_STEPS_Z + K6_RESULT_STEPS_Z + K6_HALO_STEPS_Z; i++){
		#pragma unroll
		for(int j = 0; j < K6_HALO_STEPS_Y + K6_RESULT_STEPS_Y + K6_HALO_STEPS_Y; j++){
			#pragma unroll
			for(int k = 0; k < K6_HALO_STEPS_X + K6_RESULT_STEPS_X + K6_HALO_STEPS_X; k++){
				if((imageZ - baseZ > i * K6_BLOCKDIM_Z && baseZ + i * K6_BLOCKDIM_Z >= 0) &&
				   (imageH - baseY > j * K6_BLOCKDIM_Y && baseY + j * K6_BLOCKDIM_Y >= 0) &&
				   (imageW - baseX > k * K6_BLOCKDIM_X && baseX + k * K6_BLOCKDIM_X >= 0)   )
				   s_Data[threadIdx.z + i * K6_BLOCKDIM_Z][threadIdx.y + j * K6_BLOCKDIM_Y][threadIdx.x + k * K6_BLOCKDIM_X] = d_Src[i * K6_BLOCKDIM_Z * pitchZ + j * K6_BLOCKDIM_Y * pitchY + k * K6_BLOCKDIM_X];
				else s_Data[threadIdx.z + i * K6_BLOCKDIM_Z][threadIdx.y + j * K6_BLOCKDIM_Y][threadIdx.x + k * K6_BLOCKDIM_X] = FLT_MAX;
			}
		}
	}

	//Compute and store K6_RESULTs
	__syncthreads();
	
	#pragma unroll
	for(int i = K6_HALO_STEPS_Z; i < K6_HALO_STEPS_Z + K6_RESULT_STEPS_Z; i++){
		#pragma unroll
		for(int j = K6_HALO_STEPS_Y; j < K6_HALO_STEPS_Y + K6_RESULT_STEPS_Y; j++){
			#pragma unroll
			for(int k = K6_HALO_STEPS_X; k < K6_HALO_STEPS_X + K6_RESULT_STEPS_X; k++){
				float min = FLT_MAX;
				
				#pragma unroll
				for(int z = -1 * k_radius; z <= k_radius; z++){
					#pragma unroll
					for(int y = -1 * k_radius; y <= k_radius; y++){
						#pragma unroll
						for(int x = -1 * k_radius; x <= k_radius; x++){
							float val =  s_Data[threadIdx.z + i * K6_BLOCKDIM_Z + z][threadIdx.y + j * K6_BLOCKDIM_Y + y][threadIdx.x + k * K6_BLOCKDIM_X + x];
							min = (min > val && c_Kernel6[(KERNEL_RADIUS + z)*KERNEL_LENGTH*KERNEL_LENGTH + (KERNEL_RADIUS + y)*KERNEL_LENGTH + KERNEL_RADIUS + x] > 0.0f) ? val : min;
						}
					}
				}
				
				if( (roi_xx - baseX > k * K6_BLOCKDIM_X) && 
					(roi_yy - baseY > j * K6_BLOCKDIM_Y) &&
					(roi_zz - baseZ > i * K6_BLOCKDIM_Z)    )d_Dst[i * K6_BLOCKDIM_Z * pitchZ + j * K6_BLOCKDIM_Y * pitchY + k * K6_BLOCKDIM_X] = min;

			}
		}
	}
	

}

template<int k_radius> __global__ void minimumSphereFilterKernel4(
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
	__shared__ float s_Data[(K4_RESULT_STEPS_Z + 2 * K4_HALO_STEPS_Z) * K4_BLOCKDIM_Z][(K4_RESULT_STEPS_Y + 2 * K4_HALO_STEPS_Y) * K4_BLOCKDIM_Y][(K4_RESULT_STEPS_X + 2 * K4_HALO_STEPS_X) * K4_BLOCKDIM_X];

    //Offset to the K4_HALO edges
    const int baseX = roi_x + (blockIdx.x * K4_RESULT_STEPS_X - K4_HALO_STEPS_X) * K4_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * K4_RESULT_STEPS_Y - K4_HALO_STEPS_Y) * K4_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + (blockIdx.z * K4_RESULT_STEPS_Z - K4_HALO_STEPS_Z) * K4_BLOCKDIM_Z + threadIdx.z;

	d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
	d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	//Load main data
	#pragma unroll
	for(int i = 0; i < K4_HALO_STEPS_Z + K4_RESULT_STEPS_Z + K4_HALO_STEPS_Z; i++){
		#pragma unroll
		for(int j = 0; j < K4_HALO_STEPS_Y + K4_RESULT_STEPS_Y + K4_HALO_STEPS_Y; j++){
			#pragma unroll
			for(int k = 0; k < K4_HALO_STEPS_X + K4_RESULT_STEPS_X + K4_HALO_STEPS_X; k++){
				if((imageZ - baseZ > i * K4_BLOCKDIM_Z && baseZ + i * K4_BLOCKDIM_Z >= 0) &&
				   (imageH - baseY > j * K4_BLOCKDIM_Y && baseY + j * K4_BLOCKDIM_Y >= 0) &&
				   (imageW - baseX > k * K4_BLOCKDIM_X && baseX + k * K4_BLOCKDIM_X >= 0)   )
				   s_Data[threadIdx.z + i * K4_BLOCKDIM_Z][threadIdx.y + j * K4_BLOCKDIM_Y][threadIdx.x + k * K4_BLOCKDIM_X] = d_Src[i * K4_BLOCKDIM_Z * pitchZ + j * K4_BLOCKDIM_Y * pitchY + k * K4_BLOCKDIM_X];
				else s_Data[threadIdx.z + i * K4_BLOCKDIM_Z][threadIdx.y + j * K4_BLOCKDIM_Y][threadIdx.x + k * K4_BLOCKDIM_X] = FLT_MAX;
			}
		}
	}

	//Compute and store K4_RESULTs
	__syncthreads();
	
	#pragma unroll
	for(int i = K4_HALO_STEPS_Z; i < K4_HALO_STEPS_Z + K4_RESULT_STEPS_Z; i++){
		#pragma unroll
		for(int j = K4_HALO_STEPS_Y; j < K4_HALO_STEPS_Y + K4_RESULT_STEPS_Y; j++){
			#pragma unroll
			for(int k = K4_HALO_STEPS_X; k < K4_HALO_STEPS_X + K4_RESULT_STEPS_X; k++){
				float min = FLT_MAX;
				
				#pragma unroll
				for(int z = -1 * k_radius; z <= k_radius; z++){
					#pragma unroll
					for(int y = -1 * k_radius; y <= k_radius; y++){
						#pragma unroll
						for(int x = -1 * k_radius; x <= k_radius; x++){
							float val =  s_Data[threadIdx.z + i * K4_BLOCKDIM_Z + z][threadIdx.y + j * K4_BLOCKDIM_Y + y][threadIdx.x + k * K4_BLOCKDIM_X + x];
							min = (min > val && c_Kernel6[(KERNEL_RADIUS + z)*KERNEL_LENGTH*KERNEL_LENGTH + (KERNEL_RADIUS + y)*KERNEL_LENGTH + KERNEL_RADIUS + x] > 0.0f) ? val : min;
						}
					}
				}
				
				if( (roi_xx - baseX > k * K4_BLOCKDIM_X) && 
					(roi_yy - baseY > j * K4_BLOCKDIM_Y) &&
					(roi_zz - baseZ > i * K4_BLOCKDIM_Z)    )d_Dst[i * K4_BLOCKDIM_Z * pitchZ + j * K4_BLOCKDIM_Y * pitchY + k * K4_BLOCKDIM_X] = min;

			}
		}
	}
	

}

template<int k_radius> __global__ void minimumSphereFilterKernel2(
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
	__shared__ float s_Data[(K2_RESULT_STEPS_Z + 2 * K2_HALO_STEPS_Z) * K2_BLOCKDIM_Z][(K2_RESULT_STEPS_Y + 2 * K2_HALO_STEPS_Y) * K2_BLOCKDIM_Y][(K2_RESULT_STEPS_X + 2 * K2_HALO_STEPS_X) * K2_BLOCKDIM_X];

    //Offset to the K2_HALO edges
    const int baseX = roi_x + (blockIdx.x * K2_RESULT_STEPS_X - K2_HALO_STEPS_X) * K2_BLOCKDIM_X + threadIdx.x;
    const int baseY = roi_y + (blockIdx.y * K2_RESULT_STEPS_Y - K2_HALO_STEPS_Y) * K2_BLOCKDIM_Y + threadIdx.y;
	const int baseZ = roi_z + (blockIdx.z * K2_RESULT_STEPS_Z - K2_HALO_STEPS_Z) * K2_BLOCKDIM_Z + threadIdx.z;

	d_Src += baseZ * pitchZ + baseY * pitchY + baseX;
	d_Dst += baseZ * pitchZ + baseY * pitchY + baseX;

	//Load main data
	#pragma unroll
	for(int i = 0; i < K2_HALO_STEPS_Z + K2_RESULT_STEPS_Z + K2_HALO_STEPS_Z; i++){
		#pragma unroll
		for(int j = 0; j < K2_HALO_STEPS_Y + K2_RESULT_STEPS_Y + K2_HALO_STEPS_Y; j++){
			#pragma unroll
			for(int k = 0; k < K2_HALO_STEPS_X + K2_RESULT_STEPS_X + K2_HALO_STEPS_X; k++){
				if((imageZ - baseZ > i * K2_BLOCKDIM_Z && baseZ + i * K2_BLOCKDIM_Z >= 0) &&
				   (imageH - baseY > j * K2_BLOCKDIM_Y && baseY + j * K2_BLOCKDIM_Y >= 0) &&
				   (imageW - baseX > k * K2_BLOCKDIM_X && baseX + k * K2_BLOCKDIM_X >= 0)   )
				   s_Data[threadIdx.z + i * K2_BLOCKDIM_Z][threadIdx.y + j * K2_BLOCKDIM_Y][threadIdx.x + k * K2_BLOCKDIM_X] = d_Src[i * K2_BLOCKDIM_Z * pitchZ + j * K2_BLOCKDIM_Y * pitchY + k * K2_BLOCKDIM_X];
				else s_Data[threadIdx.z + i * K2_BLOCKDIM_Z][threadIdx.y + j * K2_BLOCKDIM_Y][threadIdx.x + k * K2_BLOCKDIM_X] = FLT_MAX;
			}
		}
	}

	//Compute and store K2_RESULTs
	__syncthreads();
	
	#pragma unroll
	for(int i = K2_HALO_STEPS_Z; i < K2_HALO_STEPS_Z + K2_RESULT_STEPS_Z; i++){
		#pragma unroll
		for(int j = K2_HALO_STEPS_Y; j < K2_HALO_STEPS_Y + K2_RESULT_STEPS_Y; j++){
			#pragma unroll
			for(int k = K2_HALO_STEPS_X; k < K2_HALO_STEPS_X + K2_RESULT_STEPS_X; k++){
				float min = FLT_MAX;
				
				#pragma unroll
				for(int z = -1 * k_radius; z <= k_radius; z++){
					#pragma unroll
					for(int y = -1 * k_radius; y <= k_radius; y++){
						#pragma unroll
						for(int x = -1 * k_radius; x <= k_radius; x++){
							float val =  s_Data[threadIdx.z + i * K2_BLOCKDIM_Z + z][threadIdx.y + j * K2_BLOCKDIM_Y + y][threadIdx.x + k * K2_BLOCKDIM_X + x];
							min = (min > val && c_Kernel6[(KERNEL_RADIUS + z)*KERNEL_LENGTH*KERNEL_LENGTH + (KERNEL_RADIUS + y)*KERNEL_LENGTH + KERNEL_RADIUS + x] > 0.0f) ? val : min;
						}
					}
				}
				
				if( (roi_xx - baseX > k * K2_BLOCKDIM_X) && 
					(roi_yy - baseY > j * K2_BLOCKDIM_Y) &&
					(roi_zz - baseZ > i * K2_BLOCKDIM_Z)    )d_Dst[i * K2_BLOCKDIM_Z * pitchZ + j * K2_BLOCKDIM_Y * pitchY + k * K2_BLOCKDIM_X] = min;

			}
		}
	}
	

}


extern "C" __declspec (dllexport) void minimumSphereFilterGPU6(
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
    if(_kernel_radus <= 2){
		assert( K2_BLOCKDIM_X * K2_HALO_STEPS_X >= KERNEL_RADIUS );
		assert( K2_BLOCKDIM_Y * K2_HALO_STEPS_Y >= KERNEL_RADIUS );
		assert( K2_BLOCKDIM_Z * K2_HALO_STEPS_Z >= KERNEL_RADIUS );

		int x_blocknum = (roi_w % (K2_RESULT_STEPS_X * K2_BLOCKDIM_X) == 0) ? roi_w / (K2_RESULT_STEPS_X * K2_BLOCKDIM_X) : roi_w / (K2_RESULT_STEPS_X * K2_BLOCKDIM_X) + 1;
		int y_blocknum = (roi_h % (K2_RESULT_STEPS_Y * K2_BLOCKDIM_Y) == 0) ? roi_h / (K2_RESULT_STEPS_Y * K2_BLOCKDIM_Y) : roi_h / (K2_RESULT_STEPS_Y * K2_BLOCKDIM_Y) + 1;
		int z_blocknum = (roi_d % (K2_RESULT_STEPS_Z * K2_BLOCKDIM_Z) == 0) ? roi_d / (K2_RESULT_STEPS_Z * K2_BLOCKDIM_Z) : roi_d / (K2_RESULT_STEPS_Z * K2_BLOCKDIM_Z) + 1;

		dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
		dim3 threads(K2_BLOCKDIM_X, K2_BLOCKDIM_Y, K2_BLOCKDIM_Z);

		if(_kernel_radus == 1)
			minimumSphereFilterKernel2<1><<<blocks, threads>>>(
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
		else
			minimumSphereFilterKernel2<2><<<blocks, threads>>>(
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
	}
	else if(_kernel_radus <= 4){
		assert( K4_BLOCKDIM_X * K4_HALO_STEPS_X >= KERNEL_RADIUS );
		assert( K4_BLOCKDIM_Y * K4_HALO_STEPS_Y >= KERNEL_RADIUS );
		assert( K4_BLOCKDIM_Z * K4_HALO_STEPS_Z >= KERNEL_RADIUS );

		int x_blocknum = (roi_w % (K4_RESULT_STEPS_X * K4_BLOCKDIM_X) == 0) ? roi_w / (K4_RESULT_STEPS_X * K4_BLOCKDIM_X) : roi_w / (K4_RESULT_STEPS_X * K4_BLOCKDIM_X) + 1;
		int y_blocknum = (roi_h % (K4_RESULT_STEPS_Y * K4_BLOCKDIM_Y) == 0) ? roi_h / (K4_RESULT_STEPS_Y * K4_BLOCKDIM_Y) : roi_h / (K4_RESULT_STEPS_Y * K4_BLOCKDIM_Y) + 1;
		int z_blocknum = (roi_d % (K4_RESULT_STEPS_Z * K4_BLOCKDIM_Z) == 0) ? roi_d / (K4_RESULT_STEPS_Z * K4_BLOCKDIM_Z) : roi_d / (K4_RESULT_STEPS_Z * K4_BLOCKDIM_Z) + 1;

		dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
		dim3 threads(K4_BLOCKDIM_X, K4_BLOCKDIM_Y, K4_BLOCKDIM_Z);

		if(_kernel_radus == 3)
			minimumSphereFilterKernel4<3><<<blocks, threads>>>(
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
		else
			minimumSphereFilterKernel4<4><<<blocks, threads>>>(
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
	}
	else if(_kernel_radus <= 6){
		assert( K6_BLOCKDIM_X * K6_HALO_STEPS_X >= KERNEL_RADIUS );
		assert( K6_BLOCKDIM_Y * K6_HALO_STEPS_Y >= KERNEL_RADIUS );
		assert( K6_BLOCKDIM_Z * K6_HALO_STEPS_Z >= KERNEL_RADIUS );

		int x_blocknum = (roi_w % (K6_RESULT_STEPS_X * K6_BLOCKDIM_X) == 0) ? roi_w / (K6_RESULT_STEPS_X * K6_BLOCKDIM_X) : roi_w / (K6_RESULT_STEPS_X * K6_BLOCKDIM_X) + 1;
		int y_blocknum = (roi_h % (K6_RESULT_STEPS_Y * K6_BLOCKDIM_Y) == 0) ? roi_h / (K6_RESULT_STEPS_Y * K6_BLOCKDIM_Y) : roi_h / (K6_RESULT_STEPS_Y * K6_BLOCKDIM_Y) + 1;
		int z_blocknum = (roi_d % (K6_RESULT_STEPS_Z * K6_BLOCKDIM_Z) == 0) ? roi_d / (K6_RESULT_STEPS_Z * K6_BLOCKDIM_Z) : roi_d / (K6_RESULT_STEPS_Z * K6_BLOCKDIM_Z) + 1;

		dim3 blocks(x_blocknum, y_blocknum, z_blocknum);
		dim3 threads(K6_BLOCKDIM_X, K6_BLOCKDIM_Y, K6_BLOCKDIM_Z);

		if(_kernel_radus == 5)
			minimumSphereFilterKernel6<5><<<blocks, threads>>>(
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
		else
			minimumSphereFilterKernel6<6><<<blocks, threads>>>(
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
	}
    getLastCudaError("minimumSphereFilterKernel6() execution failed\n");
}
