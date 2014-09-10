#define DLL_CONVOLUTIONSEPARABLE
#include "convolutionSeparable_common.h"
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

//__device__ __constant__ float filter_5x5x5[5][5][5];

texture<float, 3, cudaReadModeElementType> tex_Volume;
texture<int, 3, cudaReadModeElementType> tex_VolumeI;
cudaArray *tex_volumeArray = 0;
cudaArray *tex_volumeArray2 = 0;
cudaArray *_surfArray1 = 0;
cudaArray *_surfArray2 = 0;
//surface<void, cudaSurfaceType3D> _surface1;
//surface<void, cudaSurfaceType3D> _surface2;
int *bufferI = 0;
float *buffer = 0;

extern "C" __declspec (dllexport) void setSphereFilter3D_Tex3D(int rad)
{
	int len = rad*2 + 1;
/*
	if(rad == 1){
		float tmp[3][3][3];

		for(int z = 0; z < 3; z++){
			for(int y = 0; y < 3; y++){
				for(int x = 0; x < 3; x++){
					if( sqrt( (x - rad)*(x - rad) + (y - rad)*(y - rad) + (z - rad)*(z - rad) ) <= rad )
						tmp[z][y][x] = 1.0f;
					else tmp[z][y][x] = 0.0f;
				}
			}
		}

		cudaMemcpyToSymbol(filter_3x3x3, tmp, len* len * len * sizeof(float));
	}
	if(rad == 2){
		float tmp[5][5][5];

		for(int z = 0; z < 5; z++){
			for(int y = 0; y < 5; y++){
				for(int x = 0; x < 5; x++){
					if( sqrt( (x - rad)*(x - rad) + (y - rad)*(y - rad) + (z - rad)*(z - rad) ) <= rad )
						tmp[z][y][x] = 1.0f;
					else tmp[z][y][x] = 0.0f;
				}
			}
		}

		cudaMemcpyToSymbol(filter_5x5x5, tmp, len* len * len * sizeof(float));
	}
*/
}

__device__ int Get_3D_Index(int x, int y, int z, int DATA_W, int DATA_H)
{
	return x + y * DATA_W + z * DATA_W * DATA_H;
}

__global__ void Convolution_3D_Texture_Unrolled_3x3x3(float* Filter_Response, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float max = 0.0f;
	float tmp;
	
	//0.00158 sec
	/*
	int radius = 1;
	int xst  = (x - radius >= 0)	 ? -radius : -radius + (x - radius);
	int xend = (x + radius < DATA_W) ?  radius : radius - (x - radius);
	int yst  = (y - radius >= 0)	 ? -radius : -radius + (y - radius);
	int yend = (y + radius < DATA_H) ?  radius : radius - (y - radius);
	int zst  = (z - radius >= 0)	 ? -radius : -radius + (z - radius);
	int zend = (z + radius < DATA_D) ?  radius : radius - (z - radius);
	for(int xx = xst ; xx <= xend; xx++){
		for(int yy = yst ; yy <= yend; yy++){
			for(int zz = zst ; zz <= zend; zz++){
				tmp = tex3D(tex_Volume, x - xx + 0.5f, y - yy + 0.5f, z - zz + 0.5f) * filter_3x3x3[zz+radius][yy+radius][xx+radius]; max = (max < tmp) ? tmp : max;
			}
		}
	}
	*/

	//0.00100 sec
	/*
	tmp = (x-1.0f >= 0.0f && y-1.0f >= 0.0f && z-1.0f >= 0.0f)					   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f && y-1.0f >= 0.0f)									   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f && y-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)			   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f && z-1.0f >= 0.0f)									   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f)														   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)								   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && z-1.0f >= 0.0f)			   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)								   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && z+1.0f <= DATA_D-1.0f)	   ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y-1.0f >= 0.0f && z-1.0f >= 0.0f)									   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y-1.0f >= 0.0f)														   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)								   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (z-1.0f >= 0.0f)														   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp =																			 tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][1]		   ; max = (max < tmp) ? tmp : max;
	tmp = (z+1.0f <= DATA_D-1.0f)												   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y+1.0f <= DATA_H-1.0f && z-1.0f >= 0.0f)								   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y+1.0f <= DATA_H-1.0f)												   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y+1.0f <= DATA_H-1.0f && z+1.0f <= DATA_D-1.0f)						   ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && y-1.0f >= 0.0f && z-1.0f >= 0.0f)			   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && y-1.0f >= 0.0f)								   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && y-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)	   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && z-1.0f >= 0.0f)								   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f)												   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && z+1.0f <= DATA_D-1.0f)						   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f && z-1.0f >= 0.0f)	   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f)						   ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f&& z+1.0f <= DATA_D-1.0f) ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	*/

	//0.00099 sec
	/*
	if(x-1.0f >= 0.0f && y-1.0f >= 0.0f && z-1.0f >= 0.0f)						{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f && y-1.0f >= 0.0f)										{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f && y-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)				{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f && z-1.0f >= 0.0f)										{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f)															{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)									{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && z-1.0f >= 0.0f)				{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)									{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][2]; max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && z+1.0f <= DATA_D-1.0f)		{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][2]; max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && z-1.0f >= 0.0f)										{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][1]; max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f)															{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][1]; max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)									{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][1]; max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f)															{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][1]; max = (max < tmp) ? tmp : max;}
	tmp =																			   tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][1]; max = (max < tmp) ? tmp : max;
	if(z+1.0f <= DATA_D-1.0f)													{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][1]; max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && z-1.0f >= 0.0f)									{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][1]; max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f)													{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][1]; max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && z+1.0f <= DATA_D-1.0f)							{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][1]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && y-1.0f >= 0.0f && z-1.0f >= 0.0f)				{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && y-1.0f >= 0.0f)									{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && y-1.0f >= 0.0f && z+1.0f <= DATA_D-1.0f)		{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && z-1.0f >= 0.0f)									{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f)													{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && z+1.0f <= DATA_D-1.0f)							{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f && z-1.0f >= 0.0f)		{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f)							{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][0]; max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f&& z+1.0f <= DATA_D-1.0f)	{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][0]; max = (max < tmp) ? tmp : max;}
	*/

	//0.00031 sec
	/*
	tmp = (x-1.0f >= 0.0f)		  ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][2] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y-1.0f >= 0.0f)		  ? tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (z-1.0f >= 0.0f)		  ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp =							tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][1]		  ; max = (max < tmp) ? tmp : max;
	tmp = (z+1.0f <= DATA_D-1.0f) ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y+1.0f <= DATA_H-1.0f) ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][1] : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f) ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][0] : FLT_MIN; max = (max < tmp) ? tmp : max;
	*/

	//0.00030 sec
	if(x-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	tmp =							   tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;
	if(z+1.0f <= DATA_D-1.0f)	{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f)	{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f)	{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	
	
	//0.00072 sec
	/*
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][2]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][1]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][2][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][2][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y - 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][2][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][1][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][1][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][1][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z - 1.0f + 0.5f) * filter_3x3x3[2][0][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) * filter_3x3x3[1][0][0]; max = (max < tmp) ? tmp : max;
	tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 1.0f + 0.5f, z + 1.0f + 0.5f) * filter_3x3x3[0][0][0]; max = (max < tmp) ? tmp : max;
	*/
	Filter_Response[Get_3D_Index(x,y,z,DATA_W,DATA_H)] = max;
}

extern "C" __declspec (dllexport) void dilateTextureCube1GPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ
){
	// 32 threads along x, 16 along y
	int threadsInX = 32;
	int threadsInY = 16;
	int threadsInZ = 1;

    // Round up to get sufficient number of blocks
    int blocksInX = (int)ceil((float)imageW / (float)threadsInX);
    int blocksInY = (int)ceil((float)imageH / (float)threadsInY);
	int blocksInZ = (int)ceil((float)imageZ / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	Convolution_3D_Texture_Unrolled_3x3x3<<<dimGrid, dimBlock>>>(d_Dst, imageW, imageH, imageZ);

	
}

extern "C" __declspec (dllexport) void destructCuda_Tex3D(){
	if(tex_volumeArray)checkCudaErrors(cudaFreeArray(tex_volumeArray));
	tex_volumeArray = 0;
}

extern "C" __declspec (dllexport) void initCuda_Tex3D(const float *h_volume, int imageW, int imageH, int imageZ){

	/*
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	destructCuda_LCT();
	*/
	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMalloc3DArray(&tex_volumeArray, &channelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_volume, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = tex_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex_Volume.normalized = false;                      // access with normalized texture coordinates
	tex_Volume.filterMode = cudaFilterModeLinear;       // linear interpolation
	tex_Volume.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
	tex_Volume.addressMode[1] = cudaAddressModeClamp;
	tex_Volume.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(&tex_Volume, tex_volumeArray, &channelDesc));

	/*
	sdkStopTimer(&hTimer);
    float ini_time = sdkGetTimerValue(&hTimer);
	printf("initCuda_Tex3D: time = %f\n", ini_time);
	sdkDeleteTimer(&hTimer);
	*/
}

__global__ void Dilation_Spherical_3D_Surface_Unrolled_rad_1(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float max = 0.0f;
	float tmp;
	/*
	tmp = (x-1 >= 0)		? surf3Dread<float>(_surface1, (x - 1)*4, y + 0, z + 0, cudaBoundaryModeTrap) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y-1 >= 0)		? surf3Dread<float>(_surface1, (x + 0)*4, y - 1, z + 0, cudaBoundaryModeTrap) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (z-1 >= 0)		? surf3Dread<float>(_surface1, (x + 0)*4, y + 0, z - 1, cudaBoundaryModeTrap) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp =					  surf3Dread<float>(_surface1, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap)		   ; max = (max < tmp) ? tmp : max;
	tmp = (z+1 <= DATA_D-1) ? surf3Dread<float>(_surface1, (x + 0)*4, y + 0, z + 1, cudaBoundaryModeTrap) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y+1 <= DATA_H-1) ? surf3Dread<float>(_surface1, (x + 0)*4, y + 1, z + 0, cudaBoundaryModeTrap) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1 <= DATA_W-1) ? surf3Dread<float>(_surface1, (x + 1)*4, y + 0, z + 0, cudaBoundaryModeTrap) : FLT_MIN; max = (max < tmp) ? tmp : max;
	*/
	
	/*
	if(x-1 >= 0)		{surf3Dread(&tmp, _surface1, (x - 1)*4, y + 0, z + 0, cudaBoundaryModeTrap); max = (max < tmp) ? tmp : max;}
	if(y-1 >= 0)		{surf3Dread(&tmp, _surface1, (x + 0)*4, y - 1, z + 0, cudaBoundaryModeTrap); max = (max < tmp) ? tmp : max;}
	if(z-1 >= 0)		{surf3Dread(&tmp, _surface1, (x + 0)*4, y + 0, z - 1, cudaBoundaryModeTrap); max = (max < tmp) ? tmp : max;}
						 surf3Dread(&tmp, _surface1, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap); max = (max < tmp) ? tmp : max;
	if(z+1 <= DATA_D-1)	{surf3Dread(&tmp, _surface1, (x + 0)*4, y + 0, z + 1, cudaBoundaryModeTrap); max = (max < tmp) ? tmp : max;}
	if(y+1 <= DATA_H-1)	{surf3Dread(&tmp, _surface1, (x + 0)*4, y + 1, z + 0, cudaBoundaryModeTrap); max = (max < tmp) ? tmp : max;}
	if(x+1 <= DATA_W-1)	{surf3Dread(&tmp, _surface1, (x + 1)*4, y + 0, z + 0, cudaBoundaryModeTrap); max = (max < tmp) ? tmp : max;}
	*/
	
	/*
	tmp = (z-1.0f >= 0.0f)		  ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp =							tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f)		  ; max = (max < tmp) ? tmp : max;
	tmp = (x-1.0f >= 0.0f)		  ? tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y-1.0f >= 0.0f)		  ? tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (x+1.0f <= DATA_W-1.0f) ? tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (y+1.0f <= DATA_H-1.0f) ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f) : FLT_MIN; max = (max < tmp) ? tmp : max;
	tmp = (z+1.0f <= DATA_D-1.0f) ? tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f) : FLT_MIN; max = (max < tmp) ? tmp : max;
	*/

	if(z-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	tmp =							   tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;
	if(x-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+1.0f <= DATA_W-1.0f)	{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f)	{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f)	{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f); max = (max < tmp) ? tmp : max;}
	
	/*
	*/

	//surf3Dwrite(max, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = max;
}

__global__ void Dilation_Spherical_3D_Surface_Unrolled_rad_2(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float max = 0.0f;
	float tmp;

	if(z-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	                                                                             tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;
	if(x+1.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+2.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	
	//surf3Dwrite(max, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = max;
}

__global__ void Dilation_Spherical_3D_Surface_Unrolled_rad_3(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float max = 0.0f;
	float tmp;

	if(z-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	                                                                             tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;
	if(x+1.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+2.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+3.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+3.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	
	
	//surf3Dwrite(max, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = max;
}

__global__ void Dilation_Spherical_3D_Surface_Unrolled_rad_4(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float max = 0.0f;
	float tmp;

	if(z-4.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 4.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z - 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-4.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 4.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-3.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-3.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-4.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 4.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	                                                                             tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;
	if(x+1.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+2.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+3.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(x+4.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 4.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+3.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(y+4.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 4.0 + 0.5f, z + 0.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z + 2.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 3.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	if(z+4.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 4.0 + 0.5f); max = (max < tmp) ? tmp : max;}
	
	//surf3Dwrite(max, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = max;
}


__global__ void Erosion_Spherical_3D_Surface_Unrolled_rad_1(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float min = FLT_MAX;
	float tmp;
	
	if(z-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z - 1.0f + 0.5f); min = (min > tmp) ? tmp : min;}
	tmp =							   tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); min = (min > tmp) ? tmp : min;
	if(x-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x - 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f)			{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y - 1.0f + 0.5f, z + 0.0f + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x+1.0f <= DATA_W-1.0f)	{tmp = tex3D(tex_Volume, x + 1.0f + 0.5f, y + 0.0f + 0.5f, z + 0.0f + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f)	{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 1.0f + 0.5f, z + 0.0f + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f)	{tmp = tex3D(tex_Volume, x + 0.0f + 0.5f, y + 0.0f + 0.5f, z + 1.0f + 0.5f); min = (min > tmp) ? tmp : min;}
	
	
	//surf3Dwrite(min, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = min;
}

__global__ void Erosion_Spherical_3D_Surface_Unrolled_rad_2(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float min = FLT_MAX;
	float tmp;

	if(z-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	                                                                             tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;
	if(x+1.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x+2.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	
	//surf3Dwrite(min, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = min;
}

__global__ void Erosion_Spherical_3D_Surface_Unrolled_rad_3(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float min = FLT_MAX;
	float tmp;

	if(z-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	                                                                             tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;
	if(x+1.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x+2.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x+3.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+3.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	
	
	//surf3Dwrite(min, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = min;
}

__global__ void Erosion_Spherical_3D_Surface_Unrolled_rad_4(float *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	float min = FLT_MAX;
	float tmp;

	if(z-4.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 4.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z - 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)                      {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)               {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 2.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 3.0 + 0.5f, z - 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-4.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 4.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-3.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-3.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x-3.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f)                                        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)                                 {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-4.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 4.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-3.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-2.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x-1.0f >= 0.0f)                                                          {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	                                                                             tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;
	if(x+1.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x+2.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x+3.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(x+4.0f <= DATA_W-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 4.0 + 0.5f, y + 0.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 2.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+3.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 3.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(y+4.0f <= DATA_H-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 4.0 + 0.5f, z + 0.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 2.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 3.0 + 0.5f, z + 1.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 3.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 3.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 3.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y - 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x-3.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 0.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 3.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 3.0 + 0.5f, y + 1.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 2.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 3.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 3.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 3.0 + 0.5f, z + 2.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 2.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 2.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 2.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f)               {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f)        {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y - 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && x-2.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && x-1.0f >= 0.0f)                                 {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f)                          {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 0.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 2.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 2.0 + 0.5f, y + 1.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f)        {tmp = tex3D(tex_Volume, x - 1.0 + 0.5f, y + 2.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f)                          {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 2.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f) {tmp = tex3D(tex_Volume, x + 1.0 + 0.5f, y + 2.0 + 0.5f, z + 3.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	if(z+4.0f <= DATA_D-1.0f)                                                   {tmp = tex3D(tex_Volume, x + 0.0 + 0.5f, y + 0.0 + 0.5f, z + 4.0 + 0.5f); min = (min > tmp) ? tmp : min;}
	
	//surf3Dwrite(min, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = min;
}


extern "C" __declspec (dllexport) void dilateSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
){
	if(tex_volumeArray == 0 || buffer == 0)return;

	// 32 threads along x, 16 along y
	int threadsInX = 32;
	int threadsInY = 16;
	int threadsInZ = 1;

    // Round up to get sufficient number of blocks
    int blocksInX = (int)ceil((float)imageW / (float)threadsInX);
    int blocksInY = (int)ceil((float)imageH / (float)threadsInY);
	int blocksInZ = (int)ceil((float)imageZ / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	switch(radius){
		case 1: Dilation_Spherical_3D_Surface_Unrolled_rad_1<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
		case 2: Dilation_Spherical_3D_Surface_Unrolled_rad_2<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
		case 3: Dilation_Spherical_3D_Surface_Unrolled_rad_3<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
		case 4: Dilation_Spherical_3D_Surface_Unrolled_rad_4<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
	}
	checkCudaErrors( cudaDeviceSynchronize() );
	
	copyToSuf3DFromDeviceMem(buffer, imageW, imageH, imageZ);
}

extern "C" __declspec (dllexport) void erodeSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
){
	if(tex_volumeArray == 0 || buffer == 0)return;

	// 32 threads along x, 16 along y
	int threadsInX = 32;
	int threadsInY = 16;
	int threadsInZ = 1;

    // Round up to get sufficient number of blocks
    int blocksInX = (int)ceil((float)imageW / (float)threadsInX);
    int blocksInY = (int)ceil((float)imageH / (float)threadsInY);
	int blocksInZ = (int)ceil((float)imageZ / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	switch(radius){
		case 1: Erosion_Spherical_3D_Surface_Unrolled_rad_1<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
		case 2: Erosion_Spherical_3D_Surface_Unrolled_rad_2<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
		case 3: Erosion_Spherical_3D_Surface_Unrolled_rad_3<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
		case 4: Erosion_Spherical_3D_Surface_Unrolled_rad_4<<<dimGrid, dimBlock>>>(buffer, imageW, imageH, imageZ); break;
	}
	checkCudaErrors( cudaDeviceSynchronize() );
	
	copyToSuf3DFromDeviceMem(buffer, imageW, imageH, imageZ);
}

extern "C" __declspec (dllexport) void destructCuda_Suf3D()
{
	if(tex_volumeArray)checkCudaErrors(cudaFreeArray(tex_volumeArray));
	tex_volumeArray = 0;

	if(tex_volumeArray2)checkCudaErrors(cudaFreeArray(tex_volumeArray2));
	tex_volumeArray2 = 0;

	if(buffer)checkCudaErrors( cudaFree(buffer) );
	buffer = 0;

	if(bufferI)checkCudaErrors( cudaFree(bufferI) );
	bufferI = 0;
}

extern "C" __declspec (dllexport) void initCuda_Suf3D(const float *d_volume, int imageW, int imageH, int imageZ)
{
	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMalloc3DArray(&tex_volumeArray, &channelDesc, volumeSize));
	
	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)d_volume, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = tex_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex_Volume.normalized = false;                      // access with normalized texture coordinates
	tex_Volume.filterMode = cudaFilterModePoint;       // linear interpolation
	tex_Volume.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	tex_Volume.addressMode[1] = cudaAddressModeClamp;
	tex_Volume.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex_Volume, tex_volumeArray));

	checkCudaErrors( cudaMalloc((void **)(&buffer), imageZ * imageW * imageH * sizeof(float)) );
}

extern "C" __declspec (dllexport) void copyToDeviceMemFromSuf3D(float *d_output, int imageW, int imageH, int imageZ)
{
	if(tex_volumeArray == 0)return;

	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcArray = tex_volumeArray;
	copyParams.dstPtr   = make_cudaPitchedPtr((void *)d_output, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	
	checkCudaErrors(cudaMemcpy3D(&copyParams));
}

extern "C" __declspec (dllexport) void copyToSuf3DFromDeviceMem(float *d_input, int imageW, int imageH, int imageZ)
{
	if(tex_volumeArray == 0)return;

	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)d_input, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = tex_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	checkCudaErrors(cudaUnbindTexture(tex_Volume));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaBindTextureToArray(tex_Volume, tex_volumeArray));
}


__global__ void DilationSegments_Spherical_3D_Surface_Unrolled_rad_1(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にない場合(own < 0)、カーネル範囲内の最も近いsegment領域(nbr >= 0   空白の値が-1のみであればnbr != own)があればその値に変更する(own = nbr)
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own < 0) ? d : 0;
	if(z-1.0f >= 0.0f		 &&	d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z - 1.0f); if(nbr >= 0){d = 1; own = nbr;}}
	if(x-1.0f >= 0.0f		 &&	d > 1){nbr = tex3D(tex_VolumeI, x - 1.0f, y + 0.0f, z + 0.0f); if(nbr >= 0){d = 1; own = nbr;}}
	if(y-1.0f >= 0.0f		 && d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y - 1.0f, z + 0.0f); if(nbr >= 0){d = 1; own = nbr;}}
	if(x+1.0f <= DATA_W-1.0f && d > 1){nbr = tex3D(tex_VolumeI, x + 1.0f, y + 0.0f, z + 0.0f); if(nbr >= 0){d = 1; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 1.0f, z + 0.0f); if(nbr >= 0){d = 1; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 1.0f); if(nbr >= 0){d = 1; own = nbr;}}
		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void DilationSegments_Spherical_3D_Surface_Unrolled_rad_2(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にない場合(own < 0)、カーネル範囲内の最も近いsegment領域(nbr >= 0   空白の値が-1のみであればnbr != own)があればその値に変更する(own = nbr)
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own < 0) ? d : 0;
	if(z-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(y-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(y-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(x-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(x-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(x+1.0f <= DATA_W-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(x+2.0f <= DATA_W-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 4; own = nbr;}}

		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void DilationSegments_Spherical_3D_Surface_Unrolled_rad_3(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にない場合(own < 0)、カーネル範囲内の最も近いsegment領域(nbr >= 0   空白の値が-1のみであればnbr != own)があればその値に変更する(own = nbr)
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own < 0) ? d : 0;
	if(z-3.0f >= 0.0f                                                          && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 3.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f                                        && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z - 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z - 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z - 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z - 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(y-3.0f >= 0.0f                                                          && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 0.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(y-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(y-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(x-3.0f >= 0.0f                                                          && d > 9) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(x-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(x-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(x+1.0f <= DATA_W-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(x+2.0f <= DATA_W-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(x+3.0f <= DATA_W-1.0f                                                   && d > 9) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(y+3.0f <= DATA_H-1.0f                                                   && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 0.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 1; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 1.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 1.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 1.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 4; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 2.0); if(nbr >= 0){d = 5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 2.0); if(nbr >= 0){d = 6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 2.0); if(nbr >= 0){d = 8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 2.0); if(nbr >= 0){d = 9; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f                                                   && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 3.0); if(nbr >= 0){d = 9; own = nbr;}}

		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void DilationSegments_Spherical_3D_Surface_Unrolled_rad_4(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;

	//自身がsegment領域にない場合(own < 0)、カーネル範囲内の最も近いsegment領域(nbr >= 0   空白の値が-1のみであればnbr != own)があればその値に変更する(own = nbr)
	
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own < 0) ? d : 0;
	if(z-4.0f >= 0.0f                                                          && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 4.0); if(nbr >= 0){d = 16; own = nbr;}}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-3.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-3.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-3.0f >= 0.0f                                                          && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 3.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z - 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z - 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f                                        && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z - 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-2.0f >= 0.0f                                                          && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 2.0); if(nbr >= 0){d =  4; own = nbr;}}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z - 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z - 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z - 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z - 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z - 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 3.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z - 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 3.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 2.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f                      && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z - 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z - 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 2.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f                      && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f                                        && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z-1.0f >= 0.0f                                                          && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 1.0); if(nbr >= 0){d =  1; own = nbr;}}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z - 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 2.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z - 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z - 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 2.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 3.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z - 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z - 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 3.0, z - 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(y-4.0f >= 0.0f                                                          && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 4.0, z + 0.0); if(nbr >= 0){d = 16; own = nbr;}}
	if(y-3.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 3.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y-3.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(y-3.0f >= 0.0f                                                          && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 0.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 3.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y-2.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 0.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y-2.0f >= 0.0f                                                          && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 0.0); if(nbr >= 0){d =  4; own = nbr;}}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 0.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 2.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y-1.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 0.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(y-1.0f >= 0.0f                                                          && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 0.0); if(nbr >= 0){d =  1; own = nbr;}}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 0.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(x-4.0f >= 0.0f                                                          && d > 16) {nbr = tex3D(tex_VolumeI, x - 4.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 16; own = nbr;}}
	if(x-3.0f >= 0.0f                                                          && d >  9) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 0.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(x-2.0f >= 0.0f                                                          && d >  4) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 0.0); if(nbr >= 0){d =  4; own = nbr;}}
	if(x-1.0f >= 0.0f                                                          && d >  1) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 0.0); if(nbr >= 0){d =  1; own = nbr;}}
	if(x+1.0f <= DATA_W-1.0f                                                   && d >  1) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 0.0); if(nbr >= 0){d =  1; own = nbr;}}
	if(x+2.0f <= DATA_W-1.0f                                                   && d >  4) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 0.0); if(nbr >= 0){d =  4; own = nbr;}}
	if(x+3.0f <= DATA_W-1.0f                                                   && d >  9) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 0.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(x+4.0f <= DATA_W-1.0f                                                   && d > 16) {nbr = tex3D(tex_VolumeI, x + 4.0, y + 0.0, z + 0.0); if(nbr >= 0){d = 16; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 0.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f                                                   && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 0.0); if(nbr >= 0){d =  1; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 0.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 0.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f                                                   && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 0.0); if(nbr >= 0){d =  4; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 0.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 0.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 2.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 3.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(y+3.0f <= DATA_H-1.0f                                                   && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 0.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z + 0.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 3.0, z + 0.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(y+4.0f <= DATA_H-1.0f                                                   && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 4.0, z + 0.0); if(nbr >= 0){d = 16; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 3.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 3.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 2.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 2.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x-3.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f                                                   && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 1.0); if(nbr >= 0){d =  1; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 1.0); if(nbr >= 0){d =  2; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 1.0); if(nbr >= 0){d =  3; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 2.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 1.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 1.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 1.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 2.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 3.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 1.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z + 1.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 3.0, z + 1.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f               && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x-3.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f                                                   && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 2.0); if(nbr >= 0){d =  4; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 2.0); if(nbr >= 0){d =  5; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 2.0); if(nbr >= 0){d =  6; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 2.0); if(nbr >= 0){d =  8; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 2.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 2.0); if(nbr >= 0){d = 12; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 2.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z + 2.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f                                                   && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 3.0); if(nbr >= 0){d =  9; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 3.0); if(nbr >= 0){d = 10; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 3.0); if(nbr >= 0){d = 11; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 3.0); if(nbr >= 0){d = 13; own = nbr;}}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 3.0); if(nbr >= 0){d = 14; own = nbr;}}
	if(z+4.0f <= DATA_D-1.0f                                                   && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 4.0); if(nbr >= 0){d = 16; own = nbr;}}
	
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

//画像の端は収縮しない
__global__ void ErosionSegments_Spherical_3D_Surface_Unrolled_rad_1(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
	if(z-1.0f >= 0.0f		 &&	d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z - 1.0f); if(own != nbr){d = 1; own = -1;}}
	if(x-1.0f >= 0.0f		 &&	d > 1){nbr = tex3D(tex_VolumeI, x - 1.0f, y + 0.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
	if(y-1.0f >= 0.0f		 && d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y - 1.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
	if(x+1.0f <= DATA_W-1.0f && d > 1){nbr = tex3D(tex_VolumeI, x + 1.0f, y + 0.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 1.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 1.0f); if(own != nbr){d = 1; own = -1;}}
		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void ErosionSegments_Spherical_3D_Surface_Unrolled_rad_2(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
	if(z-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 2.0); if(own != nbr){d = 4; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 1.0); if(own != nbr){d = 1; own = -1;}}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(y-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(y-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(x-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(x-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(x+1.0f <= DATA_W-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(x+2.0f <= DATA_W-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 1.0); if(own != nbr){d = 1; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 2.0); if(own != nbr){d = 4; own = -1;}}

	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void ErosionSegments_Spherical_3D_Surface_Unrolled_rad_3(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
	if(z-3.0f >= 0.0f                                                          && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 3.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f                                        && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 2.0); if(own != nbr){d = 4; own = -1;}}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z - 1.0); if(own != nbr){d = 9; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z - 1.0); if(own != nbr){d = 9; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 1.0); if(own != nbr){d = 1; own = -1;}}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z - 1.0); if(own != nbr){d = 9; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z - 1.0); if(own != nbr){d = 9; own = -1;}}
	if(y-3.0f >= 0.0f                                                          && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 0.0); if(own != nbr){d = 9; own = -1;}}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 0.0); if(own != nbr){d = 8; own = -1;}}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(y-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 0.0); if(own != nbr){d = 8; own = -1;}}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(y-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(x-3.0f >= 0.0f                                                          && d > 9) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 0.0); if(own != nbr){d = 9; own = -1;}}
	if(x-2.0f >= 0.0f                                                          && d > 4) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(x-1.0f >= 0.0f                                                          && d > 1) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(x+1.0f <= DATA_W-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(x+2.0f <= DATA_W-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(x+3.0f <= DATA_W-1.0f                                                   && d > 9) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 0.0); if(own != nbr){d = 9; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 0.0); if(own != nbr){d = 1; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 0.0); if(own != nbr){d = 2; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 0.0); if(own != nbr){d = 8; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 0.0); if(own != nbr){d = 4; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 0.0); if(own != nbr){d = 5; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 0.0); if(own != nbr){d = 8; own = -1;}}
	if(y+3.0f <= DATA_H-1.0f                                                   && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 0.0); if(own != nbr){d = 9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 1.0); if(own != nbr){d = 9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 1.0); if(own != nbr){d = 9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f                                                   && d > 1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 1.0); if(own != nbr){d = 1; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 1.0); if(own != nbr){d = 2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 1.0); if(own != nbr){d = 3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 1.0); if(own != nbr){d = 9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 1.0); if(own != nbr){d = 5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 1.0); if(own != nbr){d = 6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 1.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d > 8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f                                                   && d > 4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 2.0); if(own != nbr){d = 4; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 2.0); if(own != nbr){d = 5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 2.0); if(own != nbr){d = 6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d > 8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 2.0); if(own != nbr){d = 8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 2.0); if(own != nbr){d = 9; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f                                                   && d > 9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 3.0); if(own != nbr){d = 9; own = -1;}}

	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void ErosionSegments_Spherical_3D_Surface_Unrolled_rad_4(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
	if(z-4.0f >= 0.0f                                                          && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 4.0); if(own != nbr){d = 16; own = -1;}}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z-3.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z-3.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-3.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z-3.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z-3.0f >= 0.0f                                                          && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 3.0); if(own != nbr){d =  9; own = -1;}}
	if(z-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z-3.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z-3.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z - 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z-2.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z - 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f                                        && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z - 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z - 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z-2.0f >= 0.0f                                                          && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 2.0); if(own != nbr){d =  4; own = -1;}}
	if(z-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z - 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z - 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z-2.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z - 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z - 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z-2.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z - 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 3.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f                      && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z - 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 3.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f                      && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 2.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f                      && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z - 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z - 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z - 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z-1.0f >= 0.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 2.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f                      && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f                      && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f                      && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z - 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f                                        && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z - 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f               && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z - 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z - 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z - 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z - 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z-1.0f >= 0.0f                                                          && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z - 1.0); if(own != nbr){d =  1; own = -1;}}
	if(z-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z - 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z - 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z - 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z - 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z - 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z - 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 2.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z - 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z - 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z - 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z - 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z-1.0f >= 0.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 2.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 3.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z - 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z - 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z-1.0f >= 0.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 3.0, z - 1.0); if(own != nbr){d = 14; own = -1;}}
	if(y-4.0f >= 0.0f                                                          && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 4.0, z + 0.0); if(own != nbr){d = 16; own = -1;}}
	if(y-3.0f >= 0.0f && x-2.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 3.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y-3.0f >= 0.0f && x-1.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(y-3.0f >= 0.0f                                                          && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 0.0); if(own != nbr){d =  9; own = -1;}}
	if(y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 3.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y-2.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 2.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y-2.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 0.0); if(own != nbr){d =  8; own = -1;}}
	if(y-2.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y-2.0f >= 0.0f                                                          && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 0.0); if(own != nbr){d =  4; own = -1;}}
	if(y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 0.0); if(own != nbr){d =  8; own = -1;}}
	if(y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 2.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y-1.0f >= 0.0f && x-3.0f >= 0.0f                                        && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(y-1.0f >= 0.0f && x-2.0f >= 0.0f                                        && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y-1.0f >= 0.0f && x-1.0f >= 0.0f                                        && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 0.0); if(own != nbr){d =  2; own = -1;}}
	if(y-1.0f >= 0.0f                                                          && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 0.0); if(own != nbr){d =  1; own = -1;}}
	if(y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 0.0); if(own != nbr){d =  2; own = -1;}}
	if(y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(x-4.0f >= 0.0f                                                          && d > 16) {nbr = tex3D(tex_VolumeI, x - 4.0, y + 0.0, z + 0.0); if(own != nbr){d = 16; own = -1;}}
	if(x-3.0f >= 0.0f                                                          && d >  9) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 0.0); if(own != nbr){d =  9; own = -1;}}
	if(x-2.0f >= 0.0f                                                          && d >  4) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 0.0); if(own != nbr){d =  4; own = -1;}}
	if(x-1.0f >= 0.0f                                                          && d >  1) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 0.0); if(own != nbr){d =  1; own = -1;}}
	if(x+1.0f <= DATA_W-1.0f                                                   && d >  1) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 0.0); if(own != nbr){d =  1; own = -1;}}
	if(x+2.0f <= DATA_W-1.0f                                                   && d >  4) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 0.0); if(own != nbr){d =  4; own = -1;}}
	if(x+3.0f <= DATA_W-1.0f                                                   && d >  9) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 0.0); if(own != nbr){d =  9; own = -1;}}
	if(x+4.0f <= DATA_W-1.0f                                                   && d > 16) {nbr = tex3D(tex_VolumeI, x + 4.0, y + 0.0, z + 0.0); if(own != nbr){d = 16; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 0.0); if(own != nbr){d =  2; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f                                                   && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 0.0); if(own != nbr){d =  1; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 0.0); if(own != nbr){d =  2; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 2.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 0.0); if(own != nbr){d =  8; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f                                                   && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 0.0); if(own != nbr){d =  4; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 0.0); if(own != nbr){d =  5; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 0.0); if(own != nbr){d =  8; own = -1;}}
	if(y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 2.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 3.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(y+3.0f <= DATA_H-1.0f                                                   && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 0.0); if(own != nbr){d =  9; own = -1;}}
	if(y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z + 0.0); if(own != nbr){d = 10; own = -1;}}
	if(y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 3.0, z + 0.0); if(own != nbr){d = 13; own = -1;}}
	if(y+4.0f <= DATA_H-1.0f                                                   && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 4.0, z + 0.0); if(own != nbr){d = 16; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 3.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 3.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 2.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 2.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x-3.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d >  2) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f                                                   && d >  1) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 1.0); if(own != nbr){d =  1; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  2) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  3) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d >  2) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 1.0); if(own != nbr){d =  2; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  3) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 1.0); if(own != nbr){d =  3; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d >  6) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f && d > 11) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-3.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 2.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 1.0); if(own != nbr){d =  5; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 1.0); if(own != nbr){d =  6; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 1.0); if(own != nbr){d =  9; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 2.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 3.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 1.0); if(own != nbr){d = 10; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z + 1.0); if(own != nbr){d = 11; own = -1;}}
	if(z+1.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 3.0, z + 1.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 3.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 3.0, z + 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-3.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 3.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-2.0f >= 0.0f               && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 2.0, z + 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 2.0, z + 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-3.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y - 1.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+3.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y - 1.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x-3.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 0.0, z + 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d >  8) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d >  5) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f                                                   && d >  4) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 2.0); if(own != nbr){d =  4; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d >  8) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && x+3.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 0.0, z + 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-3.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 3.0, y + 1.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  6) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d >  5) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 2.0); if(own != nbr){d =  5; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  6) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 2.0); if(own != nbr){d =  6; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d >  9) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+3.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 3.0, y + 1.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 12) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 2.0, z + 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d >  9) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d >  8) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 2.0); if(own != nbr){d =  8; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d >  9) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 2.0); if(own != nbr){d =  9; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 12) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 2.0, z + 2.0); if(own != nbr){d = 12; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 3.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 3.0, z + 2.0); if(own != nbr){d = 13; own = -1;}}
	if(z+2.0f <= DATA_D-1.0f && y+3.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 3.0, z + 2.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x-1.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 2.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 2.0, z + 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-2.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 2.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-2.0f >= 0.0f               && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y - 1.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x-1.0f >= 0.0f               && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y - 1.0, z + 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y - 1.0, z + 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+1.0f <= DATA_W-1.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y - 1.0, z + 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y-1.0f >= 0.0f && x+2.0f <= DATA_W-1.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y - 1.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && x-2.0f >= 0.0f                                 && d > 13) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 0.0, z + 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && x-1.0f >= 0.0f                                 && d > 10) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 0.0, z + 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f                                                   && d >  9) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 3.0); if(own != nbr){d =  9; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && x+1.0f <= DATA_W-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 0.0, z + 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && x+2.0f <= DATA_W-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 0.0, z + 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-2.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 2.0, y + 1.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 11) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 1.0, z + 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f                          && d > 10) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 1.0, z + 3.0); if(own != nbr){d = 10; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 11) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 1.0, z + 3.0); if(own != nbr){d = 11; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+1.0f <= DATA_H-1.0f && x+2.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 2.0, y + 1.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x-1.0f >= 0.0f        && d > 14) {nbr = tex3D(tex_VolumeI, x - 1.0, y + 2.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f                          && d > 13) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 2.0, z + 3.0); if(own != nbr){d = 13; own = -1;}}
	if(z+3.0f <= DATA_D-1.0f && y+2.0f <= DATA_H-1.0f && x+1.0f <= DATA_W-1.0f && d > 14) {nbr = tex3D(tex_VolumeI, x + 1.0, y + 2.0, z + 3.0); if(own != nbr){d = 14; own = -1;}}
	if(z+4.0f <= DATA_D-1.0f                                                   && d > 16) {nbr = tex3D(tex_VolumeI, x + 0.0, y + 0.0, z + 4.0); if(own != nbr){d = 16; own = -1;}}

	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

//画像の端も収縮する
__global__ void ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_1(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	if(x-1.0f < 0.0f || y-1.0f < 0.0f || z-1.0f < 0.0f || x+1.0f > DATA_W-1.0f || y+1.0f > DATA_H-1.0f || z+1.0f > DATA_D-1.0f)
		own = -1;
	else {
		own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
		if(d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z - 1.0f); if(own != nbr){d = 1; own = -1;}}
		if(d > 1){nbr = tex3D(tex_VolumeI, x - 1.0f, y + 0.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
		if(d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y - 1.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
		if(d > 1){nbr = tex3D(tex_VolumeI, x + 1.0f, y + 0.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
		if(d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 1.0f, z + 0.0f); if(own != nbr){d = 1; own = -1;}}
		if(d > 1){nbr = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 1.0f); if(own != nbr){d = 1; own = -1;}}
	}
		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_2(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	if(x-2.0f < 0.0f || y-2.0f < 0.0f || z-2.0f < 0.0f || x+2.0f > DATA_W-1.0f || y+2.0f > DATA_H-1.0f || z+2.0f > DATA_D-1.0f)
		own = -1;
	else{
		own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 2); if(own != nbr){d = 4; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 1); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 1); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 2); if(own != nbr){d = 4; own = -1;}
	}
		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_3(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	if(x-3.0f < 0.0f || y-3.0f < 0.0f || z-3.0f < 0.0f || x+3.0f > DATA_W-1.0f || y+3.0f > DATA_H-1.0f || z+3.0f > DATA_D-1.0f)
		own = -1;
	else{
		own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 3); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z - 2); if(own != nbr){d = 8; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z - 2); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z - 2); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z - 2); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z - 2); if(own != nbr){d = 8; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z - 2); if(own != nbr){d = 5; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 2); if(own != nbr){d = 4; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z - 2); if(own != nbr){d = 5; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z - 2); if(own != nbr){d = 8; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z - 2); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z - 2); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z - 2); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z - 2); if(own != nbr){d = 8; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z - 2); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z - 1); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z - 1); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z - 1); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z - 1); if(own != nbr){d = 5; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 1); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z - 1); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z - 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z - 1); if(own != nbr){d = 3; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z - 1); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z - 1); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z - 1); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z - 1); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 0, y - 3, z + 0); if(own != nbr){d = 9; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z + 0); if(own != nbr){d = 8; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z + 0); if(own != nbr){d = 8; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 3, y + 0, z + 0); if(own != nbr){d = 9; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 3, y + 0, z + 0); if(own != nbr){d = 9; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 0); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 0); if(own != nbr){d = 2; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z + 0); if(own != nbr){d = 8; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 0); if(own != nbr){d = 4; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z + 0); if(own != nbr){d = 5; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z + 0); if(own != nbr){d = 8; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 0, y + 3, z + 0); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z + 1); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 1); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z + 1); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 1); if(own != nbr){d = 5; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 1)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 1); if(own != nbr){d = 1; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 1); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 2)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 1); if(own != nbr){d = 2; own = -1;}
		if(d > 3)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 1); if(own != nbr){d = 3; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z + 1); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 1); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z + 1); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z + 1); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 2); if(own != nbr){d = 8; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 2); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 2); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 2); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 2); if(own != nbr){d = 8; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 2); if(own != nbr){d = 5; own = -1;}
		if(d > 4)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 2); if(own != nbr){d = 4; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 2); if(own != nbr){d = 5; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 2); if(own != nbr){d = 8; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 2); if(own != nbr){d = 6; own = -1;}
		if(d > 5)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 2); if(own != nbr){d = 5; own = -1;}
		if(d > 6)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 2); if(own != nbr){d = 6; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 8)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 2); if(own != nbr){d = 8; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z + 2); if(own != nbr){d = 9; own = -1;}
		if(d > 9)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 3); if(own != nbr){d = 9; own = -1;}
	}
		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

__global__ void ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_4(int *out, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr;
	int d = INT_MAX;
	
	//自身がsegment領域にある場合(own >= 0)、カーネル範囲内に他のsegment領域または空白(nbr != own)があれば空白とする(own = -1(blank))
	if(x-4.0f < 0.0f || y-4.0f < 0.0f || z-4.0f < 0.0f || x+4.0f > DATA_W-1.0f || y+4.0f > DATA_H-1.0f || z+4.0f > DATA_D-1.0f)
		own = -1;
	else{
		own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own >= 0) ? d : 0;
		if(d > 16)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 4); if(own != nbr){d = 16; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z - 3); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z - 3); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z - 3); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z - 3); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z - 3); if(own != nbr){d = 13; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z - 3); if(own != nbr){d = 10; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 3); if(own != nbr){d =  9; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z - 3); if(own != nbr){d = 10; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z - 3); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z - 3); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z - 3); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z - 3); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z - 3); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z - 3); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y - 3, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y - 3, z - 2); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y - 3, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z - 2); if(own != nbr){d = 12; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z - 2); if(own != nbr){d =  8; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z - 2); if(own != nbr){d = 12; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y - 1, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z - 2); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z - 2); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z - 2); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y - 1, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 3, y + 0, z - 2); if(own != nbr){d = 13; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z - 2); if(own != nbr){d =  8; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z - 2); if(own != nbr){d =  5; own = -1;}
		if(d >  4)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 2); if(own != nbr){d =  4; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z - 2); if(own != nbr){d =  5; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z - 2); if(own != nbr){d =  8; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 3, y + 0, z - 2); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y + 1, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z - 2); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z - 2); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z - 2); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y + 1, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z - 2); if(own != nbr){d = 12; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z - 2); if(own != nbr){d =  8; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z - 2); if(own != nbr){d =  9; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z - 2); if(own != nbr){d = 12; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y + 3, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y + 3, z - 2); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y + 3, z - 2); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y - 3, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y - 3, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y - 3, z - 1); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y - 3, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y - 3, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y - 2, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z - 1); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z - 1); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z - 1); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y - 2, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 3, y - 1, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z - 1); if(own != nbr){d =  3; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z - 1); if(own != nbr){d =  2; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z - 1); if(own != nbr){d =  3; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 3, y - 1, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 3, y + 0, z - 1); if(own != nbr){d = 10; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z - 1); if(own != nbr){d =  5; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z - 1); if(own != nbr){d =  2; own = -1;}
		if(d >  1)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z - 1); if(own != nbr){d =  1; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z - 1); if(own != nbr){d =  2; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z - 1); if(own != nbr){d =  5; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 3, y + 0, z - 1); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 3, y + 1, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z - 1); if(own != nbr){d =  3; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z - 1); if(own != nbr){d =  2; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z - 1); if(own != nbr){d =  3; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 3, y + 1, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y + 2, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z - 1); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z - 1); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z - 1); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z - 1); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y + 2, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y + 3, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y + 3, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y + 3, z - 1); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y + 3, z - 1); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y + 3, z - 1); if(own != nbr){d = 14; own = -1;}
		if(d > 16)  nbr = tex3D(tex_VolumeI, x + 0, y - 4, z + 0); if(own != nbr){d = 16; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 2, y - 3, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 1, y - 3, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 0, y - 3, z + 0); if(own != nbr){d =  9; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 1, y - 3, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 2, y - 3, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 3, y - 2, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z + 0); if(own != nbr){d =  8; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d >  4)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 0); if(own != nbr){d =  4; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z + 0); if(own != nbr){d =  8; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 3, y - 2, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 3, y - 1, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 0); if(own != nbr){d =  2; own = -1;}
		if(d >  1)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 0); if(own != nbr){d =  1; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 0); if(own != nbr){d =  2; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 3, y - 1, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d > 16)  nbr = tex3D(tex_VolumeI, x - 4, y + 0, z + 0); if(own != nbr){d = 16; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 3, y + 0, z + 0); if(own != nbr){d =  9; own = -1;}
		if(d >  4)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 0); if(own != nbr){d =  4; own = -1;}
		if(d >  1)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 0); if(own != nbr){d =  1; own = -1;}
		if(d >  1)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 0); if(own != nbr){d =  1; own = -1;}
		if(d >  4)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 0); if(own != nbr){d =  4; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 3, y + 0, z + 0); if(own != nbr){d =  9; own = -1;}
		if(d > 16)  nbr = tex3D(tex_VolumeI, x + 4, y + 0, z + 0); if(own != nbr){d = 16; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 3, y + 1, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 0); if(own != nbr){d =  2; own = -1;}
		if(d >  1)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 0); if(own != nbr){d =  1; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 0); if(own != nbr){d =  2; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 3, y + 1, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 3, y + 2, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z + 0); if(own != nbr){d =  8; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d >  4)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 0); if(own != nbr){d =  4; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z + 0); if(own != nbr){d =  5; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z + 0); if(own != nbr){d =  8; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 3, y + 2, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 2, y + 3, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 1, y + 3, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 0, y + 3, z + 0); if(own != nbr){d =  9; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 1, y + 3, z + 0); if(own != nbr){d = 10; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 2, y + 3, z + 0); if(own != nbr){d = 13; own = -1;}
		if(d > 16)  nbr = tex3D(tex_VolumeI, x + 0, y + 4, z + 0); if(own != nbr){d = 16; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y - 3, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y - 3, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y - 3, z + 1); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y - 3, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y - 3, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y - 2, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z + 1); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 1); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z + 1); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y - 2, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 3, y - 1, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 1); if(own != nbr){d =  3; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 1); if(own != nbr){d =  2; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 1); if(own != nbr){d =  3; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 3, y - 1, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 3, y + 0, z + 1); if(own != nbr){d = 10; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 1); if(own != nbr){d =  5; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 1); if(own != nbr){d =  2; own = -1;}
		if(d >  1)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 1); if(own != nbr){d =  1; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 1); if(own != nbr){d =  2; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 1); if(own != nbr){d =  5; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 3, y + 0, z + 1); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 3, y + 1, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 1); if(own != nbr){d =  3; own = -1;}
		if(d >  2)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 1); if(own != nbr){d =  2; own = -1;}
		if(d >  3)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 1); if(own != nbr){d =  3; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 3, y + 1, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y + 2, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z + 1); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 1); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z + 1); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z + 1); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y + 2, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y + 3, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y + 3, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y + 3, z + 1); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y + 3, z + 1); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y + 3, z + 1); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y - 3, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y - 3, z + 2); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y - 3, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x - 2, y - 2, z + 2); if(own != nbr){d = 12; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 2); if(own != nbr){d =  8; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x + 2, y - 2, z + 2); if(own != nbr){d = 12; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y - 1, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 2); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 2); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 2); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y - 1, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 3, y + 0, z + 2); if(own != nbr){d = 13; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 2); if(own != nbr){d =  8; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 2); if(own != nbr){d =  5; own = -1;}
		if(d >  4)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 2); if(own != nbr){d =  4; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 2); if(own != nbr){d =  5; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 2); if(own != nbr){d =  8; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 3, y + 0, z + 2); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 3, y + 1, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 2); if(own != nbr){d =  6; own = -1;}
		if(d >  5)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 2); if(own != nbr){d =  5; own = -1;}
		if(d >  6)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 2); if(own != nbr){d =  6; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 3, y + 1, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x - 2, y + 2, z + 2); if(own != nbr){d = 12; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d >  8)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 2); if(own != nbr){d =  8; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z + 2); if(own != nbr){d =  9; own = -1;}
		if(d > 12)  nbr = tex3D(tex_VolumeI, x + 2, y + 2, z + 2); if(own != nbr){d = 12; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y + 3, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y + 3, z + 2); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y + 3, z + 2); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y - 2, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y - 2, z + 3); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y - 2, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y - 1, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y - 1, z + 3); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y - 1, z + 3); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y - 1, z + 3); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y - 1, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x - 2, y + 0, z + 3); if(own != nbr){d = 13; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x - 1, y + 0, z + 3); if(own != nbr){d = 10; own = -1;}
		if(d >  9)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 3); if(own != nbr){d =  9; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 1, y + 0, z + 3); if(own != nbr){d = 10; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 2, y + 0, z + 3); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 2, y + 1, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x - 1, y + 1, z + 3); if(own != nbr){d = 11; own = -1;}
		if(d > 10)  nbr = tex3D(tex_VolumeI, x + 0, y + 1, z + 3); if(own != nbr){d = 10; own = -1;}
		if(d > 11)  nbr = tex3D(tex_VolumeI, x + 1, y + 1, z + 3); if(own != nbr){d = 11; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 2, y + 1, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x - 1, y + 2, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 13)  nbr = tex3D(tex_VolumeI, x + 0, y + 2, z + 3); if(own != nbr){d = 13; own = -1;}
		if(d > 14)  nbr = tex3D(tex_VolumeI, x + 1, y + 2, z + 3); if(own != nbr){d = 14; own = -1;}
		if(d > 16)  nbr = tex3D(tex_VolumeI, x + 0, y + 0, z + 4); if(own != nbr){d = 16; own = -1;}
	}
		
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

extern "C" __declspec (dllexport) void dilateSegmentsSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
){
	if(tex_volumeArray == 0 || bufferI == 0)return;

	// 32 threads along x, 16 along y
	int threadsInX = 32;
	int threadsInY = 16;
	int threadsInZ = 1;

    // Round up to get sufficient number of blocks
    int blocksInX = (int)ceil((float)imageW / (float)threadsInX);
    int blocksInY = (int)ceil((float)imageH / (float)threadsInY);
	int blocksInZ = (int)ceil((float)imageZ / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	switch(radius){
		case 1: DilationSegments_Spherical_3D_Surface_Unrolled_rad_1<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 2: DilationSegments_Spherical_3D_Surface_Unrolled_rad_2<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 3: DilationSegments_Spherical_3D_Surface_Unrolled_rad_3<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 4: DilationSegments_Spherical_3D_Surface_Unrolled_rad_4<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
	}
	checkCudaErrors( cudaDeviceSynchronize() );

	copyToSuf3DFromDeviceMem_Int(bufferI, imageW, imageH, imageZ);
}

//画像の端は収縮しない
extern "C" __declspec (dllexport) void erodeSegmentsSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
){
	if(tex_volumeArray == 0 || bufferI == 0)return;

	// 32 threads along x, 16 along y
	int threadsInX = 32;
	int threadsInY = 16;
	int threadsInZ = 1;

    // Round up to get sufficient number of blocks
    int blocksInX = (int)ceil((float)imageW / (float)threadsInX);
    int blocksInY = (int)ceil((float)imageH / (float)threadsInY);
	int blocksInZ = (int)ceil((float)imageZ / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	switch(radius){
		case 1: ErosionSegments_Spherical_3D_Surface_Unrolled_rad_1<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 2: ErosionSegments_Spherical_3D_Surface_Unrolled_rad_2<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 3: ErosionSegments_Spherical_3D_Surface_Unrolled_rad_3<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 4: ErosionSegments_Spherical_3D_Surface_Unrolled_rad_4<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
	}
	
	copyToSuf3DFromDeviceMem_Int(bufferI, imageW, imageH, imageZ);
}

//画像の端も収縮する
extern "C" __declspec (dllexport) void erodeSegmentsSurfaceSphereGPU_v2(
	int radius,
    int imageW,
    int imageH,
	int imageZ
){
	if(tex_volumeArray == 0 || bufferI == 0)return;

	// 32 threads along x, 16 along y
	int threadsInX = 32;
	int threadsInY = 16;
	int threadsInZ = 1;

    // Round up to get sufficient number of blocks
    int blocksInX = (int)ceil((float)imageW / (float)threadsInX);
    int blocksInY = (int)ceil((float)imageH / (float)threadsInY);
	int blocksInZ = (int)ceil((float)imageZ / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

	switch(radius){
		case 1: ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_1<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 2: ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_2<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 3: ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_3<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
		case 4: ErosionSegments2_Spherical_3D_Surface_Unrolled_rad_4<<<dimGrid, dimBlock>>>(bufferI, imageW, imageH, imageZ); break;
	}
	
	copyToSuf3DFromDeviceMem_Int(bufferI, imageW, imageH, imageZ);
}

extern "C" __declspec (dllexport) void initCuda_Suf3D_Int(const int *d_volume, int imageW, int imageH, int imageZ)
{
	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	checkCudaErrors(cudaMalloc3DArray(&tex_volumeArray, &channelDesc, volumeSize));
		
	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)d_volume, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = tex_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex_VolumeI.normalized = false;                      // access with normalized texture coordinates
	tex_VolumeI.filterMode = cudaFilterModePoint;       
	tex_VolumeI.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	tex_VolumeI.addressMode[1] = cudaAddressModeClamp;
	tex_VolumeI.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	//checkCudaErrors(cudaBindTextureToArray(tex_VolumeI, tex_volumeArray));
	checkCudaErrors(cudaBindTextureToArray(&tex_VolumeI, tex_volumeArray, &channelDesc));

	checkCudaErrors( cudaMalloc((void **)(&bufferI),   imageZ * imageW * imageH * sizeof(int)) );
}

extern "C" __declspec (dllexport) void copyToDeviceMemFromSuf3D_Int(int *d_output, int imageW, int imageH, int imageZ)
{
	if(tex_volumeArray == 0)return;

	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcArray = tex_volumeArray;
	copyParams.dstPtr   = make_cudaPitchedPtr((void *)d_output, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	
	checkCudaErrors(cudaMemcpy3D(&copyParams));
}

extern "C" __declspec (dllexport) void copyToSuf3DFromDeviceMem_Int(int *d_input, int imageW, int imageH, int imageZ)
{
	if(tex_volumeArray == 0)return;

	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)d_input, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = tex_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	checkCudaErrors(cudaUnbindTexture(tex_VolumeI));

	//checkCudaErrors(cudaBindTextureToArray(tex_VolumeI, tex_volumeArray));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	checkCudaErrors(cudaBindTextureToArray(&tex_VolumeI, tex_volumeArray, &channelDesc));
}


__global__ void watershed3D_kernel(int *out, float th, int DATA_W, int DATA_H, int DATA_D)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	int own, nbr, nbr_tmp;
	int d = INT_MAX;
	float own_intensity;
	
	//自身がsegment領域にない場合(own < 0)、カーネル範囲内の最も近いsegment領域(nbr >= 0   空白の値が-1のみであればnbr != own)があればその値に変更する(own = nbr)
	own = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 0.0f); d = (own < 0) ? d : 0;
	own_intensity = tex3D(tex_Volume, x + 0.0f, y + 0.0f, z + 0.0f);
	if(own_intensity <= th){
		nbr = -1;
		if(z-1.0f >= 0.0f		 &&	d > 1){nbr_tmp = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z - 1.0f); if(nbr_tmp >= 0){d = 1; own = nbr_tmp; nbr = nbr_tmp;}}
		if(x-1.0f >= 0.0f		 &&	d > 1){nbr_tmp = tex3D(tex_VolumeI, x - 1.0f, y + 0.0f, z + 0.0f); if(nbr_tmp >= 0){d = 1; own = (nbr < 0 || nbr == nbr_tmp) ? nbr_tmp : -1; nbr = nbr_tmp;}}
		if(y-1.0f >= 0.0f		 && d > 1){nbr_tmp = tex3D(tex_VolumeI, x + 0.0f, y - 1.0f, z + 0.0f); if(nbr_tmp >= 0){d = 1; own = (nbr < 0 || nbr == nbr_tmp) ? nbr_tmp : -1; nbr = nbr_tmp;}}
		if(x+1.0f <= DATA_W-1.0f && d > 1){nbr_tmp = tex3D(tex_VolumeI, x + 1.0f, y + 0.0f, z + 0.0f); if(nbr_tmp >= 0){d = 1; own = (nbr < 0 || nbr == nbr_tmp) ? nbr_tmp : -1; nbr = nbr_tmp;}}
		if(y+1.0f <= DATA_H-1.0f && d > 1){nbr_tmp = tex3D(tex_VolumeI, x + 0.0f, y + 1.0f, z + 0.0f); if(nbr_tmp >= 0){d = 1; own = (nbr < 0 || nbr == nbr_tmp) ? nbr_tmp : -1; nbr = nbr_tmp;}}
		if(z+1.0f <= DATA_D-1.0f && d > 1){nbr_tmp = tex3D(tex_VolumeI, x + 0.0f, y + 0.0f, z + 1.0f); if(nbr_tmp >= 0){d = 1; own = (nbr < 0 || nbr == nbr_tmp) ? nbr_tmp : -1; nbr = nbr_tmp;}}
	}
	//surf3Dwrite(own, _surface2, (x + 0)*4, y + 0, z + 0, cudaBoundaryModeTrap);
	out[z*DATA_H*DATA_W + y*DATA_W + x] = own;
}

extern "C" __declspec (dllexport) void watershed3dGPU(
	float th,
    int imageW,
    int imageH,
	int imageZ
){
	if(tex_volumeArray == 0 || tex_volumeArray2 == 0 || bufferI == 0)return;

	// 32 threads along x, 16 along y
	int threadsInX = 32;
	int threadsInY = 16;
	int threadsInZ = 1;

    // Round up to get sufficient number of blocks
    int blocksInX = (int)ceil((float)imageW / (float)threadsInX);
    int blocksInY = (int)ceil((float)imageH / (float)threadsInY);
	int blocksInZ = (int)ceil((float)imageZ / (float)threadsInZ);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);
	
	watershed3D_kernel<<<dimGrid, dimBlock>>>(bufferI, th, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );

	copyToSuf3DFromDeviceMem_Int(bufferI, imageW, imageH, imageZ);
}

extern "C" __declspec (dllexport) void initCuda_Watershed(const int *h_seed, const float *h_src, int imageW, int imageH, int imageZ)
{
	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMalloc3DArray(&tex_volumeArray2, &channelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_src, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = tex_volumeArray2;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex_Volume.normalized = false;                      // access with normalized texture coordinates
	tex_Volume.filterMode = cudaFilterModePoint;       
	tex_Volume.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	tex_Volume.addressMode[1] = cudaAddressModeClamp;
	tex_Volume.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex_Volume, tex_volumeArray2));

	channelDesc = cudaCreateChannelDesc<int>();
	checkCudaErrors(cudaMalloc3DArray(&tex_volumeArray, &channelDesc, volumeSize));
	
	// copy data to 3D array
	cudaMemcpy3DParms copyParams_suf = {0};
	copyParams_suf.srcPtr   = make_cudaPitchedPtr((void *)h_seed, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams_suf.dstArray = tex_volumeArray;
	copyParams_suf.extent   = volumeSize;
	copyParams_suf.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams_suf));

	// set texture parameters
	tex_VolumeI.normalized = false;                      // access with normalized texture coordinates
	tex_VolumeI.filterMode = cudaFilterModePoint;       
	tex_VolumeI.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	tex_VolumeI.addressMode[1] = cudaAddressModeClamp;
	tex_VolumeI.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex_VolumeI, tex_volumeArray));

	checkCudaErrors( cudaMalloc((void **)(&bufferI),   imageZ * imageW * imageH * sizeof(int)) );
}

extern "C" __declspec (dllexport) void destructCuda_Watershed()
{
	if(tex_volumeArray)checkCudaErrors(cudaFreeArray(tex_volumeArray));
	tex_volumeArray = 0;
	
	if(tex_volumeArray2)checkCudaErrors(cudaFreeArray(tex_volumeArray2));
	tex_volumeArray2 = 0;

	if(bufferI)checkCudaErrors( cudaFree(bufferI) );
	bufferI = 0;
}

extern "C" __declspec (dllexport) void copyToHostMemFromSuf3D_Int(int *h_output, int imageW, int imageH, int imageZ)
{
	if(tex_volumeArray == 0)return;

	cudaExtent volumeSize;
	volumeSize.width  = imageW;
	volumeSize.height = imageH;
	volumeSize.depth  = imageZ;

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcArray = tex_volumeArray;
	copyParams.dstPtr   = make_cudaPitchedPtr((void *)h_output, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyDeviceToHost;
	
	checkCudaErrors(cudaMemcpy3D(&copyParams));
}

__global__ void segmentsMaskInt_kernel(int *dst, const int *A, const int *B, int imageW, int pitchY, int pitchZ)
{
    const int id = blockIdx.z * pitchZ + blockIdx.y * pitchY + blockIdx.x * blockDim.x + threadIdx.x;

    //dst[i] = ((A[i] >= 0) != (B[i] >= 0)) ? ((A[i] >= 0) ? A[i] : B[i]) : -1;
	if(blockIdx.x * blockDim.x + threadIdx.x < imageW)dst[id] = ((A[id] >= 0) && (B[id] < 0)) ? A[id] : -1;

}

extern "C" __declspec (dllexport) void segmentsMaskIntGPU(
    int *d_Dst,
	int *d_Src,
	int *d_Mask,
	int imageW,
    int imageH,
	int imageZ
){
	dim3 blocks((imageW + BINALIZE_BLOCKDIM - 1)/BINALIZE_BLOCKDIM, imageH, imageZ);
    dim3 threads(BINALIZE_BLOCKDIM);
	
	segmentsMaskInt_kernel<<<blocks, threads>>>(d_Dst, d_Src, d_Mask, imageW, imageW, imageW*imageH);

};

extern "C" __declspec (dllexport) void segmentsSumGPU(
    int *d_Out,
	int *d_Src,
	int n
){
	int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;//celi( n / threadsPerBlock )
};

__global__ void generateGridDataIntKernel(
	int *d_Dst,
	int *d_Src,
	int srcPitchY,
	int srcPitchZ,
	int cellPitchY,
	int cellPitchZ,
	int cellSize,
	int gridPitchX,
	int gridPitchY,
	int gridPitchZ,
	int gridSize,
	float intervalX,
	float intervalY,
	float intervalZ,
	int offsetX,
	int offsetY,
	int offsetZ
){
	unsigned int i = threadIdx.x;
	unsigned int dstId = i + blockIdx.x*gridPitchX + blockIdx.y*gridPitchY + blockIdx.z*gridPitchZ;
	unsigned int src_baseX = offsetX + lrintf(blockIdx.x*intervalX);
	unsigned int src_baseY = offsetY + lrintf(blockIdx.y*intervalY);
	unsigned int src_baseZ = offsetZ + lrintf(blockIdx.z*intervalZ);

	while(i < cellSize){
		unsigned int cellz = i / cellPitchZ;
		unsigned int celly = (i - cellz*cellPitchZ) / cellPitchY;
		unsigned int cellx = (i - cellz*cellPitchZ - celly*cellPitchY);

		d_Dst[dstId] = d_Src[(src_baseZ + cellz)*srcPitchZ + (src_baseY + celly)*srcPitchY + src_baseX + cellx];

		i += blockDim.x;
		dstId += blockDim.x;
	}
}

extern "C" __declspec (dllexport) void generateGridDataIntGPU(
    int * &d_Dst,
	int &gridsize,
	bool allocation,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int levelX,
	int levelY,
	int levelZ,
	int offsetX,
	int offsetY,
	int offsetZ
){
	int base_cellnumX = (imageW + cellW - 1) / cellW; //ceil(imageW/cellW)
	int base_cellnumY = (imageH + cellH - 1) / cellH; //ceil(imageH/cellH)
	int base_cellnumZ = (imageZ + cellZ - 1) / cellZ; //ceil(imageZ/cellZ)

	int cellnumX = (base_cellnumX - 1)*levelX + 1;
	int cellnumY = (base_cellnumY - 1)*levelY + 1;
	int cellnumZ = (base_cellnumZ - 1)*levelZ + 1;

	//領域外アクセス防止
	cellnumX -= (offsetX + cellW - 1) / cellW; //ceil(offsetX/cellW)
	cellnumY -= (offsetY + cellH - 1) / cellH; //ceil(offsetY/cellH)
	cellnumZ -= (offsetZ + cellZ - 1) / cellZ; //ceil(offsetZ/cellZ)

	//printf("CellNum X: %d  Y: %d  Z: %d\n", cellnumX, cellnumY, cellnumZ);
	
	float intervalX = (float)(imageW - cellW) / (float)(cellnumX - 1);
	float intervalY = (float)(imageH - cellH) / (float)(cellnumY - 1);
	float intervalZ = (float)(imageZ - cellZ) / (float)(cellnumZ - 1);

	//printf("Interval X: %f  Y: %f  Z: %f\n", intervalX, intervalY, intervalZ);

	gridsize = cellnumX * cellnumY * cellnumZ * cellW * cellH * cellZ;
	if(allocation)checkCudaErrors( cudaMalloc((void **)(&d_Dst), gridsize * sizeof(int)) );

	//printf("GridSize: %d\n", gridsize);

	dim3 blocks(cellnumX, cellnumY, cellnumZ);
    dim3 threads(256);

	generateGridDataIntKernel<<<blocks, threads>>>(
		d_Dst,
		d_Src,
		imageW,
		imageW*imageH,
		cellW,
		cellW*cellH,
		cellW*cellH*cellZ,
		cellW*cellH*cellZ,
		cellW*cellH*cellZ*cellnumX,
		cellW*cellH*cellZ*cellnumX*cellnumY,
		gridsize,
		intervalX,
		intervalY,
		intervalZ,
		offsetX,
		offsetY,
		offsetZ
	);
}

extern "C" __declspec (dllexport) void thresholdIntGPU(
    int *d_Dst,
	int *d_Src,
	int th,
	int n
);

extern "C" __declspec (dllexport) void axpyIntGPU(
    int *d_Dst,
	int *d_Src1,
	int *d_Src2,
	int alpha1,
	int n
);

extern "C" __declspec (dllexport) void VoxelNumberingGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void dilateSphereGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int radius,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void erodeSphereGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int radius,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void dilateSegmentsSphereGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int radius,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void erodeSegmentsSphereGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int radius,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void sumGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void maxGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void minGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void IsSingleSegmentGridDataIntGPU(
    int &result,
	int *d_Src,
	int min,
	int max,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void extractInnerStructureGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int *d_Bound,
	int dilation_radius,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ,
	int *cellList
);

extern "C" __declspec (dllexport) void copyGridDataIntGPU(
    int *d_Dst,
	int *d_Src,
	int *d_Mask,
	int dilation_radius,
	int imageW,
    int imageH,
	int imageZ,
	int cellW,
	int cellH,
	int cellZ
);

extern "C" __declspec (dllexport) void dilateSegmentsCube1IntGPU(
    int *d_Dst,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ
){
	
};