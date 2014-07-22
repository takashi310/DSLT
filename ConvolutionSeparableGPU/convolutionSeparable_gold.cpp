/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "stdafx.h"
#define DLL_CONVOLUTIONSEPARABLE
#include "convolutionSeparable_common.h"
#include <cmath>
#include <cstdio>
#define PI 3.141592653589793238462

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec (dllexport) void convolutionRowCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
    int kernelR
){
    for(int z = 0; z < imageZ; z++)
		for(int y = 0; y < imageH; y++){
		    for(int x = 0; x < imageW; x++){
		        float sum = 0;
		        for(int k = -kernelR; k <= kernelR; k++){
		            int d = x + k;
					if(d < 0)d = 0;
					if(d >= imageW)d = imageW - 1;
		            sum += h_Src[z * imageW * imageH + y * imageW + d] * h_Kernel[kernelR - k];
		        }
		        h_Dst[z * imageW * imageH + y * imageW + x] = sum;
		    }
		}
}



////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec (dllexport) void convolutionColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
    int kernelR
){
    for(int z = 0; z < imageZ; z++)
		for(int y = 0; y < imageH; y++){
		    for(int x = 0; x < imageW; x++){
		        float sum = 0;
		        for(int k = -kernelR; k <= kernelR; k++){
		            int d = y + k;
					if(d < 0)d = 0;
					if(d >= imageH)d = imageH - 1;
		            sum += h_Src[z * imageW * imageH + d * imageW + x] * h_Kernel[kernelR - k];
		        }
		        h_Dst[z * imageW * imageH + y * imageW + x] = sum;
		    }
		}
}


////////////////////////////////////////////////////////////////////////////////
// Reference Zcolumn convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec (dllexport) void convolutionZColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
    int kernelR
){
    for(int z = 0; z < imageZ; z++)
		for(int y = 0; y < imageH; y++){
		    for(int x = 0; x < imageW; x++){
		        float sum = 0;
		        for(int k = -kernelR; k <= kernelR; k++){
		            int d = z + k;
					if(d < 0)d = 0;
					if(d >= imageZ)d = imageZ - 1;
	                sum += h_Src[d * imageW * imageH + y * imageW + x] * h_Kernel[kernelR - k];
		        }
		        h_Dst[z * imageW * imageH + y * imageW + x] = sum;
		    }
		}
}


extern "C" __declspec (dllexport) void binarizeCPU(
    float *h_Dst,
	float *h_Gau,
    float *h_Src,
    int imageSize,
	float constC
){
    for(int i = 0; i < imageSize; i++){
		h_Dst[i] = (h_Src[i] > h_Gau[i] - constC) ? BINALIZE_UPPER_VAL : BINALIZE_LOWER_VAL;
		//h_Dst[i] = h_Gau[i];
	}
}


extern "C" __declspec (dllexport) void thresholdCPU(
    float *h_Dst,
	float *h_Src,
    int imageSize,
	float th
){
    for(int i = 0; i < imageSize; i++){
		h_Dst[i] = (th > h_Src[i]) ? 0.0f : 1.0f;
	}
}

double bilinearInterpolation(double a, double b, double c, double d, double px, double py)
{
	return px*py*(a-b-c+d) + px*(b-a) + py*(c-a) + a;
}
/*
extern "C" __declspec (dllexport) void allAngleLinearConvolutionMin2DCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
	int kernelR,
	int angle_d,
	float verticalC,
	float pararelC
){
	double a_interval = (PI / 2.0) / (double)angle_d;
	float sum;
	double rx, rz, val;
	int nx, nz;

	long count = imageZ*imageH*imageW/10L;
	long current = 0L;
	
	for(int z = 0; z < imageZ; z++)
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				//radian == 0
				sum = 0;
				for(int k = -kernelR; k <= kernelR; k++){
					int d = x + k;
					if(d < 0)d = 0;
					if(d >= imageW)d = imageW - 1;
					sum += h_Src[z * imageW * imageH + y * imageW + d] * h_Kernel[kernelR - k];
				}
				sum -= pararelC;
				h_Dst[z * imageW * imageH + y * imageW + x] = sum;

				//0 < radian < PI/2
				for(int a = 1; a < angle_d; a++){
					sum = 0;
					for(int k = -kernelR; k <= kernelR; k++){
						rx = x + k * cos(a_interval*a);
						rz = z + k * sin(a_interval*a);

						if(rx < 0)rx = 0;
						if(rx >= imageW)rx = imageW - 1;
						if(rz < 0)rz = 0;
						if(rz >= imageZ)rz = imageZ - 1;

						nx = (int)rx;
						nz = (int)rz;

						if(nx == imageW - 1)nx--;
						if(nz == imageZ - 1)nz--;

						val = bilinearInterpolation(h_Src[nz * imageW * imageH + y * imageW + nx],
							h_Src[nz * imageW * imageH + y * imageW + nx + 1],
							h_Src[(nz + 1) * imageW * imageH + y * imageW + nx],
							h_Src[(nz + 1) * imageW * imageH + y * imageW + nx + 1],
							rx - nx,
							rz - nz);

						sum += (float)(val * h_Kernel[kernelR - k]);
					}
					sum -= pararelC * (1.0f - (float)a/(float)angle_d) + verticalC * (float)a/(float)angle_d;
					if(h_Dst[z * imageW * imageH + y * imageW + x] > sum)h_Dst[z * imageW * imageH + y * imageW + x] = sum;
				}

				//radian == PI/2
				sum = 0;
				for(int k = -kernelR; k <= kernelR; k++){
					int d = z + k;
					if(d < 0)d = 0;
					if(d >= imageZ)d = imageZ - 1;
					sum += h_Src[d * imageW * imageH + y * imageW + x] * h_Kernel[kernelR - k];
				}
				sum -= verticalC;
				if(h_Dst[z * imageW * imageH + y * imageW + x] > sum)h_Dst[z * imageW * imageH + y * imageW + x] = sum;

				//PI/2 < radian < PI
				for(int a = 1; a < angle_d; a++){
					sum = 0;
					for(int k = -kernelR; k <= kernelR; k++){
						rx = x + k * cos(PI/2.0 + a_interval*a);
						rz = z + k * sin(PI/2.0 + a_interval*a);

						if(rx < 0)rx = 0;
						if(rx >= imageW)rx = imageW - 1;
						if(rz < 0)rz = 0;
						if(rz >= imageZ)rz = imageZ - 1;

						nx = (int)rx;
						nz = (int)rz;

						if(nx == imageW - 1)nx--;
						if(nz == imageZ - 1)nz--;

						val = bilinearInterpolation(h_Src[nz * imageW * imageH + y * imageW + nx],
							h_Src[nz * imageW * imageH + y * imageW + nx + 1],
							h_Src[(nz + 1) * imageW * imageH + y * imageW + nx],
							h_Src[(nz + 1) * imageW * imageH + y * imageW + nx + 1],
							rx - nx,
							rz - nz);

						sum += (float)(val * h_Kernel[kernelR - k]);
					}
					sum -= verticalC * (1.0f - (float)a/(float)angle_d) + pararelC * (float)a/(float)angle_d;
					if(h_Dst[z * imageW * imageH + y * imageW + x] > sum)h_Dst[z * imageW * imageH + y * imageW + x] = sum;
				}

				current++;
				if(current == count){
					current = 0L;
					putchar('>');
				}
			}
		}
		putchar('\n');
}
*/

extern "C" __declspec (dllexport) void allAngleLinearConvolutionMin2DCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
	int kernelR,
	int angle_d,
	float verticalC,
	float pararelC
){
	double a_interval = (PI / 2.0) / (double)angle_d;
	float sum;
	double rx, ry, rz, fx, fy, fz, drx, dry, drz, val;
	int ix, iy, iz;

	long count = imageZ*imageH*imageW/10L;
	long current = 0L;
	
	for(int z = 0; z < imageZ; z++)
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				//z-axis
				sum = 0;
				for(int k = -kernelR; k <= kernelR; k++){
					int d = z + k;
					if(d < 0)d = 0;
					if(d >= imageZ)d = imageZ - 1;
					sum += h_Src[d * imageW * imageH + y * imageW + x] * h_Kernel[kernelR - k];
				}
				if(x == 0 && y == 0 && z == 0)printf("Y_rot: %f  Z_rot: %f  Angle(XY_Plane): %f  C: %f\n", 0.0, 0.0, 0.5, verticalC);
				sum -= verticalC;
				h_Dst[z * imageW * imageH + y * imageW + x] = sum;

				// a: y-axis b: z-axis
				//0 < radian < PI
				for(int b = 0; b < angle_d*2; b++){
					for(int a = 1; a < angle_d*2; a++){
						sum = 0;
						drx = sin(a_interval*a) * cos(a_interval*b);
						dry = sin(a_interval*a) * sin(a_interval*b);
						drz = cos(a_interval*a);
						for(int k = -kernelR; k <= kernelR; k++){
							rx = x + k * drx;
							ry = y + k * dry;
							rz = z + k * drz;

							if(rx < 0)rx = 0;
							if(rx >= imageW)rx = imageW - 1;
							if(ry < 0)ry = 0;
							if(ry >= imageH)ry = imageH - 1;
							if(rz < 0)rz = 0;
							if(rz >= imageZ)rz = imageZ - 1;

							ix = (int)rx;
							iy = (int)ry;
							iz = (int)rz;

							if(ix == imageW - 1)ix--;
							if(iy == imageH - 1)iy--;
							if(iz == imageZ - 1)iz--;

							fx = rx - ix;
							fy = ry - iy;
							fz = rz - iz;

							float c000 = h_Src[iz*imageH*imageW + iy*imageW + ix];
							float c100 = h_Src[iz*imageH*imageW + iy*imageW + (ix + 1)];
							float c010 = h_Src[iz*imageH*imageW + (iy + 1)*imageW + ix];
							float c001 = h_Src[(iz + 1)*imageH*imageW + iy*imageW + ix];
							float c101 = h_Src[(iz + 1)*imageH*imageW + iy*imageW + (ix + 1)];
							float c011 = h_Src[(iz + 1)*imageH*imageW + (iy + 1)*imageW + ix];
							float c110 = h_Src[iz*imageH*imageW + (iy + 1)*imageW + (ix + 1)];
							float c111 = h_Src[(iz + 1)*imageH*imageW + (iy + 1)*imageW + (ix + 1)];
							val = c000*(1.0f-fx)*(1.0f-fy)*(1.0f-fz) + 
								c100*fx*(1.0f-fy)*(1.0f-fz) + 
								c010*(1.0f-fx)*fy*(1.0f-fz) +
								c001*(1.0f-fx)*(1.0f-fy)*fz +
								c101*fx*(1.0f-fy)*fz +
								c011*(1.0f-fx)*fy*fz +
								c110*fx*fy*(1.0f-fz) +
								c111*fx*fy*fz;

							sum += (float)(val * h_Kernel[kernelR - k]);
						}
						//xy-plane‚É‘Î‚·‚éŠp“x‚©‚çpararelC‚ÆverticalC‚ÌŠñ—^‚Ì“x‡‚¢‚ðŒˆ’è
						if(x == 0 && y == 0 && z == 0){
							printf("Y_rot: %f  Z_rot: %f  Angle(XY_Plane): %f  C: %f\n", a*a_interval/PI, b*a_interval/PI, acos(sqrt(drx*drx + dry*dry)/sqrt(drx*drx + dry*dry + drz*drz)) / PI, pararelC*(1.0 - abs(a*a_interval*2.0/PI - 1.0)) + verticalC * abs(a*a_interval*2.0/PI - 1.0) );
						}
						//sum -= pararelC * (1.0 - acos(sqrt(drx*drx + dry*dry)/sqrt(drx*drx + dry*dry + drz*drz)) / (PI/2) ) + verticalC * ( acos(sqrt(drx*drx + dry*dry)/sqrt(drx*drx + dry*dry + drz*drz)) / (PI/2) );
						sum -= pararelC*(1.0 - abs(a*a_interval*2.0/PI - 1.0)) + verticalC * abs(a*a_interval*2.0/PI - 1.0);
						if(h_Dst[z * imageW * imageH + y * imageW + x] > sum)h_Dst[z * imageW * imageH + y * imageW + x] = sum;
					}
				}
				
				current++;
				if(current == count){
					current = 0L;
					putchar('>');
				}
			}
		}
		putchar('\n');
}