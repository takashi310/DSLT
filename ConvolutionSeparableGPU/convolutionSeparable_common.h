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
 
/* The extrapolation method used in this program is border replication. */

#ifndef CONVOLUTIONSEPARABLE_COMMON_H
#define CONVOLUTIONSEPARABLE_COMMON_H

#ifdef DLL_CONVOLUTIONSEPARABLE
#define DECLSPEC_DLLPORT	__declspec(dllexport)
#else
#define DECLSPEC_DLLPORT	__declspec(dllimport)
#endif

#define BINALIZE_BLOCKDIM 32
#define BINALIZE_UPPER_VAL 0.8f
#define BINALIZE_LOWER_VAL 0.0f

#define GRID_OUTSIDE -42

DECLSPEC_DLLPORT void CuDeviceProp(void);

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" DECLSPEC_DLLPORT void convolutionRowCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
    int kernelR
);

extern "C" DECLSPEC_DLLPORT void convolutionColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
    int kernelR
);

extern "C" DECLSPEC_DLLPORT void convolutionZColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
	int imageZ,
    int kernelR
);

extern "C" DECLSPEC_DLLPORT void binarizeCPU(
    float *h_Dst,
	float *h_Gau,
    float *h_Src,
    int imageSize,
	float constC
);

extern "C" DECLSPEC_DLLPORT void thresholdCPU(
    float *h_Dst,
	float *h_Src,
    int imageSize,
	float th
);

extern "C" DECLSPEC_DLLPORT void allAngleLinearConvolutionMin2DCPU(
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
);


////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" DECLSPEC_DLLPORT void setConvolutionKernel8(float *h_Kernel, int kernel_radius);

extern "C" DECLSPEC_DLLPORT void convolutionRowsGPU8(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionColumnsGPU8(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionZColumnsGPU8(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void setConvolutionKernel16(float *h_Kernel, int kernel_radius);

extern "C" DECLSPEC_DLLPORT void convolutionRowsGPU16(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionColumnsGPU16(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionZColumnsGPU16(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void setConvolutionKernel32(float *h_Kernel, int kernel_radius);

extern "C" DECLSPEC_DLLPORT void convolutionRowsGPU32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionColumnsGPU32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionZColumnsGPU32(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void setConvolutionKernel64(float *h_Kernel, int kernel_radius);

extern "C" DECLSPEC_DLLPORT void convolutionRowsGPU64(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionColumnsGPU64(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void convolutionZColumnsGPU64(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void binarizeGPU(
    float *d_Dst,
	float *d_Gau,
    float *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	float constC
);

extern "C" DECLSPEC_DLLPORT void thresholdGPU(
    float *d_Dst,
	float *d_Src,
	int imageW,
    int imageH,
	int imageZ,
	float th
);

extern "C" DECLSPEC_DLLPORT void pixelwiseOrGPU(
    float *d_Dst,
	float *d_Src1,
    float *d_Src2,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void pixelwiseAndGPU(
    float *d_Dst,
	float *d_Src1,
    float *d_Src2,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void depthMapGPU(
    float *d_dmp,
	float *d_hmp,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void maximumFilterRows1GPU(
    float *d_Dst,
	float *d_Src,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void maximumFilterColumns1GPU(
    float *d_Dst,
	float *d_Src,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void maximumFilterZColumns1GPU(
    float *d_Dst,
	float *d_Src,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void maximumFilterRows1ROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void maximumFilterColumns1ROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void maximumFilterZColumns1ROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void minimumFilterRows1ROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void minimumFilterColumns1ROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void minimumFilterZColumns1ROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void upperMaskFittingGPU(
    float *d_Dst,
    float *d_Mask,
	float *d_Src,
	float mask_offset,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void lowerMaskFittingGPU(
    float *d_Dst,
    float *d_Mask,
	float *d_Src,
	float mask_offset,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void inverseGPU(
    float *d_Dst,
	float *d_Src,
	float maxval,
	float minval,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void maximumFilterRows1GPU_vHGW(
    float *d_Dst,
	float *d_Src,
	int imageW,
    int imageH,
	int imageZ
);


extern "C" DECLSPEC_DLLPORT void setSphereMaskKernel6(int radius);

extern "C" DECLSPEC_DLLPORT void maximumSphereFilterGPU6(
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
);

extern "C" DECLSPEC_DLLPORT void minimumSphereFilterGPU6(
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
);

extern "C" DECLSPEC_DLLPORT void setPointList_LCT(float *h_PointListAndWeight, int list_length);
extern "C" DECLSPEC_DLLPORT void destructCuda_LCT();
extern "C" DECLSPEC_DLLPORT void initCuda_LCT(const float *h_volume, int imageW, int imageH, int imageZ);

extern "C" DECLSPEC_DLLPORT void linearConvolutionMinGPU_Texture(
    float *d_Dst,
    int imageW,
    int imageH,
	int imageZ,
	float *d_ArgMin_lati,
	float lati,
	float *d_ArgMin_longi,
	float longi
);

extern "C" DECLSPEC_DLLPORT void lineSegmentConvolutionMinGPU(
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
);

extern "C" DECLSPEC_DLLPORT void CircleConvolutionMinGPU(
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
);

extern "C" DECLSPEC_DLLPORT void calcCorrectionTerms_DSADTH_GPU(
    float *d_Dst,
	float *d_ArgMin_lati,
	float constC_XY,
	float constC_Z,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void countUpN_LessThan_GPU(
    float *d_Dst,
	const float *d_Src,
	const float *d_T,
	float *d_N,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void fillFloatGPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	float val
);

extern "C" DECLSPEC_DLLPORT void fillFloatROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void fillIntGPU(
    int *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	int val
);

extern "C" DECLSPEC_DLLPORT void addFloatGPU(
    float *d_Dst,
	float *d_Src1,
	float *d_Src2,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void subtractFloatROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void divFloatGPU(
    float *d_Dst,
	float *d_Numer,
	float *d_Denom,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void addFloatValueGPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	float val
);

extern "C" DECLSPEC_DLLPORT void setMinValIntGPU(
    int *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	int val
);

extern "C" DECLSPEC_DLLPORT void setMinValFloatGPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ,
	float val
);

extern "C" DECLSPEC_DLLPORT void cropFloatGPU(
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
);

extern "C" DECLSPEC_DLLPORT void cropIntGPU(
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
);

extern "C" DECLSPEC_DLLPORT void copySingleSegmentIntOffsetGPU(
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
);

extern "C" DECLSPEC_DLLPORT void copySingleSegmentROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void overwriteSingleSegmentROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void replacementIntROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void replacementFloatROIGPU(
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
);



extern "C" DECLSPEC_DLLPORT void setValToSegmentedRegionROIGPU(
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
);

extern "C" DECLSPEC_DLLPORT void generateGridDataIntGPU(
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
);

extern "C" DECLSPEC_DLLPORT void thresholdIntGPU(
    int *d_Dst,
	int *d_Src,
	int th,
	int n
);

extern "C" DECLSPEC_DLLPORT void axpyIntGPU(
    int *d_Dst,
	int *d_Src1,
	int *d_Src2,
	int alpha1,
	int n
);

extern "C" DECLSPEC_DLLPORT void VoxelNumberingGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void dilateSphereGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void erodeSphereGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void dilateSegmentsSphereGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void erodeSegmentsSphereGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void sumGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void maxGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void minGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void IsSingleSegmentGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void extractInnerStructureGridDataIntGPU(
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

extern "C" DECLSPEC_DLLPORT void copyGridDataIntGPU(
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


extern "C" DECLSPEC_DLLPORT void dilateSegmentsCube1IntGPU(
    int *d_Dst,
	int *d_Src,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void dilateTextureCube1GPU(
    float *d_Dst,
	int imageW,
    int imageH,
	int imageZ
);
extern "C" DECLSPEC_DLLPORT void setSphereFilter3D_Tex3D(int rad);
extern "C" DECLSPEC_DLLPORT void destructCuda_Tex3D();
extern "C" DECLSPEC_DLLPORT void initCuda_Tex3D(const float *h_volume, int imageW, int imageH, int imageZ);


extern "C" DECLSPEC_DLLPORT void dilateSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void erodeSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void destructCuda_Suf3D();
extern "C" DECLSPEC_DLLPORT void initCuda_Suf3D(const float *d_volume, int imageW, int imageH, int imageZ);
extern "C" DECLSPEC_DLLPORT void copyToDeviceMemFromSuf3D(float *d_output, int imageW, int imageH, int imageZ);
extern "C" DECLSPEC_DLLPORT void copyToSuf3DFromDeviceMem(float *d_input, int imageW, int imageH, int imageZ);

extern "C" DECLSPEC_DLLPORT void dilateSegmentsSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void erodeSegmentsSurfaceSphereGPU(
	int radius,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void erodeSegmentsSurfaceSphereGPU_v2(
	int radius,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void initCuda_Suf3D_Int(const int *d_volume, int imageW, int imageH, int imageZ);
extern "C" DECLSPEC_DLLPORT void copyToDeviceMemFromSuf3D_Int(int *d_output, int imageW, int imageH, int imageZ);
extern "C" DECLSPEC_DLLPORT void copyToSuf3DFromDeviceMem_Int(int *d_input, int imageW, int imageH, int imageZ);

extern "C" DECLSPEC_DLLPORT void watershed3dGPU(
	float th,
    int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void initCuda_Watershed(const int *h_seed, const float *h_src, int imageW, int imageH, int imageZ);
extern "C" DECLSPEC_DLLPORT void destructCuda_Watershed();
extern "C" DECLSPEC_DLLPORT void copyToHostMemFromSuf3D_Int(int *h_output, int imageW, int imageH, int imageZ);

/*(segment, blank) is segmentID, (blank, segment) is segmentID, (blank, blank) is 0, (segment, segment) is 0*/

/*(Src, Mask)::  (segment, blank) is segmentID, (blank, segment) is blank, (blank, blank) is blank, (segment, segment) is blank*/
extern "C" DECLSPEC_DLLPORT void segmentsMaskIntGPU(
    int *d_Dst,
	int *d_Src,
	int *d_Mask,
	int imageW,
    int imageH,
	int imageZ
);

extern "C" DECLSPEC_DLLPORT void segmentsSumGPU(
    int *d_Out,
	int *d_Src,
	int n
);

extern "C" DECLSPEC_DLLPORT void notEqualIntGPU(
    float *d_Dst,
	int *d_Src1,
	int *d_Src2,
	int imageW,
    int imageH,
	int imageZ
);

#endif
