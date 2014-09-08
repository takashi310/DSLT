// これは メイン DLL ファイルです。

#include "stdafx.h"
#include <msclr/marshal.h>
#include <stdio.h>
#include <cmath>
#include "3DFilter_CLR_Interface.h"

using namespace System;
using namespace System::Drawing;
using namespace TestCVclass;
using namespace TestGPUclass;
using namespace cv;
using namespace msclr::interop;
using namespace std;

Class1::Class1()
{
	isEmpty = true;
	flt3d = new Filter3D();
	currentX = 0;
	currentY = 0;
	currentZ = 0;
	selection_copy_src = -1;
	selection_display_mode = 0;
	/*hmapoffset = 0.0f;
	hmap_depth = 0.0f;
	hmap_range = 0.0f;
	hmap_isVisible = false;
	hmap_use = false;
	proj_mode = C1_NORMAL_PROJECTION;
	proj_th = 0.0f;
	dc_isEnable = false;*/
}

Class1::~Class1()
{
	this->!Class1();
}

Class1::!Class1()
{
	if(!isEmpty){
	}
	delete flt3d;
	isEmpty = true;
}


bool Class1::set3DImage_MultiTIFF(System::String ^filename, int channel, int scalingType)
{
	const char *c_str;
	marshal_context^ ctx = gcnew marshal_context();
	c_str = ctx->marshal_as<const char *>(filename);
	flt3d->set3DImage_MultiTIFF(c_str, channel, scalingType);

	isEmpty = false;

	return true;
}

void Class1::setScalingType(int type)
{
	flt3d->switchZScaling(type);
}

void Class1::setChannel(int ch)
{
	flt3d->switchChannel(ch);
}

int Class1::getChannelNum()
{
	return flt3d->getChNum();
}

void Class1::applyChanges()
{
	flt3d->applyChanges();
}

bool Class1::imageRead(System::String ^filename)
{
	return set3DImage_MultiTIFF(filename, 0, C1_SC_NONE);
	
}

bool Class1::saveDst3DImage(System::String ^filename)
{
	const char *c_str;
	marshal_context^ ctx = gcnew marshal_context();
	c_str = ctx->marshal_as<const char *>(filename);
	flt3d->saveDst3DImage(c_str);
	flt3d->saveRGBimageSeries(c_str, bc_max, bc_min);
	return true;
}

void Class1::hsv2rgb(float h, float s, float v, unsigned int %r, unsigned int %g, unsigned int %b)
{
	int hi = (int)(h / 60.0f) % 6;
	float f, p, q, t;

	f = h / 60.0f - (float)hi;
	p = v * (1.0f - s);
	q = v * (1.0f - f*s);
	t = v * (1.0f - (1.0f - f)*s);

	switch(hi){
		case 0:
			r = (unsigned int)(v*255.0f); g = (unsigned int)(t*255.0f); b = (unsigned int)(p*255.0f);
			break;
		case 1:
			r = (unsigned int)(q*255.0f); g = (unsigned int)(v*255.0f); b = (unsigned int)(p*255.0f);
			break;
		case 2:
			r = (unsigned int)(p*255.0f); g = (unsigned int)(v*255.0f); b = (unsigned int)(t*255.0f);
			break;
		case 3:
			r = (unsigned int)(p*255.0f); g = (unsigned int)(q*255.0f); b = (unsigned int)(v*255.0f);
			break;
		case 4:
			r = (unsigned int)(t*255.0f); g = (unsigned int)(p*255.0f); b = (unsigned int)(v*255.0f);
			break;
		case 5:
			r = (unsigned int)(v*255.0f); g = (unsigned int)(p*255.0f); b = (unsigned int)(q*255.0f);
			break;

	}
}

bool Class1::saveSrc2DImage(System::String ^filename)
{
	const char *c_str;
	marshal_context^ ctx = gcnew marshal_context();
	c_str = ctx->marshal_as<const char *>(filename);
	
	unsigned int h, r, g, b;
	float imax;
	
	if(!flt3d->hmapEmpty && hmap_use){
		if(proj_mode == C1_NORMAL_PROJECTION){
			flt3d->heightMapBasedProjection(flt3d->imgdata, hmapoffset, hmap_depth, hmap_range, -1);
		}
		else{
			if(dc_isEnable && !flt3d->dmapEmpty)flt3d->heightMapSimpleProjection(flt3d->imgdata, hmapoffset, hmap_depth, hmap_range, false, proj_th, true, dmap_range);
			else flt3d->heightMapSimpleProjection(flt3d->imgdata, hmapoffset, hmap_depth, hmap_range, false, proj_th, false, 0);
		}
	}
	else{
		flt3d->simpleProjection(flt3d->imgdata, bc_max, bc_min, currentZ, hmap_range, seg_show, seg_minVol, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);
	}

	string fnameYZ = c_str;
	size_t found = fnameYZ.find_first_of(".");
	if(found == string::npos)return false;
	fnameYZ.insert(found, "YZ");
	flt3d->setBufferYZ(flt3d->imgdata, currentX, bc_max, bc_min, seg_show, seg_minVol, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);

	string fnameZX = c_str;
	found = fnameZX.find_first_of(".");
	if(found == string::npos)return false;
	fnameZX.insert(found, "ZX");
	flt3d->setBufferZX(flt3d->imgdata, currentY, bc_max, bc_min, seg_show, seg_minVol, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);
	
	Mat tmpXY = Mat(flt3d->imageH, flt3d->imageW, CV_8UC3, flt3d->bufXY);
	Mat tmpYZ = Mat(flt3d->imageH, flt3d->imageZ, CV_8UC3, flt3d->bufYZ);
	Mat tmpZX = Mat(flt3d->imageZ, flt3d->imageW, CV_8UC3, flt3d->bufZX);
	return (imwrite(c_str, tmpXY) && imwrite(fnameYZ, tmpYZ) && imwrite(fnameZX, tmpZX));
	/*
	
	flt3d->setXPlotGraph(currentX, currentY, currentZ, bc_max, bc_min, dc_isEnable, dmap_coefficient, dmap_order);
	Mat graph = Mat(flt3d->graphH, flt3d->graphW, CV_8UC3, flt3d->graph);
	cvtColor(graph, graph, CV_RGB2BGR);
	return imwrite(c_str, graph);
	*/
}

bool Class1::saveDst2DImage(System::String ^filename)
{
	const char *c_str;
	marshal_context^ ctx = gcnew marshal_context();
	c_str = ctx->marshal_as<const char *>(filename);

	if(!flt3d->hmapEmpty && hmap_use){
		if(proj_mode == C1_NORMAL_PROJECTION){
			flt3d->heightMapBasedProjection(flt3d->dstdata, hmapoffset, hmap_depth, hmap_range, 0.5f);
		}
		else{
			if(dc_isEnable && !flt3d->dmapEmpty)flt3d->heightMapSimpleProjection(flt3d->dstdata, hmapoffset, hmap_depth, hmap_range, true, 0.5f, true, dmap_range);
			else flt3d->heightMapSimpleProjection(flt3d->dstdata, hmapoffset, hmap_depth, hmap_range, true, 0.5f, false, 0);
		}
	}
	else{
		flt3d->simpleProjection(flt3d->dstdata, bc_max, bc_min, currentZ, hmap_range, dc_isEnable, dmap_coefficient, dmap_order);
	}
	
	string fnameYZ = c_str;
	size_t found = fnameYZ.find_first_of(".");
	if(found == string::npos)return false;
	fnameYZ.insert(found, "YZ");
	flt3d->setBufferYZ(flt3d->dstdata, currentX, bc_max, bc_min, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order);

	string fnameZX = c_str;
	found = fnameZX.find_first_of(".");
	if(found == string::npos)return false;
	fnameZX.insert(found, "ZX");
	flt3d->setBufferZX(flt3d->dstdata, currentY, bc_max, bc_min, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order);
	
	Mat tmpXY = Mat(flt3d->imageH, flt3d->imageW, CV_8UC3, flt3d->bufXY);
	Mat tmpYZ = Mat(flt3d->imageH, flt3d->imageZ, CV_8UC3, flt3d->bufYZ);
	Mat tmpZX = Mat(flt3d->imageZ, flt3d->imageW, CV_8UC3, flt3d->bufZX);
	return (imwrite(c_str, tmpXY) && imwrite(fnameYZ, tmpYZ) && imwrite(fnameZX, tmpZX));
}

bool Class1::saveHeightMap(System::String ^filename)
{
	const char *c_str;
	marshal_context^ ctx = gcnew marshal_context();
	c_str = ctx->marshal_as<const char *>(filename);

	return flt3d->saveHeightMap(c_str);
}

bool Class1::readHeightMap(System::String ^filename)
{
	const char *c_str;
	marshal_context^ ctx = gcnew marshal_context();
	c_str = ctx->marshal_as<const char *>(filename);

	return flt3d->readHeightMap(c_str);
}

bool Class1::empty()
{
	return flt3d->isEmpty;
}

void Class1::setX(int value)
{
	if((value < 0) || (value >= flt3d->imageW) || (flt3d->isEmpty))return;
	currentX = value;
}

void Class1::setY(int value)
{
	if((value < 0) || (value >= flt3d->imageH) || (flt3d->isEmpty))return;
	currentY = value;
}

void Class1::setZ(int value)
{
	if((value < 0) || (value >= flt3d->imageZ) || (flt3d->isEmpty))return;
	currentZ = value;
}

void Class1::setBCmax(float value)
{
	bc_max = value;
}

void Class1::setBCmin(float value)
{
	bc_min = value;
}

void Class1::setHmapVisibility(bool isVisible)
{
	hmap_isVisible = isVisible;
}

void Class1::setHmapActivity(bool value)
{
	hmap_use = value;
}

void Class1::setHmapOffset(float value)
{
	hmapoffset = value;
}

void Class1::setProjectionMode(int value)
{
	proj_mode = value;
}

void Class1::setProjectionThreshold(float value)
{
	proj_th = value;
}

void Class1::setHmapProjDepth(float value)
{
	hmap_depth = value;
}

void Class1::setHmapProjRange(float value)
{
	hmap_range = value;
}

void Class1::setDepthCodeEnable(bool isEnable)
{
	dc_isEnable = isEnable;
}

void Class1::setDmapRange(float value)
{
	dmap_range = value;
}

void Class1::setDmapCoefficient(float value)
{
	dmap_coefficient = value;
}

void Class1::setDmapOrder(float value)
{
	dmap_order = value;
}

void Class1::setZBCEnable(bool isEnable)
{
	zbc_isEnable = isEnable;
}

void Class1::setZBCparams(int channel, float coef, float order)
{
	zbc_channel = channel;
	zbc_coefficient = coef;
	zbc_order = order;
}

void Class1::setSegVisibility(bool isVisible)
{
	seg_show = isVisible;
}

void Class1::setSegMinVol(int value)
{
	seg_minVol = value;
}

void Class1::getImageSize([Runtime::InteropServices::Out] int %imgX,
						  [Runtime::InteropServices::Out] int %imgY,
						  [Runtime::InteropServices::Out] int %imgZ)
{
	if(flt3d->isEmpty){
		imgX = -1;
		imgY = -1;
		imgZ = -1;
	}
	imgX = flt3d->imageW;
	imgY = flt3d->imageH;
	imgZ = flt3d->imageZ;
}

unsigned long Class1::getImageDataArrayXY([Runtime::InteropServices::Out] IntPtr %ptr,
										  [Runtime::InteropServices::Out] int %bytesperpixel,
										  [Runtime::InteropServices::Out] long %width,
										  [Runtime::InteropServices::Out] long %height)
{
	unsigned int h, r, g, b;
	float imax;

	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	if(!flt3d->hmapEmpty && hmap_use){
		if(proj_mode == C1_NORMAL_PROJECTION){
			flt3d->heightMapBasedProjection(flt3d->imgdata, hmapoffset, hmap_depth, hmap_range, -1);
		}
		else{
			if(dc_isEnable && !flt3d->dmapEmpty)flt3d->heightMapSimpleProjection(flt3d->imgdata, hmapoffset, hmap_depth, hmap_range, false, proj_th, true, dmap_range);
			else flt3d->heightMapSimpleProjection(flt3d->imgdata, hmapoffset, hmap_depth, hmap_range, false, proj_th, false, 0);
		}
	}
	else{
		//printf("SimpleProjection Called\n");
		flt3d->simpleProjection(flt3d->imgdata, bc_max, bc_min, currentZ, hmap_range, seg_show, seg_minVol, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);
	}
	//ptr = IntPtr(&(flt3d->imgdata[currentZ*flt3d->imageW*flt3d->imageH]));
	ptr = IntPtr(flt3d->bufXY);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->imageW;
	height = flt3d->imageH;
	return flt3d->imageW*flt3d->imageH*bytesperpixel;
}

unsigned long Class1::getImageDataArrayYZ([Runtime::InteropServices::Out] IntPtr %ptr,
										  [Runtime::InteropServices::Out] int %bytesperpixel,
										  [Runtime::InteropServices::Out] long %width,
										  [Runtime::InteropServices::Out] long %height)
{
	unsigned int h, r, g, b;

	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}

	flt3d->setBufferYZ(flt3d->imgdata, currentX, bc_max, bc_min, seg_show, seg_minVol, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);

	ptr = IntPtr(flt3d->bufYZ);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->imageZ;
	height = flt3d->imageH;
	return flt3d->imageZ*flt3d->imageH*bytesperpixel;
}

unsigned long Class1::getImageDataArrayZX([Runtime::InteropServices::Out] IntPtr %ptr,
										  [Runtime::InteropServices::Out] int %bytesperpixel,
										  [Runtime::InteropServices::Out] long %width,
										  [Runtime::InteropServices::Out] long %height)
{
	unsigned int h, r, g, b;

	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	
	flt3d->setBufferZX(flt3d->imgdata, currentY, bc_max, bc_min, seg_show, seg_minVol, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);

	ptr = IntPtr(flt3d->bufZX);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->imageW;
	height = flt3d->imageZ;
	return flt3d->imageW*flt3d->imageZ*bytesperpixel;
}


unsigned long Class1::getDstImageDataArrayXY([Runtime::InteropServices::Out] IntPtr %ptr,
											 [Runtime::InteropServices::Out] int %bytesperpixel,
											 [Runtime::InteropServices::Out] long %width,
											 [Runtime::InteropServices::Out] long %height)
{
	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	if(!flt3d->hmapEmpty && hmap_use){
		if(proj_mode == C1_NORMAL_PROJECTION){
			flt3d->heightMapBasedProjection(flt3d->dstdata, hmapoffset, hmap_depth, hmap_range, 0.5f);
		}
		else{
			if(dc_isEnable && !flt3d->dmapEmpty)flt3d->heightMapSimpleProjection(flt3d->dstdata, hmapoffset, hmap_depth, hmap_range, true, 0.5f, true, dmap_range);
			else flt3d->heightMapSimpleProjection(flt3d->dstdata, hmapoffset, hmap_depth, hmap_range, true, 0.5f, false, 0);
		}
	}
	else{
		flt3d->simpleProjection(flt3d->dstdata, bc_max, bc_min, currentZ, hmap_range, seg_show, seg_minVol, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);
	}
	//ptr = IntPtr(&(flt3d->dstdata[currentZ*flt3d->imageW*flt3d->imageH]));
	ptr = IntPtr(flt3d->bufXY);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->imageW;
	height = flt3d->imageH;
	return flt3d->imageW*flt3d->imageH*bytesperpixel;
}

unsigned long Class1::getDstImageDataArrayYZ([Runtime::InteropServices::Out] IntPtr %ptr,
											 [Runtime::InteropServices::Out] int %bytesperpixel,
											 [Runtime::InteropServices::Out] long %width,
											 [Runtime::InteropServices::Out] long %height)
{
	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	
	flt3d->setBufferYZ(flt3d->dstdata, currentX, bc_max, bc_min, seg_show, seg_minVol, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);
	
	ptr = IntPtr(flt3d->bufYZ);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->imageZ;
	height = flt3d->imageH;
	return flt3d->imageZ*flt3d->imageH*bytesperpixel;
}

unsigned long Class1::getDstImageDataArrayZX([Runtime::InteropServices::Out] IntPtr %ptr,
											 [Runtime::InteropServices::Out] int %bytesperpixel,
											 [Runtime::InteropServices::Out] long %width,
											 [Runtime::InteropServices::Out] long %height)
{
	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	flt3d->setBufferZX(flt3d->dstdata, currentY, bc_max, bc_min, seg_show, seg_minVol, hmap_isVisible, hmapoffset, hmap_range, dc_isEnable, dmap_coefficient, dmap_order, zbc_isEnable, zbc_channel, zbc_coefficient, zbc_order);

	ptr = IntPtr(flt3d->bufZX);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->imageW;
	height = flt3d->imageZ;
	return flt3d->imageW*flt3d->imageZ*bytesperpixel;
}

unsigned long Class1::getHeightMapDataArrayXY([Runtime::InteropServices::Out] IntPtr %ptr,
											  [Runtime::InteropServices::Out] int %bytesperpixel,
											  [Runtime::InteropServices::Out] long %width,
											  [Runtime::InteropServices::Out] long %height)
{
	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	
	flt3d->setHeightMapToBufferXY();

	ptr = IntPtr(flt3d->bufXY);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->imageW;
	height = flt3d->imageH;
	return flt3d->imageW*flt3d->imageH*bytesperpixel;
}

unsigned long Class1::getXPlotGraph([Runtime::InteropServices::Out] IntPtr %ptr,
									[Runtime::InteropServices::Out] int %bytesperpixel,
									[Runtime::InteropServices::Out] long %width,
									[Runtime::InteropServices::Out] long %height)
{
	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	flt3d->setXPlotGraph(currentX, currentY, currentZ, bc_max, bc_min, dc_isEnable, dmap_coefficient, dmap_order);
	ptr = IntPtr(flt3d->graph);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->graphW;
	height = flt3d->graphH;
	return flt3d->graphW*flt3d->graphH*bytesperpixel;
}

unsigned long Class1::getYPlotGraph([Runtime::InteropServices::Out] IntPtr %ptr,
									[Runtime::InteropServices::Out] int %bytesperpixel,
									[Runtime::InteropServices::Out] long %width,
									[Runtime::InteropServices::Out] long %height)
{
	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	flt3d->setYPlotGraph(currentX, currentY, currentZ, bc_max, bc_min, dc_isEnable, dmap_coefficient, dmap_order);
	ptr = IntPtr(flt3d->graph);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->graphW;
	height = flt3d->graphH;
	return flt3d->graphW*flt3d->graphH*bytesperpixel;
}

unsigned long Class1::getZPlotGraph([Runtime::InteropServices::Out] IntPtr %ptr,
									[Runtime::InteropServices::Out] int %bytesperpixel,
									[Runtime::InteropServices::Out] long %width,
									[Runtime::InteropServices::Out] long %height)
{
	if(flt3d->isEmpty){
		ptr = IntPtr(0);
		bytesperpixel = -1;
		width = -1;
		height = -1;
		return 0;
	}
	flt3d->setZPlotGraph(currentX, currentY, currentZ, bc_max, bc_min, dc_isEnable, dmap_coefficient, dmap_order);
	ptr = IntPtr(flt3d->graph);
	bytesperpixel = sizeof(unsigned char)*3;
	width = flt3d->graphW;
	height = flt3d->graphH;
	return flt3d->graphW*flt3d->graphH*bytesperpixel;
}

void Class1::AdaptiveThreshold3DLineKernels(int blocksize, int angle_d, float factor_C_Z, float constC, int kernelType)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU)flt3d->allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, constC*C_factor, constC*factor_C_Z*C_factor, kernelType);
	//flt3d->drawCircle(3.141592653589793238462 / 2.0, 3.141592653589793238462 / 4.0, blocksize);
	//flt3d->drawLineSegments(blocksize, angle_d);
	//flt3d->drawIcoSphere(blocksize, angle_d);
}

void Class1::segmentation_SobelLikeADTH(int blocksize, 
								int angle_d, 
								float factor_C_Z, 
								float minC, 
								float maxC, 
								float interval, 
								int kernelType, 
								int min_segVol, 
								int minInvalidStructureArea, 
								int closing)
{
	if(isEmpty)return;
	if(flt3d->isEnableGPU){
		flt3d->segmentation_SobelLikeADTH(blocksize,
										  angle_d, 
										  factor_C_Z, 
										  minC*C_factor, 
										  maxC*C_factor, 
										  interval*C_factor, 
										  kernelType, 
										  minInvalidStructureArea, 
										  closing);
		
		flt3d->saveBackupSegmentsData();
	}
/*	
	if(flt3d->isEnableGPU)flt3d->segmentation_SobelLikeADTH(blocksize,
															angle_d, 
															factor_C_Z, 
															minC*C_factor, 
															maxC*C_factor, 
															interval*C_factor, 
															kernelType, 
															minInvalidStructureArea, 
															closing,
															currentX,
															currentY,
															currentZ,
															bc_max,
															bc_min);
*/	
	//if(flt3d->isEnableGPU && closing == 0)flt3d->adjustAirspacesBrightness(0, blocksize, angle_d, minC*C_factor, maxC*C_factor, 100, -1*interval*C_factor, factor_C_Z, kernelType);
}

void Class1::hMinimaTransform3D(float h, int chk_interval, bool copyToHostMemory)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU)flt3d->hMinimaTransform3D_GPU(h, chk_interval, true);
	
}

void Class1::segmentation_hMinimaTransform(float minh, float maxh, float interval, int min_segVol, int minInvalidStructureArea, int closing)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU){
		flt3d->segmentation_hMinimaTransform(minh, maxh, interval, min_segVol, minInvalidStructureArea, closing);
		flt3d->saveBackupSegmentsData();
	}
}

void Class1::segmentation_FloodFill(float th_val, int connectType, int minSize)
{
	flt3d->binarySegmentationLow(flt3d->dstdata, th_val, connectType, minSize, true);
	flt3d->saveBackupSegmentsData();
}

void Class1::segmentation_FloodFill(float th_val, int connectType, int minSize, int noise_th, int wall_th, int minInvalidStructureVol, bool saveValidSegments, bool saveInvalidSegments)
{
	flt3d->binarySegmentationLow(flt3d->dstdata, th_val, connectType, minSize, true, noise_th, wall_th, minInvalidStructureVol, saveValidSegments, saveInvalidSegments);
	flt3d->saveBackupSegmentsData();
}

void Class1::AdaptiveThreshold3D(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	if(isEmpty)return;
	
	if(flt3d->isEnableGPU)flt3d->AdaptiveThreshold3D_GPU(blocksize, constC*C_factor, thresholdType, copyToHostMemory);
	else flt3d->AdaptiveThreshold3D_CPU(blocksize, constC*C_factor, thresholdType);
	
/*
	if(flt3d->isEnableGPU)flt3d->segmentation_SobelLikeADTH(blocksize,
															4,
															0.2f,
															constC,
															constC+100.0f/500.0f,
															5.0f/500.0f,
															thresholdType,
															500,
															100,
															3,
															currentX,
															currentY,
															currentZ,
															bc_max,
															bc_min);
*/
}

void Class1::AdaptiveThreshold2D(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	if(isEmpty)return;
	
	if(flt3d->isEnableGPU)flt3d->AdaptiveThreshold2D_GPU(blocksize, constC*C_factor, thresholdType, copyToHostMemory);
	else flt3d->AdaptiveThreshold2D_CPU(blocksize, constC*C_factor, thresholdType);
	
}

void Class1::threshold3D(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU)flt3d->threshold3D_GPU(blocksize, constC*C_factor, thresholdType, copyToHostMemory);
	else flt3d->threshold3D_CPU(blocksize, constC*C_factor, thresholdType);

	//flt3d->resetSegmentColors();

	//flt3d->fillingHoles(flt3d->dstdata, 12, 4, 0);
	//flt3d->estimateWallThickness(1);
	
}

void Class1::threshold2D(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU)flt3d->threshold2D_GPU(blocksize, constC*C_factor, thresholdType, copyToHostMemory);
	else flt3d->threshold2D_CPU(blocksize, constC*C_factor, thresholdType);

	//flt3d->binarySegmentationLow(flt3d->dstdata, 0.1f, SEGMENT_CONNECT6, 500, true, 4, 8, 1000, true, true);
	//flt3d->binarySegmentationLow(flt3d->dstdata, 0.1f, SEGMENT_CONNECT6, 500, true);

	//if(flt3d->isEnableGPU)flt3d->openingSpherical(blocksize, true);

	//flt3d->saveRGBimageSeries("SegmentedImage", bc_max, bc_min);
}

void Class1::smooth3D(int blocksize, int filterType, bool copyToHostMemory)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU)flt3d->smoothing3D_GPU(blocksize, filterType, copyToHostMemory);
	//if(flt3d->isEnableGPU)flt3d->smoothing2D_GPU(blocksize, filterType, copyToHostMemory);
}

void Class1::dilation3D(int radius, int filterShape, bool copyToHostMemory)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU){
		
		if(filterShape == C1_FLT_SPHERE)flt3d->dilationSpherical(radius, copyToHostMemory);
		if(filterShape == C1_FLT_CUBE)  flt3d->dilationCubic(radius, copyToHostMemory);
		
		//if(filterShape == C1_FLT_SPHERE)flt3d->closingSpherical(radius, true);
		//flt3d->DoG(radius, true);
	}
}

void Class1::erosion3D(int radius, int filterShape, bool copyToHostMemory)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU){
		if(filterShape == C1_FLT_SPHERE)flt3d->erosionSpherical(radius, copyToHostMemory);
		if(filterShape == C1_FLT_CUBE)  flt3d->erosionCubic(radius, copyToHostMemory);
	}
}

void Class1::generateHeightMap(int blocksize_xy, int blocksize_z, float th, int thresholdType, int smoothLv)
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU)flt3d->generateHeightMapGPU(blocksize_xy, blocksize_z, th, thresholdType, smoothLv);
	else flt3d->generateHeightMapCPU(blocksize_xy, blocksize_z, th, thresholdType);
}

void Class1::inactivateHeightMap()
{
	if(isEmpty)return;

	flt3d->clearHeightMap();
}

void Class1::generateDepthMap()
{
	if(isEmpty)return;

	if(flt3d->isEnableGPU)flt3d->generateDepthMapGPU();
	else flt3d->generateDepthMap();
}
/*
void Class1::drawSelectionZX(int brush_size, int x, int y, int z, int mode)
{
	flt3d->drawSelection(brush_size, x, y, z, mode);
}

void Class1::setCopySelectionSliceSrcZX(int slice_id)
{
	if(slice_id < 0)selection_copy_src = currentY;
	else selection_copy_src = slice_id;
}

void Class1::copySelectionSliceZX(int src, int dst)
{
	if(src < 0 || dst < 0)flt3d->copySelectionSliceZX(selection_copy_src, currentY);
	else flt3d->copySelectionSliceZX(src, dst);
}

void Class1::maskDstCall(bool inv)
{
	if(isEmpty)return;
	flt3d->maskDst(inv);
}
*/
void Class1::calcHmapAreaCall(int reso)
{
	if(isEmpty)return;
	flt3d->calcHmapArea(reso);
}

void Class1::applyBlightnessContrast()
{
	if(isEmpty)return;

	flt3d->applyBC(bc_max, bc_min, dc_isEnable, dmap_coefficient, dmap_order);
}

void Class1::selectSegment(int x, int y, int z)
{
	if(isEmpty)return;

	flt3d->selectSegment(x, y, z);
}

void Class1::selectAllSegments()
{
	if(isEmpty)return;

	flt3d->selectAllSegments();
}

void Class1::deselectSegment()
{
	if(isEmpty)return;

	flt3d->deselectSegment();
}

void Class1::selectSegments_TH(float th)
{
	if(isEmpty)return;

	flt3d->autoSelectSegmentsTH(th);
}

void Class1::deselectSegments_TH(float th)
{
	if(isEmpty)return;

	flt3d->autoDeselectSegmentsTH(th);
}

void Class1::cropSegments()
{
	if(isEmpty)return;

	flt3d->cropSegments();
}

void Class1::joinSelectedSegments()
{
	if(isEmpty)return;

	flt3d->joinSelectedSegments();
}

void Class1::separateSelectedSegments()
{
	if(isEmpty)return;

	flt3d->separateSelectedSegments();
}

void Class1::dilateSphereSelectedSegments(int radius)
{
	if(isEmpty)return;

	flt3d->dilateSphereSelectedSegments(radius);
}

void Class1::erodeSphereSelectedSegments(int radius, bool clamp)
{
	if(isEmpty)return;

	flt3d->erodeSphereSelectedSegments(radius, clamp);
}

void Class1::loadBackupSegmentsData()
{
	if(isEmpty)return;

	flt3d->loadBackupSegmentsData();
}

void Class1::setCroppingParams(bool isEnable, bool useHmap, int upper, int lower, int border)
{
	if(isEmpty)return;

	flt3d->setCroppingParams(isEnable, useHmap, upper, lower, border);
}

void Class1::watershed3D(float stride)
{
	if(isEmpty)return;

	flt3d->watershed3D(stride, seg_minVol);
}