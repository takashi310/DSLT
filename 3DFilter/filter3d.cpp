//#include <msclr/marshal.h>
#include <stdio.h>
#include <cmath>

#define DLL_FILTER3D

#include "filter3d.h"
#include <queue>
#include <algorithm>
#include<cublas.h>

using namespace cv;
using namespace multif;
using namespace TestGPUclass;
using namespace std;

#define PI 3.141592653589793238462
#define SAFE_DELETE(p) { if (p){ delete p; } p = NULL; }
#define SAFE_DELETE_ARRAY(p) { if (p){ delete [] p; } p = NULL; }

template <typename _T> void clear2DVector(std::vector<std::vector<_T>>* &vec2d)
{
	if(vec2d == NULL)return;
	for(int i = 0; i < vec2d->size(); i++)std::vector<_T>().swap((*vec2d)[i]);
	std::vector<std::vector<_T>>().swap(*vec2d);
}

template <typename _T> void copy2DVector(std::vector<std::vector<_T>>* &dst, std::vector<std::vector<_T>>* &src)
{
	if(src == NULL || dst == NULL)return;
	clear2DVector(dst);
	for(int i = 0; i < src->size(); i++){
		dst->push_back(std::vector<_T>());
		copy((*src)[i].begin(), (*src)[i].end(), back_inserter((*dst)[i]));
	}
}

template <typename _T> void copy1DVector(std::vector<_T>* &dst, std::vector<_T>* &src)
{
	if(src == NULL || dst == NULL)return;
	std::vector<_T>().swap(*dst);
	copy(src->begin(), src->end(), back_inserter(*dst));
}

Filter3D::Filter3D()
{
	imageW = 0;
	imageH = 0;
	imageZ = 0;
	nSlices = 0;
	nChannels = 0;
	zScaleFactor = 1.0f;
	isEmpty = true;
	hmapEmpty = true;
	dmapEmpty = true;
	isEnableGPU = false;
	
	tifio = NULL;
	rawdata = NULL;
	rawdata_resized = NULL;
	imgdata = NULL;
	bufdata = NULL;
	dstdata = NULL;
	//selectdata = NULL;
	segdata = NULL;
	hmap = NULL;
	hmap_normalized = NULL;
	hmbproj = NULL;
	normals = NULL;
	areas = NULL;
	dmap = NULL;
	bufXY = NULL;
	bufZX = NULL;
	bufYZ = NULL;
	seg_colors = NULL;
	seg_bbox = NULL;
	invalid_seg_bbox = NULL;
	segments = NULL;
	invalid_segments = NULL;
	selected_segment = NULL;

	seg_colors_bk = NULL;
	seg_bbox_bk = NULL;
	segments_bk = NULL;
	segdata = NULL;

	graph_tmpX = NULL;
	graph_tmpY = NULL;
	graph_tmpZ = NULL;
	graph = NULL;

	d_Tmp = NULL;
	d_Input = NULL;
	d_Buffer = NULL;
	d_Output = NULL;

	crop_isEnable = false;
	crop_useHmap = false;
	crop_upper = 0;
	crop_lower = 0;
}

Filter3D::~Filter3D()
{
	clear();
}

void Filter3D::clear()
{
	if(!isEmpty){
		SAFE_DELETE(tifio);
		SAFE_DELETE_ARRAY(rawdata);
		SAFE_DELETE_ARRAY(rawdata_resized);
		SAFE_DELETE_ARRAY(imgdata);
		SAFE_DELETE_ARRAY(bufdata);
		SAFE_DELETE_ARRAY(dstdata);
		//SAFE_DELETE_ARRAY(selectdata);
		SAFE_DELETE_ARRAY(segdata);
		SAFE_DELETE_ARRAY(hmap);
		SAFE_DELETE_ARRAY(hmap_normalized);
		SAFE_DELETE_ARRAY(hmbproj);
		SAFE_DELETE_ARRAY(normals);
		SAFE_DELETE_ARRAY(areas);
		SAFE_DELETE_ARRAY(dmap);
		SAFE_DELETE_ARRAY(bufXY);
		SAFE_DELETE_ARRAY(bufZX);
		SAFE_DELETE_ARRAY(bufYZ);
				
		if(seg_colors != NULL)std::vector<float>().swap(*seg_colors);
		SAFE_DELETE(seg_colors);
		if(seg_bbox != NULL)std::vector<Box3D>().swap(*seg_bbox);
		SAFE_DELETE(seg_bbox);
		if(invalid_seg_bbox != NULL)std::vector<Box3D>().swap(*invalid_seg_bbox);
		SAFE_DELETE(invalid_seg_bbox);
		clear2DVector(segments);
		SAFE_DELETE(segments);
		clear2DVector(invalid_segments);
		SAFE_DELETE(invalid_segments);
		if(selected_segment != NULL)std::vector<int>().swap(*selected_segment);
		SAFE_DELETE(selected_segment);

		if(seg_colors_bk != NULL)std::vector<float>().swap(*seg_colors_bk);
		SAFE_DELETE(seg_colors_bk);
		if(seg_bbox_bk != NULL)std::vector<Box3D>().swap(*seg_bbox_bk);
		SAFE_DELETE(seg_bbox_bk);
		clear2DVector(segments_bk);
		SAFE_DELETE(segments_bk);
		SAFE_DELETE_ARRAY(segdata_bk);

		SAFE_DELETE_ARRAY(graph_tmpX);
		SAFE_DELETE_ARRAY(graph_tmpY);
		SAFE_DELETE_ARRAY(graph_tmpZ);
		SAFE_DELETE_ARRAY(graph);
		if(isEnableGPU){
			if(d_Tmp)   checkCudaErrors( cudaFree(d_Tmp) );
			if(d_Buffer)checkCudaErrors( cudaFree(d_Buffer) );
			if(d_Output)checkCudaErrors( cudaFree(d_Output) );
			if(d_Input) checkCudaErrors( cudaFree(d_Input) );
			d_Tmp = NULL;
			d_Buffer = NULL;
			d_Output = NULL;
			d_Input = NULL;
			cudaDeviceReset();
		}
		else{
			SAFE_DELETE_ARRAY(h_tmp);
		}
		
		imageW = 0;
		imageH = 0;
		imageZ = 0;
		nSlices = 0;
		nChannels = 0;
		zScaleFactor = 1.0f;
		
		isEmpty = true;
		hmapEmpty = true;
		dmapEmpty = true;
		isEnableGPU = false;
	}
}

void Filter3D::setDevice()
{
	int deviceCount;
	printf("<<CUDA: setDevice>>\n");
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
        isEnableGPU = false;
		return;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
		isEnableGPU = false;
		return;
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

	if(deviceCount >= 0){
		int devID = gpuGetMaxGflopsDeviceId();
		printf("gpuGetMaxGflopsDeviceId: devID = %d\n", devID);
		checkCudaErrors( cudaSetDevice(devID) );
		isEnableGPU = true;
		printf("cudaSetDevice: Succeeded\n");
	}
	else{
		isEnableGPU = false;
	}
}

bool Filter3D::set3DImage_MultiTIFF(const char filename[], int channel, int z_scaling)
{
	char *tmpdata;
	uint16	samplesperpixel, bitspersample;

	clear();
	
	tiffhndl::FileOpener fo(filename);
	if(fo.empty()){
		printf("set3DImage_MultiTIFF: cannot open file\n");
		clear();
		return false;
	}

	isEmpty = false;
	tiffhndl::Calibration calib = fo.getImageCalibration();
	imageW = calib.width;
	imageH = calib.height;
	imageZ = nSlices = calib.slices;
	nChannels = calib.channels;
	if(calib.calibrated)zScaleFactor = calib.pixelDepth/calib.pixelWidth;
	else zScaleFactor = 1.0f;

	int depth;
	if(zScaleFactor > 1.0f)depth = (int)(nSlices*zScaleFactor);
	else depth = nSlices;
	
	printf("slices: %d  width: %f  depth: %f  imageZ(Z_Scaling ON): %f  channels: %d\n", calib.slices, calib.pixelWidth, calib.pixelDepth, calib.slices*calib.pixelDepth/calib.pixelWidth, nChannels);
	printf("imageW: %d  imageH: %d  depth: %d\n", imageW, imageH, depth);
	
	try{
		rawdata = new float[imageW*imageH*nSlices*nChannels];
		rawdata_resized = new float[imageW*imageH*depth*nChannels];
		imgdata = new float[imageW*imageH*depth];
		bufdata = new float[imageW*imageH*depth];
		dstdata = new float[imageW*imageH*depth];
		//selectdata = new float[imageW*imageH*depth];
		segdata = new int[imageW*imageH*depth];
		hmap = new float[imageW*imageH];
		hmap_normalized = new float[imageW*imageH];
		hmbproj = new float[imageW*imageH];
		normals = new Point3f[imageW*imageH];
		areas = new float[imageW*imageH];
		dmap = new float[imageW*imageH*depth];
		bufXY = new unsigned char[imageH*imageW*3];
		bufZX = new unsigned char[depth*imageW*3];
		bufYZ = new unsigned char[imageH*depth*3];

		seg_colors = new vector<float>;
		seg_bbox = new vector<Box3D>;
		invalid_seg_bbox = new vector<Box3D>;
		segments = new vector< vector<Point3i> >;
		invalid_segments = new vector< vector<Point3i> >;
		selected_segment = new vector<int>; 

		seg_colors_bk = new vector<float>;
		seg_bbox_bk = new vector<Box3D>;
		segments_bk = new vector< vector<Point3i> >;
		segdata_bk = new int[imageW*imageH*depth];
	}
	catch(std::bad_alloc){
		printf("set3DImage_MultiTIFF: BAD ALLOC Exception1\n");
		clear();
		return false;
	}
	printf("set3DImage_MultiTIFF: Memoly allocation succeeded\n");
	

	float *tmp = rawdata;
	for(int i = 0; i < nChannels; i++){
		if(!fo.getXYZStackFloat(tmp, true, i)){
			printf("set3DImage_MultiTIFF: getXYZStackFloat failed\n");
		}
		tmp += imageW*imageH*nSlices;
	}

	if(channel < nChannels && channel >= 0)curCh = channel;
	else curCh = 0;
	switchZScaling(z_scaling);

	setDevice();
	if(isEnableGPU){
		checkCudaErrors( cudaMalloc((void **)(&d_Input),   depth * imageW * imageH * sizeof(float)) );
		checkCudaErrors( cudaMalloc((void **)(&d_Output),  depth * imageW * imageH * sizeof(float)) );
		checkCudaErrors( cudaMalloc((void **)(&d_Buffer) , depth * imageW * imageH * sizeof(float)) );
		checkCudaErrors( cudaMalloc((void **)(&d_Tmp) , depth * imageW * imageH * sizeof(float)) );
		
		checkCudaErrors( cudaMemcpy(d_Input, imgdata, depth * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
		printf("GPU Mode\n");
		printf("allocated memory: %d (MB)\n", depth * imageW * imageH * sizeof(float) * 4 / (1024 * 1024));
		//CuDeviceProp(); //SetDeviceを行うのでDeviceが2つあるときはだめ
	}
	else{
		try{
			h_tmp = new float[imageW*imageH*depth];
		}
		catch(std::bad_alloc){
			printf("set3DImageMat_MultiTIFF: BAD ALLOC Exception2\n");
			clear();
			return false;
		}
		memcpy(h_tmp, imgdata, imageW*imageH*depth*sizeof(float));
		printf("CPU Mode\n");
	}

	//if(calib.calibrated && z_scaling)imageZ = calib.slices * calib.pixelDepth / calib.pixelWidth;

	generateGraphTemplate();

	return true;
}

void Filter3D::switchZScaling(int type)
{
	if(isEmpty)return;

	zScaling = type;
	
	int depth;
	if(zScaling == SC_NONE)depth = nSlices;
	else depth = (int)(nSlices*zScaleFactor);

	for(int i = 0; i < nChannels; i++){
		if(zScaling == SC_NONE || zScaleFactor == 1.0f)memcpy(rawdata_resized + imageW*imageH*nSlices*i, rawdata + imageW*imageH*nSlices*i, imageW*imageH*nSlices*sizeof(float));
		if(zScaling == SC_AREA_AVE)scalingAreaAveZ(rawdata_resized + imageW*imageH*depth*i, rawdata + imageW*imageH*nSlices*i, imageW, imageH, nSlices, zScaleFactor);
		if(zScaling == SC_LANCZOS2)scalingLanczosZ(rawdata_resized + imageW*imageH*depth*i, rawdata + imageW*imageH*nSlices*i, imageW, imageH, nSlices, zScaleFactor, 2);
		if(zScaling == SC_LANCZOS3)scalingLanczosZ(rawdata_resized + imageW*imageH*depth*i, rawdata + imageW*imageH*nSlices*i, imageW, imageH, nSlices, zScaleFactor, 3);
	}

	imageZ = depth;

	switchChannel(curCh);
}

void Filter3D::switchChannel(int channel)
{
	if(isEmpty)return;

	if(channel < nChannels && channel >= 0){
		curCh = channel;
		memcpy(imgdata, rawdata_resized + imageW*imageH*imageZ*curCh, imageW*imageH*imageZ*sizeof(float));
		memcpy(bufdata, imgdata, imageW*imageH*imageZ*sizeof(float));
		memcpy(dstdata, imgdata, imageW*imageH*imageZ*sizeof(float));

		if(isEnableGPU)checkCudaErrors( cudaMemcpy(d_Input, imgdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	}
}

int Filter3D::getChNum()
{
	if(isEmpty)return 0;
	
	return nChannels;
}

void Filter3D::applyChanges()
{
	if(isEmpty)return;
	memcpy(imgdata, dstdata, imageW*imageH*imageZ*sizeof(float));
	memcpy(rawdata_resized + imageW*imageH*imageZ*curCh, dstdata, imageW*imageH*imageZ*sizeof(float));
	
	if(isEnableGPU)checkCudaErrors( cudaMemcpy(d_Input, imgdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
}

void Filter3D::scalingAreaAveZ(float *dst, const float *src, int src_width, int src_height, int src_depth, float factor)
{
	if(factor < 1.0f){
		float recipro = 1.0f/factor;
		for(unsigned int y = 0; y < src_height; y++){
			for(unsigned int x = 0; x < src_width; x++){
				for(unsigned int i = 0; i < (int)floor(src_depth*factor); i++){
					float st = i * recipro - 1;
					float st_voxel_fac = 1.0f - (st - floor(st));
					int st_id =  (int)floor(st + st_voxel_fac);

					float ed = (i + 1) * recipro - 1;
					float ed_voxel_fac = ed - floor(ed);
					int ed_id = (int)floor(ed + 1.0f - ed_voxel_fac);

					float sum = src[st_id*src_height*src_width + y*src_width + x] * st_voxel_fac;
					float n = st_voxel_fac;

					for(unsigned int j = st_id + 1; j < ed_id; j++){
						if(j < src_depth){
							sum += src[j*src_height*src_width + y*src_width + x];
							n += 1.0f;
						}
					}

					if(ed_id < src_depth){
						sum += src[ed_id*src_height*src_width + y*src_width + x] * ed_voxel_fac;
						n += ed_voxel_fac;
					}
					dst[i*src_height*src_width + y*src_width + x] = sum / n;
				}
			}
		}
	}
	else if(factor > 1.0f){
		float recipro = 1.0f/factor;
		for(unsigned int y = 0; y < src_height; y++){
			for(unsigned int x = 0; x < src_width; x++){
				for(unsigned int i = 0; i < (int)floor(src_depth*factor); i++){
					if((int)floor(i*recipro) < (int)ceil((i+1)*recipro)-1){
						float st = i * recipro - 1;
						float st_voxel_fac = 1.0f - (st - floor(st));
						int st_id =  (int)floor(st + st_voxel_fac);

						float ed = (i + 1) * recipro - 1;
						float ed_voxel_fac = ed - floor(ed);
						int ed_id = (int)floor(ed + 1.0f - ed_voxel_fac);

						float sum = src[st_id*src_height*src_width + y*src_width + x] * st_voxel_fac;
						float n = st_voxel_fac;

						if(ed_id < src_depth){
							sum += src[ed_id*src_height*src_width + y*src_width + x] * ed_voxel_fac;
							n += ed_voxel_fac;
						}
						dst[i*src_height*src_width + y*src_width + x] = sum / n;
					}
					else{
						dst[i*src_height*src_width + y*src_width + x] = src[(int)floor(i*recipro)*src_height*src_width + y*src_width + x];
					}
				}
			}
		}
	}
}

double round(double number)
{
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

void Filter3D::scalingLanczosZ(float *dst, const float *src, int src_width, int src_height, int src_depth, float factor, int n)
{
	if(factor < 1.0f){
		for(unsigned int y = 0; y < src_height; y++){
			for(unsigned int x = 0; x < src_width; x++){
				for(int z = 0; z < (int)floor(src_depth*factor); z++){
					double weight_sum = 0.0;
					dst[z*src_height*src_width + y*src_width + x] = 0.0f;

					for(int i = (int)round((z+0.5-n)/factor); i < (int)round((z+0.5+n)/factor); i++){//lanczos関数は境界で0なのでroundそのままでOK
						double d = abs((i + 0.5)*factor - (z + 0.5));
						double weight = d == 0.0 ? 1.0 : sin(PI*d) * sin(PI*d/n) / (PI*d*(PI*d/n));
						double srcval;

						if(i < 0)srcval = src[0*src_height*src_width + y*src_width + x];
						else if(i >= src_depth)srcval = src[(src_depth-1)*src_height*src_width + y*src_width + x];
						else srcval = src[i*src_height*src_width + y*src_width + x];

						dst[z*src_height*src_width + y*src_width + x] += srcval * weight;
						weight_sum += weight;
					}
					dst[z*src_height*src_width + y*src_width + x] /= weight_sum;
				}
			}
		}
	}
	else if(factor > 1.0f){
		for(unsigned int y = 0; y < src_height; y++){
			for(unsigned int x = 0; x < src_width; x++){
				for(int z = 0; z < (int)floor(src_depth*factor); z++){
					double weight_sum = 0.0;
					dst[z*src_height*src_width + y*src_width + x] = 0.0f;

					for(int i = (int)round((z+0.5)/factor - n); i < (int)round((z+0.5)/factor + n); i++){//lanczos関数は境界で0なのでroundそのままでOK
						double d = abs((z + 0.5)/factor - (i + 0.5));
						double weight = d == 0.0 ? 1.0 : sin(PI*d) * sin(PI*d/n) / (PI*d*(PI*d/n));
						double srcval;

						if(i < 0)srcval = src[0*src_height*src_width + y*src_width + x];
						else if(i >= src_depth)srcval = src[(src_depth-1)*src_height*src_width + y*src_width + x];
						else srcval = src[i*src_height*src_width + y*src_width + x];

						dst[z*src_height*src_width + y*src_width + x] += srcval * weight;
						weight_sum += weight;
					}
					dst[z*src_height*src_width + y*src_width + x] /= weight_sum;
				}
			}
		}
	}
}

void Filter3D::adjustAirspacesBrightness(int ar_dataCh, int blocksize, int angle_d, float ar_minC, float ar_maxC, int iteration, float targetC, float factor_C_Z, int kernelType)
{
	float interval = (ar_maxC - ar_minC)/iteration;
	float curC = ar_minC;
	vector<float> sumlist;
	vector<float> clist;

	const char fnamebased[] = "dmap";
	const char fnamebaseb[] = "border";
	const char fnamebasedb[] = "dmap_border";
	const char fnamebaset[] = "target";
	const char fnamebasef[] = "final";
	const char fnamebase[] = "airspace";
	int imid = 0;
	
	allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, targetC, targetC * factor_C_Z, kernelType);
	int radius = 1;//estimateWallThickness(0, 0.5f)/2;
	//checkCudaErrors( cudaMemcpy(d_Tmp, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	memcpy(dmap, dstdata, imageW*imageH*imageZ*sizeof(float));
	
	memcpy(imgdata, rawdata_resized + imageW*imageH*imageZ*ar_dataCh, imageW*imageH*imageZ*sizeof(float));
	
	int loopcount = 0;
	while(curC < ar_maxC + interval){
		if(curC > ar_maxC)curC = ar_maxC;
		printf("\n[C = %f]\n", curC);

		allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, curC, curC * factor_C_Z, kernelType);
		//AdaptiveThreshold3D_GPU(blocksize, curC, kernelType, true);
		
		float *d_1, *d_2;
		d_1 = d_Output;
		d_2 = d_Buffer;
		if(radius >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < radius / 6; i++){
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
				swap(d_1, d_2);
			}
			if(radius % 6 > 0){
				setSphereMaskKernel6(radius % 6);
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
				swap(d_1, d_2);
			}
		}
		else {
			setSphereMaskKernel6(radius);
			maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
			swap(d_1, d_2);
		}
		checkCudaErrors( cudaDeviceSynchronize() );
		if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		//checkCudaErrors( cudaMemcpy(bufdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		/*
		checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		d_1 = d_Output;
		d_2 = d_Buffer;
		if(radius >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < radius / 6; i++){
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
				swap(d_1, d_2);
			}
			if(radius % 6 > 0){
				setSphereMaskKernel6(radius % 6);
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
				swap(d_1, d_2);
			}
		}
		else {
			setSphereMaskKernel6(radius);
			minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
			swap(d_1, d_2);
		}
		checkCudaErrors( cudaDeviceSynchronize() );
		if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		*/
		//checkCudaErrors( cudaMemcpy(d_Buffer, bufdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		//subtractFloatROIGPU(d_Output, d_Buffer, d_Output, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaMemcpy(d_Buffer, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		subtractFloatROIGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);

		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		saveOrthViewSimple_Numbered(dstdata, fnamebaseb, imid, 100, 100, 15, 1.0, 0);
		checkCudaErrors( cudaMemcpy(dstdata, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		saveOrthViewSimple_Numbered(dstdata, fnamebaset, imid, 100, 100, 15, 1.0, 0);
		
				
		//checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		//saveOrthViewSimple_Numbered(dstdata, fnamebasedb, imid, 100, 100, 15, 1.0, 0);
		//int border_sum = cublasSasum(imageW * imageH * imageZ, d_Output, 1);
		/*
		for(int x = 0; x < imageW; x++){
			for(int y = 0; y < imageH; y++){
				for(int z = 0; z < imageZ; z++){
					if(dstdata[z*imageH*imageW + y*imageW + x] > BINALIZE_LOWER_VAL)border_sum++;
				}
			}
		}
		*/
		checkCudaErrors( cudaMemcpy(d_Tmp, dmap, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		pixelwiseAndGPU(d_Output, d_Output, d_Tmp, imageW, imageH, imageZ);
		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		
		saveOrthViewSimple_Numbered(dstdata, fnamebasef, imid, 100, 100, 15, 1.0, 0);
		imid++;

		int dst_sum = cublasSasum(imageW * imageH * imageZ, d_Output, 1);
		/*
		for(int x = 0; x < imageW; x++){
			for(int y = 0; y < imageH; y++){
				for(int z = 0; z < imageZ; z++){
					if(dstdata[z*imageH*imageW + y*imageW + x] > BINALIZE_LOWER_VAL)dst_sum++;
				}
			}
		}
		*/
		clist.push_back(curC*500);
		//sumlist.push_back((float)dst_sum/(float)border_sum);
		sumlist.push_back((float)dst_sum);
		
		loopcount++;
		curC = ar_minC + interval * loopcount;
	}

	ofstream ofs("list.txt");
	for(int i = 0; i < sumlist.size(); i++){
		cout << clist[i] << ": " << sumlist[i] << endl;
		ofs << clist[i] << "\t" << sumlist[i] << endl;
	}
	ofs.close();
}

bool Filter3D::saveDst3DImage(const char filename[])
{
	stringstream ss;
	ss << filename << ".tif";
	return multif::MultiTiffIO::SaveImageData(ss.str().c_str(), (char *)dstdata, imageW, imageH, imageZ, sizeof(float)*8, 1);
}

void Filter3D::AdaptiveThreshold3D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	checkCudaErrors( cudaMemcpy(d_Input, imgdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );

	float *h_Kernel = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU8(d_Buffer, d_Input, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU8(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU8(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_Buffer, d_Input, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU16(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU16(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_Buffer, d_Input, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU32(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU32(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU64(d_Buffer, d_Input, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU64(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU64(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}

	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	
	delete[] h_Kernel;
}

void Filter3D::threshold3D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU8(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU8(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU8(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU16(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU16(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU32(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU32(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU64(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionColumnsGPU64(d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		convolutionZColumnsGPU64(d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}

	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	
	delete[] h_Kernel;
}

void Filter3D::smoothing3D_GPU(int blocksize, int filterType, bool copyToHostMemory)
{
	float *d_1 = d_Output;
	float *d_2 = d_Buffer;

	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(filterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(filterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	checkCudaErrors( cudaMemcpy(d_1, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	
	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU8(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU8(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionZColumnsGPU8(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU16(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionZColumnsGPU16(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU32(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionZColumnsGPU32(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU64(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU64(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionZColumnsGPU64(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
	}
	
	checkCudaErrors( cudaDeviceSynchronize() );

	if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Filter3D::smoothing2D_GPU(int blocksize, int filterType, bool copyToHostMemory)
{
	float *d_1 = d_Output;
	float *d_2 = d_Buffer;

	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(filterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(filterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	checkCudaErrors( cudaMemcpy(d_1, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	
	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU8(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU8(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU16(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU32(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		
	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU64(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		convolutionColumnsGPU64(d_2, d_1, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		
	}
	
	checkCudaErrors( cudaDeviceSynchronize() );

	if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Filter3D::dilationCubic(int radius, bool copyToHostMemory){
	
	float *d_1 = d_Output;
	float *d_2 = d_Buffer;

	checkCudaErrors( cudaMemcpy(d_Output, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Tmp, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );

	checkCudaErrors( cudaDeviceSynchronize() );

	for(int i = 0; i < radius; i++){
		maximumFilterRows1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		maximumFilterColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		maximumFilterZColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		//upperMaskFittingGPU(d_2, d_Tmp, d_1, 0.1f, imageW, imageH, imageZ);
		//swap(d_1, d_2);
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Filter3D::erosionCubic(int radius, bool copyToHostMemory){
	
	float *d_1 = d_Output;
	float *d_2 = d_Buffer;

//	checkCudaErrors( cudaMemcpy(d_Output, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
//	checkCudaErrors( cudaMemcpy(d_Tmp, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Tmp, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaDeviceSynchronize() );

	for(int i = 0; i < radius; i++){
		minimumFilterRows1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		minimumFilterColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		minimumFilterZColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		//lowerMaskFittingGPU(d_2, d_Tmp, d_1, -0.1f, imageW, imageH, imageZ);
		//swap(d_1, d_2);
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

/*d_1から処理開始*/
void Filter3D::maximumSphereFilter(int radius, float *d_1, float *d_buf, int width, int height, int depth, int roi_x, int roi_y, int roi_z, int roi_w, int roi_h, int roi_d){
	
	float *d_src = d_1;
	
	if(radius >= 4){
		setSphereMaskKernel6(4);
		checkCudaErrors( cudaDeviceSynchronize() );
		for(int i = 0; i < radius / 4; i++){
			maximumSphereFilterGPU6(d_buf, d_1, imageW, imageH, imageZ, roi_x, roi_y, roi_z, roi_w, roi_h, roi_d);
			checkCudaErrors( cudaDeviceSynchronize() );
			swap(d_1, d_buf);
		}
		if(radius % 4 > 0){
			setSphereMaskKernel6(radius % 4);
			checkCudaErrors( cudaDeviceSynchronize() );
			maximumSphereFilterGPU6(d_buf, d_1, imageW, imageH, imageZ, roi_x, roi_y, roi_z, roi_w, roi_h, roi_d);
			swap(d_1, d_buf);
		}
	}
	else {
		setSphereMaskKernel6(radius);
		checkCudaErrors( cudaDeviceSynchronize() );
		maximumSphereFilterGPU6(d_buf, d_1, imageW, imageH, imageZ, roi_x, roi_y, roi_z, roi_w, roi_h, roi_d);
		swap(d_1, d_buf);
	}
	if(d_1 != d_src)checkCudaErrors( cudaMemcpy(d_buf, d_1, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	checkCudaErrors( cudaDeviceSynchronize() );
	
	float dst_sum = cublasSasum(imageW*imageH*imageZ, d_1, 1);
	printf("Sum: %f\n", dst_sum);
}

void Filter3D::minimumSphereFilter(int radius, float *d_1, float *d_buf, int width, int height, int depth, int roi_x, int roi_y, int roi_z, int roi_w, int roi_h, int roi_d){
	
	float *d_src = d_1;
	
	if(radius >= 4){
		setSphereMaskKernel6(4);
		checkCudaErrors( cudaDeviceSynchronize() );
		for(int i = 0; i < radius / 4; i++){
			minimumSphereFilterGPU6(d_buf, d_1, imageW, imageH, imageZ, roi_x, roi_y, roi_z, roi_w, roi_h, roi_d);
			checkCudaErrors( cudaDeviceSynchronize() );
			swap(d_1, d_buf);
		}
		if(radius % 4 > 0){
			setSphereMaskKernel6(radius % 4);
			checkCudaErrors( cudaDeviceSynchronize() );
			minimumSphereFilterGPU6(d_buf, d_1, imageW, imageH, imageZ, roi_x, roi_y, roi_z, roi_w, roi_h, roi_d);
			swap(d_1, d_buf);
		}
	}
	else {
		setSphereMaskKernel6(radius);
		checkCudaErrors( cudaDeviceSynchronize() );
		minimumSphereFilterGPU6(d_buf, d_1, imageW, imageH, imageZ, roi_x, roi_y, roi_z, roi_w, roi_h, roi_d);
		swap(d_1, d_buf);
	}
	if(d_1 != d_src)checkCudaErrors( cudaMemcpy(d_buf, d_1, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );

	checkCudaErrors( cudaDeviceSynchronize() );

	float dst_sum = cublasSasum(imageW*imageH*imageZ, d_1, 1);
	printf("Sum: %f\n", dst_sum);
}

void Filter3D::dilationSpherical(int radius, bool copyToHostMemory){
	
	checkCudaErrors( cudaMemcpy(d_Output, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	
	maximumSphereFilter(radius, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	/*
	for(int z = 0; z < imageZ; z++){
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				segdata[z*imageH*imageW + y*imageW + x] = z;
			}
		}
	}
	*/
	int *d_int;
	checkCudaErrors( cudaMalloc((void **)(&d_int) , imageZ * imageW * imageH * sizeof(int)) );
	checkCudaErrors( cudaMemcpy(d_int, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(start, 0);

	initCuda_Suf3D_Int(d_int, imageW, imageH, imageZ);

	copyToDeviceMemFromSuf3D_Int(d_int, imageW, imageH, imageZ);

	int iterations = 1;
	float *d_1 = d_Output;
	float *d_2 = d_Buffer;
	int *d_Grid;
	int gridsize;
	int *grid;
	for(int i = -1; i < iterations; i++){
		//i == -1 -- warmup iteration
		if(i == 0){
			checkCudaErrors( cudaDeviceSynchronize() );
			cudaEventRecord(start, 0);
		}
		//maximumSphereFilter(radius, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		/*
		maximumFilterRows1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		maximumFilterColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		maximumFilterZColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		*/
		
		//checkCudaErrors( cudaMalloc((void **)(&d_Grid) , imageZ * imageW * imageH * 3 * sizeof(int)) );
		//generateGridDataIntGPU(d_Grid, gridsize, true, d_int, imageW, imageH, imageZ, 33, 33, 33, 1, 1, 1, 0, 0, 0);

		/*
		grid = new int[gridsize];
		checkCudaErrors( cudaMemcpy(grid, d_Grid, gridsize * sizeof(int), cudaMemcpyDeviceToHost) );
		for(unsigned int j = 0; j < 32*32*32; j++)printf("%d ", grid[j]);
		putchar('\n');
		for(unsigned int i = 0; i < gridsize/(32*32*32); i++)printf("%d %d\n", grid[i*32*32*32], grid[(i+1)*32*32*32-1]);
		delete [] grid;
		*/
		checkCudaErrors( cudaDeviceSynchronize() );
		//checkCudaErrors( cudaFree(d_Grid) );
	}

	if(radius >= 4){
		checkCudaErrors( cudaDeviceSynchronize() );
		for(int i = 0; i < radius / 4; i++){
			dilateSegmentsSurfaceSphereGPU(4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		if(radius % 4 > 0){
			dilateSegmentsSurfaceSphereGPU(radius % 4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
	}
	else {
		dilateSegmentsSurfaceSphereGPU(radius, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	if(radius >= 4){
		checkCudaErrors( cudaDeviceSynchronize() );
		for(int i = 0; i < radius / 4; i++){
			erodeSegmentsSurfaceSphereGPU(4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		if(radius % 4 > 0){
			erodeSegmentsSurfaceSphereGPU(radius % 4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
	}
	else {
		erodeSegmentsSurfaceSphereGPU(radius, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	copyToDeviceMemFromSuf3D_Int(d_int, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime / (double)iterations;
	printf("dilationSpherical, Throughput = %.4f MPixels/sec, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n\n", 
		(1.0e-6 * (double)(imageW * imageH * imageZ)/ gpuTime), gpuTime, (imageW * imageH * imageZ), 1, 0);
	checkCudaErrors( cudaMemcpy(segdata, d_int, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );

	//checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

	destructCuda_Suf3D();
	checkCudaErrors( cudaFree(d_int) );
	
	#define KERNEL_RADIUS 6
	#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
	float kernel[KERNEL_LENGTH*KERNEL_LENGTH*KERNEL_LENGTH];
	//int _radius = 2;
	ofstream ofs("code.txt");
	stringstream ss;
	
	for(int z = 0; z < KERNEL_LENGTH; z++){
		for(int y = 0; y < KERNEL_LENGTH; y++){
			for(int x = 0; x < KERNEL_LENGTH; x++){
				if( sqrt( (x - KERNEL_RADIUS)*(x - KERNEL_RADIUS) + (y - KERNEL_RADIUS)*(y - KERNEL_RADIUS) + (z - KERNEL_RADIUS)*(z - KERNEL_RADIUS) ) <= radius )
					kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] = 1.0f;
				else kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] = 0.0f;
				printf("%.1f ", kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x]);
				/*
				if(kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] > 0.0f){
					int dx = x - KERNEL_RADIUS;
					int dy = y - KERNEL_RADIUS;
					int dz = z - KERNEL_RADIUS;
					bool bchk = false;
					ss.str("");
					ss << "\ttmp = ";
					ss << "(" << fixed << setprecision(1);
					if(dz > 0){
						ss << "z+" << (float)abs(dz) << "f <= DATA_D-1.0f";
						bchk = true;
					}
					if(dz < 0){
						ss << "z-" << (float)abs(dz) << "f >= 0.0f";
						bchk = true;
					}
					if(dy > 0){
						if(bchk)ss << " && ";
						ss << "y+" << (float)abs(dy) << "f <= DATA_H-1.0f";
						bchk = true;
					}
					if(dy < 0){
						if(bchk)ss << " && ";
						ss << "y-" << (float)abs(dy) << "f >= 0.0f";
						bchk = true;
					}
					if(dx > 0){
						if(bchk)ss << " && ";
						ss << "x+" << (float)abs(dx) << "f <= DATA_W-1.0f";
						bchk = true;
					}
					if(dx < 0){
						if(bchk)ss << " && ";
						ss << "x-" << (float)abs(dx) << "f >= 0.0f";
						bchk = true;
					}
					ss << ")";

					int maxlen = strlen("\ttmp = (x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f&& z+1.0f <= DATA_D-1.0f)");
					int indent = maxlen - ss.str().length() + 1;
					for(int i = 0; i < indent; i++)ss << " ";
					
					if(bchk)ss << " ? ";
					else ss << "   ";

					ss << "tex3D(tex_Volume, x";
					if(dx >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dx);

					ss << " + 0.5f, y";
					if(dy >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dy);

					ss << " + 0.5f, z";
					if(dz >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dz);

					ss << " + 0.5f)";

					if(bchk)ss << " : FLT_MIN";
					else ss << "          ";

					ss << "; max = (max < tmp) ? tmp : max;\n";

					ofs << ss.str().c_str();
				}
				*/
				/*
				if(kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] > 0.0f){
					int dx = x - KERNEL_RADIUS;
					int dy = y - KERNEL_RADIUS;
					int dz = z - KERNEL_RADIUS;
					bool bchk = false;
					ss.str("");
					ss << "\tif(" << fixed << setprecision(1);
					if(dz > 0){
						ss << "z+" << (float)abs(dz) << "f <= DATA_D-1.0f";
						bchk = true;
					}
					if(dz < 0){
						ss << "z-" << (float)abs(dz) << "f >= 0.0f";
						bchk = true;
					}
					if(dy > 0){
						if(bchk)ss << " && ";
						ss << "y+" << (float)abs(dy) << "f <= DATA_H-1.0f";
						bchk = true;
					}
					if(dy < 0){
						if(bchk)ss << " && ";
						ss << "y-" << (float)abs(dy) << "f >= 0.0f";
						bchk = true;
					}
					if(dx > 0){
						if(bchk)ss << " && ";
						ss << "x+" << (float)abs(dx) << "f <= DATA_W-1.0f";
						bchk = true;
					}
					if(dx < 0){
						if(bchk)ss << " && ";
						ss << "x-" << (float)abs(dx) << "f >= 0.0f";
						bchk = true;
					}
					ss << ")";

					int maxlen = strlen("\tif(x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f&& z+1.0f <= DATA_D-1.0f)");
					int indent = maxlen - ss.str().length() + 1;
					for(int i = 0; i < indent; i++)ss << " ";
					
					if(bchk)ss << " {";
					else ss << "  ";

					ss << "tmp = ";

					ss << "tex3D(tex_Volume, x";
					if(dx >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dx);

					ss << " + 0.5f, y";
					if(dy >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dy);

					ss << " + 0.5f, z";
					if(dz >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dz);

					ss << " + 0.5f)";

					ss << "; max = (max < tmp) ? tmp : max;";

					if(bchk)ss << "}\n";
					else ss << "\n";

					ofs << ss.str().c_str();
				}
				*/
				/*
				if(kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] > 0.0f){
					int dx = x - KERNEL_RADIUS;
					int dy = y - KERNEL_RADIUS;
					int dz = z - KERNEL_RADIUS;
					bool bchk = false;
					ss.str("");
					ss << "\tif(" << fixed << setprecision(1);
					if(dz > 0){
						ss << "z+" << (float)abs(dz) << "f <= DATA_D-1.0f";
						bchk = true;
					}
					if(dz < 0){
						ss << "z-" << (float)abs(dz) << "f >= 0.0f";
						bchk = true;
					}
					if(dy > 0){
						if(bchk)ss << " && ";
						ss << "y+" << (float)abs(dy) << "f <= DATA_H-1.0f";
						bchk = true;
					}
					if(dy < 0){
						if(bchk)ss << " && ";
						ss << "y-" << (float)abs(dy) << "f >= 0.0f";
						bchk = true;
					}
					if(dx > 0){
						if(bchk)ss << " && ";
						ss << "x+" << (float)abs(dx) << "f <= DATA_W-1.0f";
						bchk = true;
					}
					if(dx < 0){
						if(bchk)ss << " && ";
						ss << "x-" << (float)abs(dx) << "f >= 0.0f";
						bchk = true;
					}
					int maxlen = strlen("\tif(x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f&& z+1.0f <= DATA_D-1.0f");
					int indent = maxlen - ss.str().length() + 1;
					for(int i = 0; i < indent; i++)ss << " ";
					ss << " && d > ";
					for(int i = 0; i < (int)log10(radius*radius) - (int)log10(dx*dx + dy*dy + dz*dz); i++)ss << " ";
					ss << dx*dx + dy*dy + dz*dz << ")";
					
					
					if(bchk)ss << " {";
					else ss << "  ";

					ss << "nbr = ";

					ss << "tex3D(tex_VolumeI, x";
					if(dx >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dx);

					ss << ", y";
					if(dy >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dy);

					ss << ", z";
					if(dz >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dz);

					ss << "); if(nbr >= 0){d = ";
					for(int i = 0; i < (int)log10(radius*radius) - (int)log10(dx*dx + dy*dy + dz*dz); i++)ss << " ";
					ss << dx*dx + dy*dy + dz*dz << "; own = nbr;}";
					
					if(bchk)ss << "}\n";
					else ss << "\n";

					ofs << ss.str().c_str();
				}
				*/
				/*
				if(kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] > 0.0f){
					int dx = x - KERNEL_RADIUS;
					int dy = y - KERNEL_RADIUS;
					int dz = z - KERNEL_RADIUS;
					bool bchk = false;
					ss.str("");
					ss << "\tif(" << fixed << setprecision(1);
					if(dz > 0){
						ss << "z+" << (float)abs(dz) << "f <= DATA_D-1.0f";
						bchk = true;
					}
					if(dz < 0){
						ss << "z-" << (float)abs(dz) << "f >= 0.0f";
						bchk = true;
					}
					if(dy > 0){
						if(bchk)ss << " && ";
						ss << "y+" << (float)abs(dy) << "f <= DATA_H-1.0f";
						bchk = true;
					}
					if(dy < 0){
						if(bchk)ss << " && ";
						ss << "y-" << (float)abs(dy) << "f >= 0.0f";
						bchk = true;
					}
					if(dx > 0){
						if(bchk)ss << " && ";
						ss << "x+" << (float)abs(dx) << "f <= DATA_W-1.0f";
						bchk = true;
					}
					if(dx < 0){
						if(bchk)ss << " && ";
						ss << "x-" << (float)abs(dx) << "f >= 0.0f";
						bchk = true;
					}
					int maxlen = strlen("\tif(x+1.0f <= DATA_W-1.0f && y+1.0f <= DATA_H-1.0f&& z+1.0f <= DATA_D-1.0f");
					int indent = maxlen - ss.str().length() + 1;
					for(int i = 0; i < indent; i++)ss << " ";
					ss << " && d > ";
					for(int i = 0; i < (int)log10(radius*radius) - (int)log10(dx*dx + dy*dy + dz*dz); i++)ss << " ";
					ss << dx*dx + dy*dy + dz*dz << ")";
					
					
					if(bchk)ss << " {";
					else ss << "  ";

					ss << "nbr = ";

					ss << "tex3D(tex_VolumeI, x";
					if(dx >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dx);

					ss << ", y";
					if(dy >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dy);

					ss << ", z";
					if(dz >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dz);

					ss << "); if(own != nbr){d = ";
					for(int i = 0; i < (int)log10(radius*radius) - (int)log10(dx*dx + dy*dy + dz*dz); i++)ss << " ";
					ss << dx*dx + dy*dy + dz*dz << "; own = -1;}";
					
					if(bchk)ss << "}\n";
					else ss << "\n";

					ofs << ss.str().c_str();
				}
				*/
				if(kernel[z*KERNEL_LENGTH*KERNEL_LENGTH + y*KERNEL_LENGTH + x] > 0.0f){
					int dx = x - KERNEL_RADIUS;
					int dy = y - KERNEL_RADIUS;
					int dz = z - KERNEL_RADIUS;
					bool bchk = false;
					ss.str("");
					ss << "\t\tif(d > ";
					
					for(int i = 0; i < (int)log10(radius*radius) - (int)log10(dx*dx + dy*dy + dz*dz); i++)ss << " ";
					ss << dx*dx + dy*dy + dz*dz << ")";
					
					
					if(bchk)ss << " {";
					else ss << "  ";

					ss << "nbr = ";

					ss << "tex3D(tex_VolumeI, x";
					if(dx >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dx);

					ss << ", y";
					if(dy >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dy);

					ss << ", z";
					if(dz >= 0)ss << " + ";
					else ss << " - ";
					ss << (float)abs(dz);

					ss << "); if(own != nbr){d = ";
					for(int i = 0; i < (int)log10(radius*radius) - (int)log10(dx*dx + dy*dy + dz*dz); i++)ss << " ";
					ss << dx*dx + dy*dy + dz*dz << "; own = -1;}";
					
					if(bchk)ss << "}\n";
					else ss << "\n";

					ofs << ss.str().c_str();
				}
			}
			putchar('\n');
		}
		putchar('\n');
		putchar('\n');
	}
	
	ofs.close();
	
}

void Filter3D::erosionSpherical(int radius, bool copyToHostMemory){
	
	checkCudaErrors( cudaMemcpy(d_Output, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	minimumSphereFilter(radius, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Filter3D::closingSpherical(int radius, bool copyToHostMemory){
	
	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	maximumSphereFilter(radius, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	minimumSphereFilter(radius, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);

	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Filter3D::openingSpherical(int radius, bool copyToHostMemory){
	
	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	minimumSphereFilter(radius, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	maximumSphereFilter(radius, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);

	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Filter3D::DoG(int blocksize, bool copyToHostMemory)
{
	if(isEmpty)return;

	smoothing3D_GPU(blocksize, FILTER_GAUSSIAN, true);
	for(int z = 0; z < imageZ; z++)
		for(int y = 0; y < imageH; y++)
			for(int x = 0; x < imageW; x++)
				dstdata[z*imageH*imageW + y*imageW + x] = 2.0f*imgdata[z*imageH*imageW + y*imageW + x] - dstdata[z*imageH*imageW + y*imageW + x];
	/*
	for(int z = 0; z < imageZ; z++){
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				dstdata[z*imageH*imageW + y*imageW + x] = 0.0f;
				if(x > 0 && x < imageW-1 && y > 0 && y < imageH-1 && z > 0 && z < imageZ-1){
					for(int dz = -1; dz <= 1; dz++){
						for(int dy = -1; dy <= 1; dy++){
							for(int dx = -1; dx <= 1; dx++){
								if(dx == 0 && dy == 0 && dz == 0)continue;
								if( bufdata[(z+dz)*imageH*imageW + (y+dy)*imageW + x+dx]*bufdata[(z-dz)*imageH*imageW + (y-dy)*imageW + x-dx] < 0.0f &&
								   abs(bufdata[(z+dz)*imageH*imageW + (y+dy)*imageW + x+dx] - bufdata[(z-dz)*imageH*imageW + (y-dy)*imageW + x-dx]) > 0.003f )
									dstdata[z*imageH*imageW + y*imageW + x] = 1.0f;
							}
						}
					}
				}
			}
		}
	}
	*/
}

void Filter3D::hMinimaTransform3D_GPU(float h, int chk_interval, bool copyToHostMemory)
{
		
	if(isEmpty || chk_interval <= 0)return;

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("<<hMinimaTransform3D_GPU: h = %f  chk_interval = %d>>\n", h, chk_interval);

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(start, 0);
	checkCudaErrors( cudaMemcpy(d_Output, imgdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	addFloatValueGPU(d_Output, imageW, imageH, imageZ, h);

	float *d_1 = d_Output;
	float *d_2 = d_Buffer;
	float *before = new float[imageW*imageH*imageZ];
	checkCudaErrors( cudaMemcpy(before, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

	int count = 0;
	int total_count = 0;
	while(1){
		if(count >= chk_interval){
			checkCudaErrors( cudaMemcpy(bufdata, d_1, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
			bool isDiff = false;
			for(long long i = 0; i < (long long)imageW*(long long)imageH*(long long)imageZ; i++){
				if(bufdata[i] != before[i]){
					isDiff = true;
					break;
				}
			}
			if(!isDiff)break;
			checkCudaErrors( cudaMemcpy(before, d_1, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
			count = 0;
		}
		minimumFilterRows1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		minimumFilterColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		minimumFilterZColumns1ROIGPU(d_2, d_1, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		lowerMaskFittingGPU(d_2, d_Input, d_1, 0.0f, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		swap(d_1, d_2);
		count++;
		total_count++;
	}
	
	//segmentation用に調整
	subtractFloatROIGPU(d_2, d_1, d_Input, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	swap(d_1, d_2);
	thresholdGPU(d_2, d_1, imageW, imageH, imageZ, 0.00001f);
	checkCudaErrors( cudaDeviceSynchronize() );
	swap(d_1, d_2);
	inverseGPU(d_2, d_1, BINALIZE_UPPER_VAL, BINALIZE_LOWER_VAL, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	swap(d_1, d_2);

	if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("hMinimaTransformGPU, Time = %.5f s, Size = %u Pixels, Iteration = %d\n", gpuTime, (imageW * imageH * imageZ), total_count);

	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	/*
	printf("Running GPU Maximum Filter vHGW (%u identical iterations)...\n", iterations);
	for(int i = -1; i < iterations; i++){
		//i == -1 -- warmup iteration
		if(i == 0){
			checkCudaErrors( cudaDeviceSynchronize() );
			cudaEventRecord(start, 0);
		}
		maximumFilterRows1GPU_vHGW(d_Output, d_Input, imageW, imageH, imageZ);
	}

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	gpuTime = 0.001 * elapsedTime / (double)iterations;
	printf("maximumFilterRows1GPU_vHGW, Throughput = %.4f MPixels/sec, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n\n", 
		(1.0e-6 * (double)(imageW * imageH * imageZ)/ gpuTime), gpuTime, (imageW * imageH * imageZ), 1, 0);
	
	//if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	*/
	delete [] before;
}

void Filter3D::AdaptiveThreshold2D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	AdaptiveThreshold3D_GPU(blocksize, constC, thresholdType, copyToHostMemory);

	float constC2 = constC/* - 10.0f/500.0f*/;

	float *h_Kernel = new float[kernel_length];
	float *h_Kernel_half = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);
	
	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU8(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC2*4);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionColumnsGPU16(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Buffer, d_Buffer, d_Input, imageW, imageH, imageZ, constC2*4);
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU16(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Buffer, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
/*		//full length
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU16(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);
		checkCudaErrors( cudaDeviceSynchronize() );

		//left
		float sum = 0.0f;
		for(int i = 0; i <= blocksize; i++){h_Kernel_half[i] = h_Kernel[i]; sum += h_Kernel[i];}
		for(int i = 0; i <= blocksize; i++)h_Kernel_half[i] /= sum;
		for(int i = 0; i < blocksize; i++)h_Kernel_half[blocksize + 1 + i] = 0.0f;
		setConvolutionKernel16(h_Kernel_half, blocksize);
		convolutionZColumnsGPU16(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Buffer, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );

		//right
		sum = 0.0f;
		for(int i = 0; i <= blocksize; i++)h_Kernel_half[i] = 0.0f;
		for(int i = 0; i < blocksize; i++){h_Kernel_half[blocksize + 1 + i] = h_Kernel[blocksize + 1 + i]; sum += h_Kernel[blocksize + 1 + i];}
		for(int i = 0; i < blocksize; i++)h_Kernel_half[blocksize + 1 + i] /= sum;
		setConvolutionKernel16(h_Kernel_half, blocksize);
		convolutionZColumnsGPU16(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Buffer, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
*/
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC2*4);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionColumnsGPU32(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Buffer, d_Buffer, d_Input, imageW, imageH, imageZ, constC2*4);
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU32(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Buffer, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
/*
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU32(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);

		checkCudaErrors( cudaDeviceSynchronize() );
*/	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU64(d_Buffer, d_Input, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC2);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	/*
	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU8(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU8(d_Buffer, d_Output, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU16(d_Buffer, d_Output, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU32(d_Buffer, d_Output, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU64(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU64(d_Buffer, d_Output, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Buffer, d_Input, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	*/
	checkCudaErrors( cudaMemcpy(d_Tmp, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(bufdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

	for(unsigned long i = 1; i < imageZ*imageH*imageW; i++){
		if(dstdata[i] < 1.0f)dstdata[i] = bufdata[i];
	}

	delete[] h_Kernel;
	delete[] h_Kernel_half;
}

void Filter3D::threshold2D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU8(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU16(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU32(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU64(d_Buffer, d_Tmp, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
/*
	if(blocksize <= 8){
		setConvolutionKernel8(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU8(d_Output, d_Tmp, imageW, imageH, imageZ);
		convolutionColumnsGPU8(d_Buffer, d_Output, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 16){
		setConvolutionKernel16(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_Output, d_Tmp, imageW, imageH, imageZ);
		convolutionColumnsGPU16(d_Buffer, d_Output, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 32){
		setConvolutionKernel32(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_Output, d_Tmp, imageW, imageH, imageZ);
		convolutionColumnsGPU32(d_Buffer, d_Output, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize <= 64){
		setConvolutionKernel64(h_Kernel, blocksize);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU64(d_Output, d_Tmp, imageW, imageH, imageZ);
		convolutionColumnsGPU64(d_Buffer, d_Output, imageW, imageH, imageZ);
		thresholdGPU(d_Output, d_Buffer, imageW, imageH, imageZ, constC);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
*/
	if(copyToHostMemory)checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	
	delete[] h_Kernel;
}

void Filter3D::drawCircle(float longi, float lati, int radius)
{
	Point3i center(imageW/2, imageH/2, imageZ/2);
	float *list = new float[(radius*2 + 1)*(radius*2 + 1)*4];
	float *h_Kernel = new float[radius*2 + 1];
	getMeanFilter1D(h_Kernel, radius*2 + 1);
/*	
	float rot[3][3] = { { cos(lati)*cos(longi), cos(lati)*sin(longi), sin(lati) },
						{ -sin(longi)		  , cos(longi)			, 0 },
						{ sin(lati)*cos(longi), sin(lati)*sin(longi), cos(lati) } };
*/
	double rot[3][3] = {{ cos(longi)*cos(lati),  sin(longi), cos(longi)*sin(lati)  },
						{ -sin(longi)*cos(lati), cos(longi), -sin(longi)*sin(lati) },
						{ sin(lati),			 0,			 cos(lati)			   }};
	
	int count = 0;
	double k_sum = 0.0;
	for(int z = -radius; z <= radius; z++){
		for(int y = -radius; y <= radius; y++){
			if( z*z + y*y <= radius*radius){
				double tmpx, tmpy, tmpz, tmpk;

				tmpx = 0.0;
				tmpy = y;
				tmpz = z;
				
				list[count*4]     = rot[0][0]*tmpx + rot[0][1]*tmpy + rot[0][2]*tmpz;
				list[count*4 + 1] = rot[1][0]*tmpx + rot[1][1]*tmpy + rot[1][2]*tmpz;
				list[count*4 + 2] = rot[2][0]*tmpx + rot[2][1]*tmpy + rot[2][2]*tmpz;
				list[count*4 + 3] = h_Kernel[z + radius] * h_Kernel[y + radius];
				k_sum += list[count*4 + 3];
				count++;
			}
		}
	}
	for(int i = 0; i < count; i++)list[i*4 + 3] /= k_sum;
	
	/*
	double chk_sum = 0.0;
	for(int i = 0; i < count; i++)chk_sum += list[i*4 + 3];
	cout << "chk kernel sum: " << chk_sum << endl;
	*/

	for(int i = 0; i < count; i++){
		if(    (center.z + (int)list[i*4 + 2]) < imageZ && (center.z + (int)list[i*4 + 2]) >= 0
			&& (center.y + (int)list[i*4 + 1]) < imageH && (center.y + (int)list[i*4 + 1]) >= 0
			&& (center.x + (int)list[i*4])     < imageW && (center.x + (int)list[i*4])     >= 0 )
			dstdata[(center.z + (int)round(list[i*4 + 2]))*imageH*imageW + (center.y + (int)round(list[i*4 + 1]))*imageW + center.x + (int)round(list[i*4])] = 1.0f;
	}

	delete [] list;
	delete [] h_Kernel;
}

void Filter3D::drawLineSegments(int radius, int level)
{
	Point3i center(imageW/2, imageH/2, imageZ/2);
	float *plist = new float[radius*2*4];
	float dx, dy, dz;
	double a_interval = (PI / 2.0) / (double)level;

	float *h_Kernel = new float[radius*2 + 1];
	getMeanFilter1D(h_Kernel, radius*2 + 1);

	float lati;
	float longi;

	lati = PI/2.0f;
	longi = 0.0f;
	dx = cos(longi) * cos(lati);
	dy = sin(longi) * cos(lati);
	dz = sin(lati);
	for(int i = 0; i < radius; i++){
		plist[(i*2)*4]       = -dx*(i+1);
		plist[(i*2)*4 + 1]   = -dy*(i+1);
		plist[(i*2)*4 + 2]   = -dz*(i+1);
		plist[(i*2)*4 + 3]   = h_Kernel[radius - (i + 1)];
		plist[(i*2+1)*4]     = dx*(i+1);
		plist[(i*2+1)*4 + 1] = dy*(i+1);
		plist[(i*2+1)*4 + 2] = dz*(i+1);
		plist[(i*2+1)*4 + 3] = h_Kernel[radius + (i + 1)];
	}
	for(int i = 0; i < radius*2; i++){
		if(    (center.z + (int)plist[i*4 + 2]) < imageZ && (center.z + (int)plist[i*4 + 2]) >= 0
			&& (center.y + (int)plist[i*4 + 1]) < imageH && (center.y + (int)plist[i*4 + 1]) >= 0
			&& (center.x + (int)plist[i*4])     < imageW && (center.x + (int)plist[i*4])     >= 0 )
			dstdata[(center.z + (int)round(plist[i*4 + 2]))*imageH*imageW + (center.y + (int)round(plist[i*4 + 1]))*imageW + center.x + (int)round(plist[i*4])] = 1.0f;
	}

	for(int b = 0; b < level*2; b++){
		for(int a = 1; a < level*2; a++){
			lati = PI/2.0f - a_interval*a;
			longi = a_interval*b;
			dx = cos(longi) * cos(lati);
			dy = sin(longi) * cos(lati);
			dz = sin(lati);
			for(int i = 0; i < radius; i++){
				plist[(i*2)*4]       = -dx*(i+1);
				plist[(i*2)*4 + 1]   = -dy*(i+1);
				plist[(i*2)*4 + 2]   = -dz*(i+1);
				plist[(i*2)*4 + 3]   = h_Kernel[radius - (i + 1)];
				plist[(i*2+1)*4]     = dx*(i+1);
				plist[(i*2+1)*4 + 1] = dy*(i+1);
				plist[(i*2+1)*4 + 2] = dz*(i+1);
				plist[(i*2+1)*4 + 3] = h_Kernel[radius + (i + 1)];
			}
			for(int i = 0; i < radius*2; i++){
				if(    (center.z + (int)plist[i*4 + 2]) < imageZ && (center.z + (int)plist[i*4 + 2]) >= 0
					&& (center.y + (int)plist[i*4 + 1]) < imageH && (center.y + (int)plist[i*4 + 1]) >= 0
					&& (center.x + (int)plist[i*4])     < imageW && (center.x + (int)plist[i*4])     >= 0 )
					dstdata[(center.z + (int)round(plist[i*4 + 2]))*imageH*imageW + (center.y + (int)round(plist[i*4 + 1]))*imageW + center.x + (int)round(plist[i*4])] = 1.0f;
			}
		}
	}
	
	delete [] plist;
	delete [] h_Kernel;
}

void Filter3D::drawIcoSphere(int radius, int level)
{
	const Point3f v[] = { Point3f( 0.000000f, -0.000000f,  1.000000f),
						  Point3f( 0.723600f,  0.525720f,  0.447215f),
						  Point3f(-0.276385f,  0.850640f,  0.447215f),
						  Point3f(-0.894425f, -0.000000f,  0.447215f),
						  Point3f(-0.276385f, -0.850640f,  0.447215f),
						  Point3f( 0.723600f, -0.525720f,  0.447215f),
						  Point3f( 0.276385f,  0.850640f, -0.447215f),
						  Point3f(-0.723600f,  0.525720f, -0.447215f),
						  Point3f(-0.723600f, -0.525720f, -0.447215f),
						  Point3f( 0.276385f, -0.850640f, -0.447215f),
						  Point3f( 0.894425f,  0.000000f, -0.447215f),
						  Point3f(-0.000000f,  0.000000f, -1.000000f) };
	vector<Point3f> ver(begin(v), end(v));

	const Point3i f[] = { Point3i(0, 1, 2),
						  Point3i(1, 0, 5),
						  Point3i(0, 2, 3),
						  Point3i(0, 3, 4),
						  Point3i(0, 4, 5),
						  Point3i(1, 5, 10),
						  Point3i(2, 1, 6),
						  Point3i(3, 2, 7),
						  Point3i(4, 3, 8),
						  Point3i(5, 4, 9),
						  Point3i(1, 10, 6),
						  Point3i(2, 6, 7),
						  Point3i(3, 7, 8),
						  Point3i(4, 8, 9),
						  Point3i(5, 9, 10),
						  Point3i(6, 10, 11),
						  Point3i(7, 6, 11),
						  Point3i(8, 7, 11),
						  Point3i(9, 8, 11),
						  Point3i(9, 10, 11) };

	vector<Point3i> faces(begin(f), end(f));

	for(int i = 0; i < level; i++){
		vector<Point3i> newFaces;
		for(int j = 0; j < faces.size(); j++){
			int i0 = faces[j].x;
			int i1 = faces[j].y;
			int i2 = faces[j].z;
			
			Point3f midpoint;
			int m01, m12, m02;
			vector<Point3f>::iterator pos;
			
			midpoint = (ver[i0] + ver[i1]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m01 = pos - ver.begin();
			else { m01 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i1] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m12 = pos - ver.begin();
			else { m12 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i0] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m02 = pos - ver.begin();
			else { m02 = ver.size(); ver.push_back(midpoint); }
			
			newFaces.push_back(Point3i(i0,  m01, m02));
			newFaces.push_back(Point3i(i1,  m12, m01));
			newFaces.push_back(Point3i(i2,  m02, m12));
			newFaces.push_back(Point3i(m02, m01, m12));
		}
		faces.clear();
		std::copy(newFaces.begin(), newFaces.end(), back_inserter(faces));
	}

	vector<Point3f> e;
	for(int i = 0; i < ver.size(); i++)
		if(std::find(e.begin(), e.end(), ver[i] * -1) == e.end())e.push_back(ver[i]);
	
	cout << "var_size: " << ver.size() << "  e_size: " << e.size() << endl;

	for(int k = 0; k < e.size(); k++)e[k] = e[k] * (1.0 / sqrt(e[k].x*e[k].x + e[k].y*e[k].y + e[k].z*e[k].z));

	Point3i center(imageW/2, imageH/2, imageZ/2);
	float *plist = new float[radius*2*4];
	float *h_Kernel = new float[radius*2 + 1];
	getMeanFilter1D(h_Kernel, radius*2 + 1);

	for(int i = 0; i < e.size(); i++){
		for(int j = 0; j < radius; j++){
				plist[(j*2)*4]       = -e[i].x*(j+1);
				plist[(j*2)*4 + 1]   = -e[i].y*(j+1);
				plist[(j*2)*4 + 2]   = -e[i].z*(j+1);
				plist[(j*2)*4 + 3]   = h_Kernel[radius - (j + 1)];
				plist[(j*2+1)*4]     = e[i].x*(j+1);
				plist[(j*2+1)*4 + 1] = e[i].y*(j+1);
				plist[(j*2+1)*4 + 2] = e[i].z*(j+1);
				plist[(j*2+1)*4 + 3] = h_Kernel[radius + (j + 1)];
		}
		for(int j = 0; j < radius*2; j++){
			if(    (center.z + (int)round(plist[j*4 + 2])) < imageZ && (center.z + (int)round(plist[j*4 + 2])) >= 0
				&& (center.y + (int)round(plist[j*4 + 1])) < imageH && (center.y + (int)round(plist[j*4 + 1])) >= 0
				&& (center.x + (int)round(plist[j*4]))     < imageW && (center.x + (int)round(plist[j*4]))     >= 0 )
				dstdata[(center.z + (int)round(plist[j*4 + 2]))*imageH*imageW + (center.y + (int)round(plist[j*4 + 1]))*imageW + center.x + (int)round(plist[j*4])] = 1.0f;
		}
	}

	delete [] plist;
	delete h_Kernel;
}

void Filter3D::allAngleLinearConvolutionMinGPU(float *output, float *input, int blocksize, int angle_d, int FilterType, float *argmin_lati, float *argmin_longi)
{
	if(isEmpty || blocksize <= 0 || angle_d < 1)return;

	double a_interval = (PI / 2.0) / (double)angle_d;
	int kernel_length = blocksize*2 + 1;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("<<allAngleLinearConvolutionMinGPU>>\n");
	/*
	float *h_Kernel = new float[kernel_length];
	
	if(FilterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(FilterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);
	*/
	initCuda_LCT(input, imageW, imageH, imageZ);
	checkCudaErrors( cudaMemcpy(d_Input, input, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	fillFloatGPU(d_Output, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Buffer, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Tmp, imageW, imageH, imageZ, FLT_MAX);
	checkCudaErrors( cudaDeviceSynchronize() );

	printf("Running allAngleLinearConvolutionMin...\n");
	
	cudaEventRecord(start, 0);

	//2D用
	for(int r = blocksize; r > 0; r--){
		int len = r*2 + 1; 
		float *h_Kernel = new float[len];

		if(FilterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, len);
		if(FilterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, len);
		
		for(int b = 0; b < angle_d*2; b++){
			lineSegmentConvolutionMinGPU(d_Output, imageW, imageH, imageZ, r, h_Kernel, d_Buffer, 0.0f, d_Tmp, a_interval*b);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		delete h_Kernel;
	}

	/*
	//z-axis
	lineSegmentConvolutionMinGPU(d_Output, imageW, imageH, imageZ, blocksize, h_Kernel, d_Buffer, PI/2.0f, d_Tmp, 0);//各voxelで結果がd_Output以下の時だけ書き換える
	//CircleConvolutionMinGPU(d_Output, imageW, imageH, imageZ, blocksize, h_Kernel, d_Buffer, PI/2.0f, d_Tmp, 0);
	checkCudaErrors( cudaDeviceSynchronize() );

	for(int b = 0; b < angle_d*2; b++){
		for(int a = 1; a < angle_d*2; a++){
			lineSegmentConvolutionMinGPU(d_Output, imageW, imageH, imageZ, blocksize, h_Kernel, d_Buffer, PI/2.0f - a_interval*a, d_Tmp, a_interval*b);
			//CircleConvolutionMinGPU(d_Output, imageW, imageH, imageZ, blocksize, h_Kernel, d_Buffer, PI/2.0f - a_interval*a, d_Tmp, a_interval*b);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
	}
	*/
	/*
	int blocksize_v = 0;
	float *plist = new float[(blocksize*2 + 1)*(blocksize_v*2 + 1)*(blocksize_v*2 + 1)*4];
	float *h_Kernel_v = new float[blocksize_v*2 + 1];

	if(FilterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel_v, blocksize_v*2 + 1);
	if(FilterType == FILTER_MEAN)getMeanFilter1D(h_Kernel_v, blocksize_v*2 + 1);
	for(int b = 0; b < angle_d*2; b++){
		for(int a = 0; a < angle_d*2; a++){
			float longi = PI/2.0f - a_interval*a;
			float lati = a_interval*b;

			const double rot[3][3] = {{ cos(longi)*cos(lati),  sin(longi), cos(longi)*sin(lati)  },
									  { -sin(longi)*cos(lati), cos(longi), -sin(longi)*sin(lati) },
									  { sin(lati),			   0,		    cos(lati)			 }};

			int count = 0;
			for(int x = -blocksize; x <= blocksize; x++){
				for(int y = -blocksize_v; y <= blocksize_v; y++){
					for(int z = -blocksize_v; z <= blocksize_v; z++){
						double tmpx, tmpy, tmpz;

						tmpx = x;
						tmpy = y;
						tmpz = z;

						plist[count*4]     = rot[0][0]*tmpx + rot[0][1]*tmpy + rot[0][2]*tmpz;
						plist[count*4 + 1] = rot[1][0]*tmpx + rot[1][1]*tmpy + rot[1][2]*tmpz;
						plist[count*4 + 2] = rot[2][0]*tmpx + rot[2][1]*tmpy + rot[2][2]*tmpz;
						plist[count*4 + 3] = h_Kernel[x + blocksize] * h_Kernel_v[y + blocksize_v] * h_Kernel_v[z + blocksize_v];
						count++;
					}
				}
			}

			setPointList_LCT(plist, (blocksize*2 + 1)*(blocksize_v*2 + 1)*(blocksize_v*2 + 1));
			checkCudaErrors( cudaDeviceSynchronize() );
			linearConvolutionMinGPU_Texture(d_Output, imageW, imageH, imageZ, d_Buffer, lati, d_Tmp, longi);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
	}
	delete [] plist;
	delete [] h_Kernel_v;
	*/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("allAngleLinearConvolutionMin, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n", gpuTime, (imageW * imageH * imageZ), 1, 0);

	checkCudaErrors( cudaMemcpy(output, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	if(argmin_lati != NULL)checkCudaErrors( cudaMemcpy(argmin_lati, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	if(argmin_longi != NULL)checkCudaErrors( cudaMemcpy(argmin_longi, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	destructCuda_LCT();

	//delete [] h_Kernel;
}

void Filter3D::allAngleLinearConvolutionMinGPU_geodesic(float *output, float *input, int radius, int level, int FilterType, float *argmin_lati, float *argmin_longi)
{
	if(isEmpty || radius <= 0 || level < 1)return;

	int kernel_length = radius * 2 + 1;
	cudaEvent_t start, stop;
	
	const Point3f v[] = { Point3f( 0.000000f, -0.000000f,  1.000000f),
						  Point3f( 0.723600f,  0.525720f,  0.447215f),
						  Point3f(-0.276385f,  0.850640f,  0.447215f),
						  Point3f(-0.894425f, -0.000000f,  0.447215f),
						  Point3f(-0.276385f, -0.850640f,  0.447215f),
						  Point3f( 0.723600f, -0.525720f,  0.447215f),
						  Point3f( 0.276385f,  0.850640f, -0.447215f),
						  Point3f(-0.723600f,  0.525720f, -0.447215f),
						  Point3f(-0.723600f, -0.525720f, -0.447215f),
						  Point3f( 0.276385f, -0.850640f, -0.447215f),
						  Point3f( 0.894425f,  0.000000f, -0.447215f),
						  Point3f(-0.000000f,  0.000000f, -1.000000f) };
	vector<Point3f> ver(begin(v), end(v));

	const Point3i f[] = { Point3i(0, 1, 2),
						  Point3i(1, 0, 5),
						  Point3i(0, 2, 3),
						  Point3i(0, 3, 4),
						  Point3i(0, 4, 5),
						  Point3i(1, 5, 10),
						  Point3i(2, 1, 6),
						  Point3i(3, 2, 7),
						  Point3i(4, 3, 8),
						  Point3i(5, 4, 9),
						  Point3i(1, 10, 6),
						  Point3i(2, 6, 7),
						  Point3i(3, 7, 8),
						  Point3i(4, 8, 9),
						  Point3i(5, 9, 10),
						  Point3i(6, 10, 11),
						  Point3i(7, 6, 11),
						  Point3i(8, 7, 11),
						  Point3i(9, 8, 11),
						  Point3i(9, 10, 11) };

	vector<Point3i> faces(begin(f), end(f));

	for(int i = 0; i < level; i++){
		vector<Point3i> newFaces;
		for(int j = 0; j < faces.size(); j++){
			int i0 = faces[j].x;
			int i1 = faces[j].y;
			int i2 = faces[j].z;
			
			Point3f midpoint;
			int m01, m12, m02;
			vector<Point3f>::iterator pos;
			
			midpoint = (ver[i0] + ver[i1]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m01 = pos - ver.begin();
			else { m01 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i1] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m12 = pos - ver.begin();
			else { m12 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i0] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m02 = pos - ver.begin();
			else { m02 = ver.size(); ver.push_back(midpoint); }
			
			newFaces.push_back(Point3i(i0,  m01, m02));
			newFaces.push_back(Point3i(i1,  m12, m01));
			newFaces.push_back(Point3i(i2,  m02, m12));
			newFaces.push_back(Point3i(m02, m01, m12));
		}
		faces.clear();
		std::copy(newFaces.begin(), newFaces.end(), back_inserter(faces));
	}

	vector<Point3f> e;
	for(int i = 0; i < ver.size(); i++)
		if(std::find(e.begin(), e.end(), ver[i] * -1) == e.end())e.push_back(ver[i]);
	
	cout << "var_size: " << ver.size() << "  e_size: " << e.size() << endl;

	for(int k = 0; k < e.size(); k++)e[k] = e[k] * (1.0 / sqrt(e[k].x*e[k].x + e[k].y*e[k].y + e[k].z*e[k].z));
	/*
	float *plist = new float[(radius*2 + 1)*4];
	float *h_Kernel = new float[kernel_length];
	
	if(FilterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(FilterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("<<allAngleLinearConvolutionMinGPU>>\n");
	
	initCuda_LCT(input, imageW, imageH, imageZ);
	checkCudaErrors( cudaMemcpy(d_Input, input, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	fillFloatGPU(d_Output, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Buffer, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Tmp, imageW, imageH, imageZ, FLT_MAX);
	checkCudaErrors( cudaDeviceSynchronize() );

	printf("Running allAngleLinearConvolutionMin_geodesic...\n");
	
	cudaEventRecord(start, 0);

	for(int i = 0; i < e.size(); i++){
		for(int j = 0; j < radius*2 + 1; j++){
				plist[j*4]       = e[i].x*(j-radius);
				plist[j*4 + 1]   = e[i].y*(j-radius);
				plist[j*4 + 2]   = e[i].z*(j-radius);
				plist[j*4 + 3]   = h_Kernel[j];
		}
		double sq = sqrt(e[i].x*e[i].x + e[i].y*e[i].y);
		float lati = acos(sq);
		float longi = (sq == 0) ? 0.0f : asin(e[i].x/sq);
		setPointList_LCT(plist, radius*2 + 1);
		checkCudaErrors( cudaDeviceSynchronize() );
		linearConvolutionMinGPU_Texture(d_Output, imageW, imageH, imageZ, d_Buffer, lati, d_Tmp, longi);
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	*/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("<<allAngleLinearConvolutionMinGPU>>\n");

	initCuda_LCT(input, imageW, imageH, imageZ);
	checkCudaErrors( cudaMemcpy(d_Input, input, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	fillFloatGPU(d_Output, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Buffer, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Tmp, imageW, imageH, imageZ, FLT_MAX);
	checkCudaErrors( cudaDeviceSynchronize() );

	printf("Running allAngleLinearConvolutionMin_geodesic...\n");

	cudaEventRecord(start, 0);

	for(int r = radius; r > 0; r--){
		int len = r*2 + 1; 
		float *plist = new float[(r*2 + 1)*4];
		float *h_Kernel = new float[len];

		if(FilterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, len);
		if(FilterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, len);

		for(int i = 0; i < e.size(); i++){
			for(int j = 0; j < r*2 + 1; j++){
				plist[j*4]       = e[i].x*(j-r);
				plist[j*4 + 1]   = e[i].y*(j-r);
				plist[j*4 + 2]   = e[i].z*(j-r);
				plist[j*4 + 3]   = h_Kernel[j];
			}
			double sq = sqrt(e[i].x*e[i].x + e[i].y*e[i].y);
			float lati = acos(sq);
			float longi = (sq == 0) ? 0.0f : asin(e[i].x/sq);
			setPointList_LCT(plist, r*2 + 1);
			checkCudaErrors( cudaDeviceSynchronize() );
			linearConvolutionMinGPU_Texture(d_Output, imageW, imageH, imageZ, d_Buffer, lati, d_Tmp, longi);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		delete [] plist;
		delete h_Kernel;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("allAngleLinearConvolutionMin, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n", gpuTime, (imageW * imageH * imageZ), 1, 0);

	checkCudaErrors( cudaMemcpy(output, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	if(argmin_lati != NULL)checkCudaErrors( cudaMemcpy(argmin_lati, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	if(argmin_longi != NULL)checkCudaErrors( cudaMemcpy(argmin_longi, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	destructCuda_LCT();

	//delete [] plist;
	//delete h_Kernel;
}

//サンプル点の密度分布からMeanで通常のGaussianと同じような効果になる
void Filter3D::allAngleLinearConvolutionMinGPU_geodesic_v2(float *output, float *input, int radius, int level, int FilterType, float *argmin_lati, float *argmin_longi)
{
	if(isEmpty || radius <= 0 || level < 1)return;

	int kernel_length = radius * 2 + 1;
	cudaEvent_t start, stop;
	
	const Point3f v[] = { Point3f( 0.000000f, -0.000000f,  1.000000f),
						  Point3f( 0.723600f,  0.525720f,  0.447215f),
						  Point3f(-0.276385f,  0.850640f,  0.447215f),
						  Point3f(-0.894425f, -0.000000f,  0.447215f),
						  Point3f(-0.276385f, -0.850640f,  0.447215f),
						  Point3f( 0.723600f, -0.525720f,  0.447215f),
						  Point3f( 0.276385f,  0.850640f, -0.447215f),
						  Point3f(-0.723600f,  0.525720f, -0.447215f),
						  Point3f(-0.723600f, -0.525720f, -0.447215f),
						  Point3f( 0.276385f, -0.850640f, -0.447215f),
						  Point3f( 0.894425f,  0.000000f, -0.447215f),
						  Point3f(-0.000000f,  0.000000f, -1.000000f) };
	vector<Point3f> ver(begin(v), end(v));

	const Point3i f[] = { Point3i(0, 1, 2),
						  Point3i(1, 0, 5),
						  Point3i(0, 2, 3),
						  Point3i(0, 3, 4),
						  Point3i(0, 4, 5),
						  Point3i(1, 5, 10),
						  Point3i(2, 1, 6),
						  Point3i(3, 2, 7),
						  Point3i(4, 3, 8),
						  Point3i(5, 4, 9),
						  Point3i(1, 10, 6),
						  Point3i(2, 6, 7),
						  Point3i(3, 7, 8),
						  Point3i(4, 8, 9),
						  Point3i(5, 9, 10),
						  Point3i(6, 10, 11),
						  Point3i(7, 6, 11),
						  Point3i(8, 7, 11),
						  Point3i(9, 8, 11),
						  Point3i(9, 10, 11) };

	vector<Point3i> faces(begin(f), end(f));

	for(int i = 0; i < level; i++){
		vector<Point3i> newFaces;
		for(int j = 0; j < faces.size(); j++){
			int i0 = faces[j].x;
			int i1 = faces[j].y;
			int i2 = faces[j].z;
			
			Point3f midpoint;
			int m01, m12, m02;
			vector<Point3f>::iterator pos;
			
			midpoint = (ver[i0] + ver[i1]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m01 = pos - ver.begin();
			else { m01 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i1] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m12 = pos - ver.begin();
			else { m12 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i0] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m02 = pos - ver.begin();
			else { m02 = ver.size(); ver.push_back(midpoint); }
			
			newFaces.push_back(Point3i(i0,  m01, m02));
			newFaces.push_back(Point3i(i1,  m12, m01));
			newFaces.push_back(Point3i(i2,  m02, m12));
			newFaces.push_back(Point3i(m02, m01, m12));
		}
		faces.clear();
		std::copy(newFaces.begin(), newFaces.end(), back_inserter(faces));
	}

	vector<Point3f> e;
	for(int i = 0; i < ver.size(); i++)
		if(std::find(e.begin(), e.end(), ver[i] * -1) == e.end())e.push_back(ver[i]);
	
	cout << "var_size: " << ver.size() << "  e_size: " << e.size() << endl;

	for(int k = 0; k < e.size(); k++)e[k] = e[k] * (1.0 / sqrt(e[k].x*e[k].x + e[k].y*e[k].y + e[k].z*e[k].z));

	float *plist = new float[radius*2*4];
	float *h_Kernel = new float[kernel_length];
	
	if(FilterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(FilterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	h_Kernel[radius] = 0.0f;
	double k_sum = 0.0;
	for(int i = 0; i < kernel_length; i++)k_sum += h_Kernel[i];
	for(int i = 0; i < kernel_length; i++)h_Kernel[i] /= k_sum;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("<<allAngleLinearConvolutionMinGPU>>\n");
	
	initCuda_LCT(input, imageW, imageH, imageZ);
	checkCudaErrors( cudaMemcpy(d_Input, input, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	
	printf("Running allAngleLinearConvolutionMin_geodesic...\n");
	
	cudaEventRecord(start, 0);

	fillFloatGPU(d_Output, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Buffer, imageW, imageH, imageZ, 0.0f);
	fillFloatGPU(d_Tmp, imageW, imageH, imageZ, 0.0f);
	checkCudaErrors( cudaDeviceSynchronize() );

	for(int i = 0; i < e.size(); i++){
		for(int j = 0; j < radius; j++){
				plist[(j*2)*4]       = -e[i].x*(j+1);
				plist[(j*2)*4 + 1]   = -e[i].y*(j+1);
				plist[(j*2)*4 + 2]   = -e[i].z*(j+1);
				plist[(j*2)*4 + 3]   = h_Kernel[radius - (j + 1)];
				plist[(j*2+1)*4]     = e[i].x*(j+1);
				plist[(j*2+1)*4 + 1] = e[i].y*(j+1);
				plist[(j*2+1)*4 + 2] = e[i].z*(j+1);
				plist[(j*2+1)*4 + 3] = h_Kernel[radius + (j + 1)];
		}
		double sq = sqrt(e[i].x*e[i].x + e[i].y*e[i].y);
		float lati = acos(sq);
		float longi = (sq == 0) ? 0.0f : asin(e[i].x/sq);
		fillFloatGPU(d_Output, imageW, imageH, imageZ, FLT_MAX);
		setPointList_LCT(plist, radius*2);
		checkCudaErrors( cudaDeviceSynchronize() );
		linearConvolutionMinGPU_Texture(d_Output, imageW, imageH, imageZ, NULL, lati, NULL, longi);
		checkCudaErrors( cudaDeviceSynchronize() );
		countUpN_LessThan_GPU(d_Buffer, d_Input, d_Output, d_Tmp, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	checkCudaErrors( cudaDeviceSynchronize() );
	divFloatGPU(d_Output, d_Buffer, d_Tmp, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("allAngleLinearConvolutionMin, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n", gpuTime, (imageW * imageH * imageZ), 1, 0);
	
	checkCudaErrors( cudaMemcpy(output, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	destructCuda_LCT();

	if(argmin_lati != NULL){
		fillFloatGPU(d_Buffer, imageW, imageH, imageZ, 0.0f);
		checkCudaErrors( cudaMemcpy(argmin_lati, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	}
	if(argmin_longi != NULL){
		fillFloatGPU(d_Tmp, imageW, imageH, imageZ, 0.0f);
		checkCudaErrors( cudaMemcpy(argmin_longi, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	}

	delete [] plist;
	delete h_Kernel;
}

//線分片側のみ
void Filter3D::allAngleLinearConvolutionMinGPU_geodesic_v3(float *output, float *input, int radius, int level, int FilterType, float *argmin_lati, float *argmin_longi)
{
	if(isEmpty || radius <= 0 || level < 1)return;

	int kernel_length = radius * 2 + 1;
	cudaEvent_t start, stop;
	
	const Point3f v[] = { Point3f( 0.000000f, -0.000000f,  1.000000f),
						  Point3f( 0.723600f,  0.525720f,  0.447215f),
						  Point3f(-0.276385f,  0.850640f,  0.447215f),
						  Point3f(-0.894425f, -0.000000f,  0.447215f),
						  Point3f(-0.276385f, -0.850640f,  0.447215f),
						  Point3f( 0.723600f, -0.525720f,  0.447215f),
						  Point3f( 0.276385f,  0.850640f, -0.447215f),
						  Point3f(-0.723600f,  0.525720f, -0.447215f),
						  Point3f(-0.723600f, -0.525720f, -0.447215f),
						  Point3f( 0.276385f, -0.850640f, -0.447215f),
						  Point3f( 0.894425f,  0.000000f, -0.447215f),
						  Point3f(-0.000000f,  0.000000f, -1.000000f) };
	vector<Point3f> ver(begin(v), end(v));

	const Point3i f[] = { Point3i(0, 1, 2),
						  Point3i(1, 0, 5),
						  Point3i(0, 2, 3),
						  Point3i(0, 3, 4),
						  Point3i(0, 4, 5),
						  Point3i(1, 5, 10),
						  Point3i(2, 1, 6),
						  Point3i(3, 2, 7),
						  Point3i(4, 3, 8),
						  Point3i(5, 4, 9),
						  Point3i(1, 10, 6),
						  Point3i(2, 6, 7),
						  Point3i(3, 7, 8),
						  Point3i(4, 8, 9),
						  Point3i(5, 9, 10),
						  Point3i(6, 10, 11),
						  Point3i(7, 6, 11),
						  Point3i(8, 7, 11),
						  Point3i(9, 8, 11),
						  Point3i(9, 10, 11) };

	vector<Point3i> faces(begin(f), end(f));

	for(int i = 0; i < level; i++){
		vector<Point3i> newFaces;
		for(int j = 0; j < faces.size(); j++){
			int i0 = faces[j].x;
			int i1 = faces[j].y;
			int i2 = faces[j].z;
			
			Point3f midpoint;
			int m01, m12, m02;
			vector<Point3f>::iterator pos;
			
			midpoint = (ver[i0] + ver[i1]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m01 = pos - ver.begin();
			else { m01 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i1] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m12 = pos - ver.begin();
			else { m12 = ver.size(); ver.push_back(midpoint); }
			
			midpoint = (ver[i0] + ver[i2]) * 0.5f;
			pos = std::find(ver.begin(), ver.end(), midpoint);
			if(pos != ver.end())m02 = pos - ver.begin();
			else { m02 = ver.size(); ver.push_back(midpoint); }
			
			newFaces.push_back(Point3i(i0,  m01, m02));
			newFaces.push_back(Point3i(i1,  m12, m01));
			newFaces.push_back(Point3i(i2,  m02, m12));
			newFaces.push_back(Point3i(m02, m01, m12));
		}
		faces.clear();
		std::copy(newFaces.begin(), newFaces.end(), back_inserter(faces));
	}

	
	for(int k = 0; k < ver.size(); k++)ver[k] = ver[k] * (1.0 / sqrt(ver[k].x*ver[k].x + ver[k].y*ver[k].y + ver[k].z*ver[k].z));

	float *plist = new float[(radius + 1)*4];
	float *h_Kernel = new float[kernel_length];
	
	if(FilterType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(FilterType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	double k_sum = 0.0;
	for(int i = radius; i < kernel_length; i++)k_sum += h_Kernel[i];
	for(int i = radius; i < kernel_length; i++)h_Kernel[i] /= k_sum;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("<<allAngleLinearConvolutionMinGPU>>\n");
	
	initCuda_LCT(input, imageW, imageH, imageZ);
	checkCudaErrors( cudaMemcpy(d_Input, input, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	fillFloatGPU(d_Output, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Buffer, imageW, imageH, imageZ, FLT_MAX);
	fillFloatGPU(d_Tmp, imageW, imageH, imageZ, FLT_MAX);
	checkCudaErrors( cudaDeviceSynchronize() );

	printf("Running allAngleLinearConvolutionMin_geodesic...\n");
	
	cudaEventRecord(start, 0);

	for(int i = 0; i < ver.size(); i++){
		for(int j = 0; j <= radius; j++){
				plist[j*4]       = ver[i].x * j;
				plist[j*4 + 1]   = ver[i].y * j;
				plist[j*4 + 2]   = ver[i].z * j;
				plist[j*4 + 3]   = h_Kernel[radius + j];
		}
		double sq = sqrt(ver[i].x*ver[i].x + ver[i].y*ver[i].y);
		float lati = acos(sq);
		float longi = (sq == 0) ? 0.0f : asin(ver[i].x/sq);
		setPointList_LCT(plist, radius + 1);
		checkCudaErrors( cudaDeviceSynchronize() );
		linearConvolutionMinGPU_Texture(d_Output, imageW, imageH, imageZ, d_Buffer, lati, d_Tmp, longi);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("allAngleLinearConvolutionMin, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n", gpuTime, (imageW * imageH * imageZ), 1, 0);

	checkCudaErrors( cudaMemcpy(output, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	if(argmin_lati != NULL)checkCudaErrors( cudaMemcpy(argmin_lati, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	if(argmin_longi != NULL)checkCudaErrors( cudaMemcpy(argmin_longi, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	destructCuda_LCT();


	delete [] plist;
	delete h_Kernel;
}

void Filter3D::allAngleLinearAdaptiveThresholdGPU(int blocksize, int angle_d, float constC_XY, float constC_Z, int thresholdType)
{
	if(isEmpty || blocksize <= 0 || angle_d < 1)return;

	//allAngleLinearConvolutionMinGPU(dstdata, imgdata, blocksize, angle_d, thresholdType, bufdata, NULL);
	allAngleLinearConvolutionMinGPU_geodesic(dstdata, imgdata, blocksize, angle_d, thresholdType, bufdata, NULL);
	//allAngleLinearConvolutionMinGPU_geodesic_v2(dstdata, imgdata, blocksize, angle_d, thresholdType, bufdata, NULL);
	//allAngleLinearConvolutionMinGPU_geodesic_v3(dstdata, imgdata, blocksize, angle_d, thresholdType, bufdata, NULL);

	checkCudaErrors( cudaMemcpy(d_Buffer, bufdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	
	calcCorrectionTerms_DSADTH_GPU(d_Buffer, d_Buffer, constC_XY, constC_Z, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	addFloatGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	binarizeGPU(d_Output, d_Output, d_Input, imageW, imageH, imageZ, 0.0f);
	
	/*
	addFloatValueGPU(d_Output, imageW, imageH, imageZ, -constC_XY);
	checkCudaErrors( cudaDeviceSynchronize() );
	binarizeGPU(d_Output, d_Output, d_Input, imageW, imageH, imageZ, 0.0f);
	*/

	checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

void Filter3D::savitzky_golay_Z(const int npast, const int nfuture,const int deriv = 0, const int poly_order = 2)
{
	if(isEmpty)return;

	int imj, ipj, k, kk, mm;
	float fac, sum;

	Mat a(poly_order + 1, poly_order + 1, CV_32F);
	Mat b(poly_order + 1, 1, CV_32F, 0.0f);

	const int np = npast + nfuture + 1;

	for (ipj = 0; ipj <= 2 * poly_order; ipj++)
	{
		sum = 0.0;
		if ( ipj == 0 )
			sum = 1.0;
		for (k = 1; k <= nfuture; k++)
			sum += pow( (double)k, (double)ipj);
		for (k = 1; k <= npast; k++)
			sum += pow( (double)(-k), (double)ipj);
		mm = min( ipj, 2 * poly_order - ipj );
		for (imj = -mm; imj <= mm; imj += 2)
			a.at<float>((ipj + imj)/2, (ipj - imj)/2) = sum;

	}


	b.at<float>(deriv,0) = 1.0;

	solve(a, b, b, DECOMP_LU);

	float *h_Kernel = new float [np];

	for (k = 0; k < np; k++)
		h_Kernel[k] = 0.0f;

	for (k = -npast, kk = np - 1; k <= nfuture; k++, kk--)
	{
		sum = b.at<float>(0, 0);
		fac = 1.0f;
		for (mm = 1; mm <= poly_order; mm++)
		{
			fac *= k;
			sum += b.at<float>(mm, 0) * fac;
		}

		h_Kernel[kk] = sum;
	}

	for (k = 0; k < np; k++)printf("%f ", h_Kernel[k]);
	printf("\n");

	int radius = npast;

	if(radius <= 8){
		setConvolutionKernel8(h_Kernel, radius);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU8(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU8(d_Output, d_Output, imageW, imageH, imageZ);
		convolutionZColumnsGPU8(d_Output, d_Output, imageW, imageH, imageZ);
		
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(radius <= 16){
		setConvolutionKernel16(h_Kernel, radius);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU16(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU16(d_Output, d_Output, imageW, imageH, imageZ);
		convolutionZColumnsGPU16(d_Output, d_Output, imageW, imageH, imageZ);
		
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(radius <= 32){
		setConvolutionKernel32(h_Kernel, radius);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU32(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU32(d_Output, d_Output, imageW, imageH, imageZ);
		convolutionZColumnsGPU32(d_Output, d_Output, imageW, imageH, imageZ);
		
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(radius <= 64){
		setConvolutionKernel64(h_Kernel, radius);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionRowsGPU64(d_Output, d_Input, imageW, imageH, imageZ);
		convolutionColumnsGPU64(d_Output, d_Output, imageW, imageH, imageZ);
		convolutionZColumnsGPU64(d_Output, d_Output, imageW, imageH, imageZ);
		
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	
	checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	
	delete [] h_Kernel;
	/*
	vector<int> up_id;
	vector<int> up_type; // 0:minima 1:maxima
	float peak_th = 0.4f;
	float *tmpZ = new float [imageZ];
	
	for(unsigned int x = 0; x < imageW; x++){
		for(unsigned int y = 0; y < imageH; y++){
			if(dstdata[0*imageW*imageH + y*imageW + x] < dstdata[1*imageW*imageH + y*imageW + x]){up_id.push_back(0); up_type.push_back(0);}
			if(dstdata[0*imageW*imageH + y*imageW + x] > dstdata[1*imageW*imageH + y*imageW + x]){up_id.push_back(0); up_type.push_back(1);}
			for(unsigned int z = 1; z < imageZ - 1; z++){
				if(dstdata[z*imageW*imageH + y*imageW + x] < dstdata[(z-1)*imageW*imageH + y*imageW + x] && dstdata[z*imageW*imageH + y*imageW + x] < dstdata[(z+1)*imageW*imageH + y*imageW + x]){up_id.push_back(z); up_type.push_back(0);}
				if(dstdata[z*imageW*imageH + y*imageW + x] > dstdata[(z-1)*imageW*imageH + y*imageW + x] && dstdata[z*imageW*imageH + y*imageW + x] > dstdata[(z+1)*imageW*imageH + y*imageW + x]){up_id.push_back(z); up_type.push_back(1);}
			}
			if(dstdata[(imageZ-1)*imageW*imageH + y*imageW + x] < dstdata[(imageZ-2)*imageW*imageH + y*imageW + x]){up_id.push_back(imageZ-1); up_type.push_back(0);}
			if(dstdata[(imageZ-1)*imageW*imageH + y*imageW + x] > dstdata[(imageZ-2)*imageW*imageH + y*imageW + x]){up_id.push_back(imageZ-1); up_type.push_back(1);}

			for(k = 0; k < up_id.size(); k++){
				float left_def, right_def, max_def;
				int st, ed;
				if(up_type[k] == 1){
					if(k == 0){left_def = 0.0f; st = 0;}
					else {left_def = dstdata[up_id[k]*imageW*imageH + y*imageW + x] - dstdata[up_id[k-1]*imageW*imageH + y*imageW + x]; st = up_id[k-1];}
					if(k == up_id.size() - 1){right_def = 0.0f; ed = imageZ - 1;}
					else {right_def = dstdata[up_id[k]*imageW*imageH + y*imageW + x] - dstdata[up_id[k+1]*imageW*imageH + y*imageW + x]; ed = up_id[k+1];}
					max_def = max(left_def, right_def);
					for(int i = st; i <= ed; i++){
						if(dstdata[up_id[k]*imageW*imageH + y*imageW + x] - dstdata[i*imageW*imageH + y*imageW + x] < max_def * peak_th)dstdata[i*imageW*imageH + y*imageW + x] = 1.0f;
						else dstdata[i*imageW*imageH + y*imageW + x] = 0.0f;
					}
				}
			}
			up_id.clear();
			up_type.clear();
		}
	}
	*/
}

void Filter3D::binarySegmentatorLow(const float *data, Box3D &bbox, float th_val, int connect_type, int x, int y, int z, int id)
{
	if(isEmpty)return;
	if(!(x >= 0 && x < imageW && y >= 0 && y < imageH && z >=0 && z < imageZ))return;
	if(segments == NULL)return;

	queue<long long> que;
	int count = 0;
	int cx, cy, cz;

	long long tmp;
	
	segdata[z*imageH*imageW + y*imageW + x] = id;

	if(segments->size() < id + 1)segments->resize(id + 1, vector<Point3i>::vector());
	else if(!segments->at(id).empty())segments->at(id).clear();
	segments->at(id).push_back(Point3i(x, y, z));


	que.push(((long long)x << 16 | (long long)y) << 16 | (long long)z);

	int minx, miny, minz, maxx, maxy, maxz;
	minx = x;
	miny = y;
	minz = z;
	maxx = x;
	maxy = y;
	maxz = z;

	while(que.size()){
		count++;
		tmp = que.front();
		que.pop();
		
		cx = tmp >> 32;
		cy = (tmp >> 16) & 0xFFFF;
		cz = tmp & 0xFFFF;

		//printf("x: %d  y: %d  z: %d\n", cx, cy, cz);

		if(connect_type == SEGMENT_CONNECT6){
			if(cz + 1 < imageZ)
				if(data[(cz + 1)*imageH*imageW + cy*imageW + cx] <= th_val && segdata[(cz + 1)*imageH*imageW + cy*imageW + cx] == SEGMENT_BLANK){
					segdata[(cz + 1)*imageH*imageW + cy*imageW + cx] = id;
					que.push(((long long)cx << 16 | (long long)cy) << 16 | (long long)(cz+1));
					if(cz + 1 > maxz)maxz = cz + 1;
					(*segments)[id].push_back(Point3i(cx, cy, cz + 1));
				}
			if(cz - 1 >= 0)
				if(data[(cz - 1)*imageH*imageW + cy*imageW + cx] <= th_val && segdata[(cz - 1)*imageH*imageW + cy*imageW + cx] == SEGMENT_BLANK){
					segdata[(cz - 1)*imageH*imageW + cy*imageW + cx] = id;
					que.push(((long long)cx << 16 | (long long)cy) << 16 | (long long)(cz-1));
					if(cz - 1 < minz)minz = cz - 1;
					(*segments)[id].push_back(Point3i(cx, cy, cz - 1));
				}
			if(cy + 1 < imageH)
				if(data[cz*imageH*imageW + (cy + 1)*imageW + cx] <= th_val && segdata[cz*imageH*imageW + (cy + 1)*imageW + cx] == SEGMENT_BLANK){
					segdata[cz*imageH*imageW + (cy + 1)*imageW + cx] = id;
					que.push(((long long)cx << 16 | (long long)(cy+1)) << 16 | (long long)cz);
					if(cy + 1 > maxy)maxy = cy + 1;
					(*segments)[id].push_back(Point3i(cx, cy + 1, cz));
				}
			if(cy - 1 >= 0)
				if(data[cz*imageH*imageW + (cy - 1)*imageW + cx] <= th_val && segdata[cz*imageH*imageW + (cy - 1)*imageW + cx] == SEGMENT_BLANK){
					segdata[cz*imageH*imageW + (cy - 1)*imageW + cx] = id;
					que.push(((long long)cx << 16 | (long long)(cy-1)) << 16 | (long long)cz);
					if(cy - 1 < miny)miny = cy - 1;
					(*segments)[id].push_back(Point3i(cx, cy - 1, cz));
				}
			if(cx + 1 < imageW)
				if(data[cz*imageH*imageW + cy*imageW + (cx + 1)] <= th_val && segdata[cz*imageH*imageW + cy*imageW + (cx + 1)] == SEGMENT_BLANK){
					segdata[cz*imageH*imageW + cy*imageW + (cx + 1)] = id;
					que.push(((long long)(cx+1) << 16 | (long long)cy) << 16 | (long long)cz);
					if(cx + 1 > maxx)maxx = cx + 1;
					(*segments)[id].push_back(Point3i(cx + 1, cy, cz));
				}
			if(cx - 1 >= 0)
				if(data[cz*imageH*imageW + cy*imageW + (cx - 1)] <= th_val && segdata[cz*imageH*imageW + cy*imageW + (cx - 1)] == SEGMENT_BLANK){
					segdata[cz*imageH*imageW + cy*imageW + (cx - 1)] = id;
					que.push(((long long)(cx-1) << 16 | (long long)cy) << 16 | (long long)cz);
					if(cx - 1 < minx)minx = cx - 1;
					(*segments)[id].push_back(Point3i(cx - 1, cy, cz));
				}

		}
		else if(connect_type == SEGMENT_CONNECT26){
			
			for(int dz = -1; dz <= 1; dz++){
				for(int dy = -1; dy <= 1; dy++){
					for(int dx = -1; dx <= 1; dx++){

						if(dz != 0 || dy != 0 || dx != 0){
							if(cz + dz < imageZ && cz + dz >= 0 && cy + dy < imageH && cy + dy >= 0 && cx + dx < imageW && cx + dx >= 0){
								if(data[(cz + dz)*imageH*imageW + (cy + dy)*imageW + cx + dx] <= th_val && segdata[(cz + dz)*imageH*imageW + (cy + dy)*imageW + cx + dx] == SEGMENT_BLANK){
									segdata[(cz + dz)*imageH*imageW + (cy + dy)*imageW + cx + dx] = id;
									que.push(((long long)(cx + dx) << 16 | (long long)(cy + dy)) << 16 | (long long)(cz + dz));
									if(cx + dx > maxx)maxx = cx + dx;
									if(cx + dx < minx)minx = cx + dx;
									if(cy + dy > maxy)maxy = cy + dy;
									if(cy + dy < miny)miny = cy + dy;
									if(cz + dz > maxz)maxz = cz + dz;
									if(cz + dz < minz)minz = cz + dz;
									(*segments)[id].push_back(Point3i(cx + dx, cy + dy, cz + dz));
								}
							}
						}

					}
				}
			}

		}
	}

	bbox.x = minx;
	bbox.y = miny;
	bbox.z = minz;
	bbox.width  = maxx - minx + 1;
	bbox.height = maxy - miny + 1;
	bbox.depth  = maxz - minz + 1;

	/*
	if(connect_type == SEGMENT_CONNECT6){
		if(z + 1 < imageZ)
			if(data[(z + 1)*imageH*imageW + y*imageW + x] <= th_val && segdata[(z + 1)*imageH*imageW + y*imageW + x] == -1)
				binarySegmentatorLow(data, th_val, connect_type, x, y, z+1, id);
		if(z - 1 >= 0)
			if(data[(z - 1)*imageH*imageW + y*imageW + x] <= th_val && segdata[(z - 1)*imageH*imageW + y*imageW + x] == -1)
				binarySegmentatorLow(data, th_val, connect_type, x, y, z-1, id);
		if(y + 1 < imageH)
			if(data[z*imageH*imageW + (y + 1)*imageW + x] <= th_val && segdata[z*imageH*imageW + (y + 1)*imageW + x] == -1)
				binarySegmentatorLow(data, th_val, connect_type, x, y+1, z, id);
		if(y - 1 >= 0)
			if(data[z*imageH*imageW + (y - 1)*imageW + x] <= th_val && segdata[z*imageH*imageW + (y - 1)*imageW + x] == -1)
				binarySegmentatorLow(data, th_val, connect_type, x, y-1, z, id);
		if(x + 1 < imageW)
			if(data[z*imageH*imageW + y*imageW + x + 1] <= th_val && segdata[z*imageH*imageW + y*imageW + x + 1] == -1)
				binarySegmentatorLow(data, th_val, connect_type, x+1, y, z, id);
		if(x - 1 >= 0)
			if(data[z*imageH*imageW + y*imageW + x - 1] <= th_val && segdata[z*imageH*imageW + y*imageW + x - 1] == -1)
				binarySegmentatorLow(data, th_val, connect_type, x-1, y, z, id);

	}
	else if(connect_type == SEGMENT_CONNECT18){

	}
	else if(connect_type == SEGMENT_CONNECT26){

	}
	*/
}

void Filter3D::binarySegmentationLow(const float *data, float th_val, int connect_type, int min_size, bool clearOldResult, const int *seed)
{
	int id = 0;
	Box3D t_boundingBox;
	
	printf("<<binarySegmentationLow(SegmentsValidation OFF)>>\n");

	int *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	
	if(invalid_seg_bbox != NULL)std::vector<Box3D>().swap(*invalid_seg_bbox);
	clear2DVector(invalid_segments);
	if(clearOldResult){
		if(seg_bbox != NULL)std::vector<Box3D>().swap(*seg_bbox);
		clear2DVector(segments);
		fillIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
		checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	}
	else {
		id = segments->size();
		//SEGMENT_INVALID SEGMENT_TOOSMALL SEGMENT_NOTSAVEDをSEGMENT_BLANKに置き換える
		//SEGMENT_INVALID SEGMENT_TOOSMALL SEGMENT_NOTSAVEDはすべてSEGMENT_BLANKより小さい値に設定してある
		checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
		setMinValIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
		checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	}
	
	for(int z = 0; z < imageZ; z++){
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				if(seed != NULL)if(seed[z*imageH*imageW + y*imageW + x] < 0)continue;
				if(data[z*imageH*imageW + y*imageW + x] <= th_val && segdata[z*imageH*imageW + y*imageW + x] == SEGMENT_BLANK){
					binarySegmentatorLow(data, t_boundingBox, th_val, connect_type, x, y, z, id);
					if((*segments)[id].size() > min_size){
						//printf("(segment %d) x: %d y: %d z: %d size: %d\n", id, segments->at(id).at(0).x, segments->at(id).at(0).y, segments->at(id).at(0).z, segments->at(id).size());
						id++;
						seg_bbox->push_back(t_boundingBox);
					}
					else{
						for(int i = 0; i < (*segments)[id].size(); i++)
							segdata[(*segments)[id][i].z*imageW*imageH + (*segments)[id][i].y*imageW + (*segments)[id][i].x] = SEGMENT_TOOSMALL;
						
						segments->pop_back();
					}
				}
			}
		}
	}
	printf("binarySegmentation: segments %d\n", id);
	std::vector<float>().swap(*seg_colors);
	for(int i = 0; i < segments->size(); i++)seg_colors->push_back(rand() % 360);

	if(selected_segment != NULL)std::vector<int>().swap(*selected_segment);

	checkCudaErrors( cudaFree(d_Seg) );
}

void Filter3D::binarySegmentationLow(const float *data, float th_val, int connect_type, int min_size, bool clearOldResult, int noise_th, int wall_th, int minInvalidStructureVol, bool saveVaildSegments, bool saveInvalidSegments)
{
	int id = 0;
	Box3D t_boundingBox;
	
	int noise_rad = noise_th >> 1;
	int wall_rad = wall_th >> 1;
	int rad_max = MAX(noise_rad, wall_rad);
	//int rad_max =MAX(wall_rad, dilation_rad);

	Box3D bb;
	
	printf("<<binarySegmentationLow(SegmentsValidation ON): noise_rad = %d wall_rad = %d  minInvalidStructureVol = %d>>\n", noise_rad, wall_rad, minInvalidStructureVol);
	
	int *d_Seg, *d_Seg2;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	checkCudaErrors( cudaMalloc((void **)(&d_Seg2) , imageZ * imageW * imageH * sizeof(int)) );

	//メモリリーク？
	if(invalid_seg_bbox != NULL)std::vector<Box3D>().swap(*invalid_seg_bbox);
	clear2DVector(invalid_segments);
	if(clearOldResult){
		if(seg_bbox != NULL)std::vector<Box3D>().swap(*seg_bbox);
		clear2DVector(segments);
		//checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
		fillIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
		checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	}
	else {
		id = segments->size();
		//SEGMENT_INVALID SEGMENT_TOOSMALL SEGMENT_NOTSAVEDをSEGMENT_BLANKに置き換える
		//SEGMENT_INVALID SEGMENT_TOOSMALL SEGMENT_NOTSAVEDはすべてSEGMENT_BLANKより小さい値に設定してある
		checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
		setMinValIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
		checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	}

	for(int z = 0; z < imageZ; z++){
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				if(data[z*imageH*imageW + y*imageW + x] <= th_val && segdata[z*imageH*imageW + y*imageW + x] == SEGMENT_BLANK){
					binarySegmentatorLow(data, t_boundingBox, th_val, connect_type, x, y, z, id);
					if((*segments)[id].size() > min_size){
						id++;
						seg_bbox->push_back(t_boundingBox);
					}
					else{
						for(int i = 0; i < (*segments)[id].size(); i++)
							segdata[(*segments)[id][i].z*imageW*imageH + (*segments)[id][i].y*imageW + (*segments)[id][i].x] = SEGMENT_TOOSMALL;
						segments->pop_back();
					}
				}
			}
		}
	}
	printf("binarySegmentation: segments %d\n", id);

	checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
	initCuda_Suf3D_Int(d_Seg, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	
	if(wall_rad >= 4){
		checkCudaErrors( cudaDeviceSynchronize() );
		for(int i = 0; i < wall_rad / 4; i++){
			dilateSegmentsSurfaceSphereGPU(4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		if(wall_rad % 4 > 0){
			dilateSegmentsSurfaceSphereGPU(wall_rad % 4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
	}
	else {
		dilateSegmentsSurfaceSphereGPU(wall_rad, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	if(wall_rad >= 4){
		checkCudaErrors( cudaDeviceSynchronize() );
		for(int i = 0; i < wall_rad / 4; i++){
			erodeSegmentsSurfaceSphereGPU_v2(4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		if(wall_rad % 4 > 0){
			erodeSegmentsSurfaceSphereGPU_v2(wall_rad % 4, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
	}
	else {
		erodeSegmentsSurfaceSphereGPU_v2(wall_rad, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	copyToDeviceMemFromSuf3D_Int(d_Seg2, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	destructCuda_Suf3D();

	segmentsMaskIntGPU(d_Seg2, d_Seg2, d_Seg, imageW*imageH*imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );

	int *inv_segdata = new int[imageW*imageH*imageZ];
	int *inv_sum = new int[segments->size()];
	checkCudaErrors( cudaMemcpy(inv_segdata, d_Seg2, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	for(int i = 0; i < segments->size(); i++)inv_sum[i] = 0;
	
	for(int z = 0; z < imageZ; z++){
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				if(inv_segdata[z*imageH*imageW + y*imageW + x] >= 0)inv_sum[inv_segdata[z*imageH*imageW + y*imageW + x]]++;
			}
		}
	}

	/*segmentデータベクタからの不正segment要素の削除は行わない*/
	vector<Box3D>::iterator b_ite = seg_bbox->begin();
	vector< vector<Point3i> >::iterator s_ite = segments->begin();
	int i = 0;
	while( b_ite != seg_bbox->end() && s_ite != segments->end() ){
		if(inv_sum[i] > minInvalidStructureVol){
			printf("[invalidSeg] id: %d size: %d inv_sum: %d\n", i, s_ite->size(), inv_sum[i]);
			replacementIntROIGPU(d_Seg, i, SEGMENT_BLANK, imageW, imageH, imageZ, (*b_ite).x, (*b_ite).y, (*b_ite).z, (*b_ite).width, (*b_ite).height, (*b_ite).depth);
			checkCudaErrors( cudaDeviceSynchronize() );
			invalid_seg_bbox->push_back(*b_ite);
			invalid_segments->push_back(*s_ite);
			std::vector<cv::Point3i>().swap((*segments)[i]);
		}
		b_ite++;
		s_ite++;
		i++;
	}
	checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );

	std::vector<float>().swap(*seg_colors);
	for(int i = 0; i < segments->size(); i++)seg_colors->push_back(rand() % 360);

	if(selected_segment != NULL)std::vector<int>().swap(*selected_segment);

	delete [] inv_segdata;
	delete [] inv_sum;
	checkCudaErrors( cudaMemcpy(d_Output, data, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaFree(d_Seg) );
	checkCudaErrors( cudaFree(d_Seg2) );
}

void Filter3D::resetSegmentColors(){
	if(seg_colors == NULL)return;
	
	std::vector<float>().swap(*seg_colors);
	for(int i = 0; i < segments->size(); i++)seg_colors->push_back(rand() % 360);

}

void Filter3D::segmentsValidation(float *data, int noise_th, int wall_th, int minInvalidStructureVol){
	
	if(isEmpty || segments->empty())return;

	int noise_rad = noise_th >> 1;
	int wall_rad = wall_th >> 1;
	int rad_max = MAX(noise_rad, wall_rad);
	
	printf("<<segmentsValidation: noise_rad = %d wall_rad = %d  minInvalidStructureVol = %d>>\n", noise_rad, wall_rad, minInvalidStructureVol);

	cudaEvent_t start, stop;
	Box3D bb;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
		
	float *d_buffer2;
	checkCudaErrors( cudaMalloc((void **)(&d_buffer2),   imageZ * imageW * imageH * sizeof(float)) );
	fillFloatGPU(d_Tmp, imageW, imageH, imageZ, 0.0f);

	vector< vector<Point3i> >::iterator it = segments->begin();
	vector<Box3D>::iterator bbox_it = seg_bbox->begin();
	int count = 0;
	while(it != segments->end() || bbox_it != seg_bbox->end()){

		fillFloatGPU(d_Output, imageW, imageH, imageZ, 0.0f);
		checkCudaErrors( cudaMemcpy(d_Buffer, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

		for(int i = 0; i < it->size(); i++){
			dstdata[it->at(i).z*imageW*imageH + it->at(i).y*imageW + it->at(i).x] = 0.8;
		}
		checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_buffer2, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );//d_buffer2はsegmentの元データ

		bb.x = (bbox_it->x - rad_max >= 0) ? bbox_it->x - rad_max : 0;
		bb.y = (bbox_it->y - rad_max >= 0) ? bbox_it->y - rad_max : 0;
		bb.z = (bbox_it->z - rad_max >= 0) ? bbox_it->z - rad_max : 0;
		bb.width  = (bbox_it->x + bbox_it->width  + rad_max < imageW) ? bbox_it->x + bbox_it->width  + rad_max - bb.x : imageW - bb.x;
		bb.height = (bbox_it->y + bbox_it->height + rad_max < imageH) ? bbox_it->y + bbox_it->height + rad_max - bb.y : imageH - bb.y;
		bb.depth  = (bbox_it->z + bbox_it->depth  + rad_max < imageZ) ? bbox_it->z + bbox_it->depth  + rad_max - bb.z : imageZ - bb.z;

		
		float *d_1, *d_2;
		
		d_1 = d_buffer2;
		d_2 = d_Buffer;
		
		if(noise_rad >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < noise_rad / 6; i++){
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(noise_rad % 6 > 0){
				setSphereMaskKernel6(noise_rad % 6);
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", noise_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(noise_rad);
			maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", noise_rad);
		}

		if(noise_rad >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < noise_rad / 6; i++){
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(noise_rad % 6 > 0){
				setSphereMaskKernel6(noise_rad % 6);
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", noise_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(noise_rad);
			minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", noise_rad);
		}
		if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_buffer2, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		


		d_1 = d_Output;
		d_2 = d_Buffer;
		if(wall_rad >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < wall_rad / 6; i++){
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(wall_rad % 6 > 0){
				setSphereMaskKernel6(wall_rad % 6);
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(wall_rad);
			maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad);
		}

		int wall_rad2 = wall_rad + noise_rad;
		if(wall_rad2 >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < wall_rad2 / 6; i++){
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(wall_rad2 % 6 > 0){
				setSphereMaskKernel6(wall_rad2 % 6);
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(wall_rad2);
			minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad);
		}

		subtractFloatROIGPU(d_2, d_1, d_buffer2, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		//swap(d_1, d_2);

		checkCudaErrors( cudaMemcpy(bufdata, d_2, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		int sum = 0;
		for(int x = bb.x; x < bb.x + bb.width; x++){
			for(int y = bb.y; y < bb.y + bb.height; y++){
				for(int z = bb.z; z < bb.z + bb.depth; z++){
					if(bufdata[z*imageH*imageW + y*imageW + x] > 0.0f)sum++; 
				}
			}
		}
		

		if(sum > minInvalidStructureVol){
			printf("Segment %d: sum = %d INVALID SEGMENT\n", count, sum);
			swap(d_1, d_2);
			pixelwiseOrGPU(d_2, d_1, d_Tmp, imageW, imageH, imageZ);
			swap(d_1, d_2);
			if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
			checkCudaErrors( cudaMemcpy(d_Tmp, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		}
		else{
			printf("Segment %d: sum = %d\n", count, sum);
		}
		it++;
		bbox_it++;
		count++;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("fillingHolesGPU\nTime = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n\n", gpuTime, (imageW * imageH * imageZ), 1, 0);

	checkCudaErrors( cudaMemcpy(d_Output, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	checkCudaErrors( cudaMemcpy(dstdata, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaFree(d_buffer2) );

	std::vector<float>().swap(*seg_colors);
	for(int i = 0; i < segments->size(); i++)seg_colors->push_back(rand() % 360);
}

void Filter3D::fillingHoles(float *data, int wall_th, int hole_rad, int minInvalidStructureVol){
	
	if(isEmpty)return;

	int wall_rad = wall_th >> 1;
	int rad_max = MAX(hole_rad, wall_rad);
	
	printf("<<fillingHoles: wall_rad = %d  hole_rad = %d  minInvalidStructureVol = %d  (rad_max = %d)>>\n", wall_rad, hole_rad, minInvalidStructureVol, rad_max);

	cudaEvent_t start, stop;
	Box3D bb;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
		
	float *d_buffer2;
	checkCudaErrors( cudaMalloc((void **)(&d_buffer2),   imageZ * imageW * imageH * sizeof(float)) );
	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Tmp, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );

	for(int id = 0; id < invalid_segments->size(); id++){

		fillFloatGPU(d_Output, imageW, imageH, imageZ, 0.0f);
		checkCudaErrors( cudaMemcpy(d_Buffer, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(bufdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

		for(int i = 0; i < (*invalid_segments)[id].size(); i++){
			bufdata[(*invalid_segments)[id][i].z*imageW*imageH + (*invalid_segments)[id][i].y*imageW + (*invalid_segments)[id][i].x] = BINALIZE_UPPER_VAL;
		}
		checkCudaErrors( cudaMemcpy(d_Output, bufdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_buffer2, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );//d_buffer2はsegmentの元データ

		bb.x = ((*invalid_seg_bbox)[id].x - rad_max >= 0) ? (*invalid_seg_bbox)[id].x - rad_max : 0;
		bb.y = ((*invalid_seg_bbox)[id].y - rad_max >= 0) ? (*invalid_seg_bbox)[id].y - rad_max : 0;
		bb.z = ((*invalid_seg_bbox)[id].z - rad_max >= 0) ? (*invalid_seg_bbox)[id].z - rad_max : 0;
		bb.width  = ((*invalid_seg_bbox)[id].x + (*invalid_seg_bbox)[id].width  + rad_max < imageW) ? (*invalid_seg_bbox)[id].x + (*invalid_seg_bbox)[id].width  + rad_max - bb.x : imageW - bb.x;
		bb.height = ((*invalid_seg_bbox)[id].y + (*invalid_seg_bbox)[id].height + rad_max < imageH) ? (*invalid_seg_bbox)[id].y + (*invalid_seg_bbox)[id].height + rad_max - bb.y : imageH - bb.y;
		bb.depth  = ((*invalid_seg_bbox)[id].z + (*invalid_seg_bbox)[id].depth  + rad_max < imageZ) ? (*invalid_seg_bbox)[id].z + (*invalid_seg_bbox)[id].depth  + rad_max - bb.z : imageZ - bb.z;

		
		float *d_1, *d_2;
		
		d_1 = d_Output;
		d_2 = d_Buffer;
		if(wall_rad >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < wall_rad / 6; i++){
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(wall_rad % 6 > 0){
				setSphereMaskKernel6(wall_rad % 6);
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(wall_rad);
			maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad);
		}

		if(wall_rad >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < wall_rad / 6; i++){
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(wall_rad % 6 > 0){
				setSphereMaskKernel6(wall_rad % 6);
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(wall_rad);
			minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", wall_rad);
		}

		subtractFloatROIGPU(d_2, d_1, d_buffer2, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		swap(d_1, d_2);


		//printf("Processing... [Segment %d (size = %d)]\n", id, invalid_segments->at(id).size());
		if(hole_rad >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < hole_rad / 6; i++){
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(hole_rad % 6 > 0){
				setSphereMaskKernel6(hole_rad % 6);
				maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", hole_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(hole_rad);
			maximumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("maximumSphereFilterGPU6(): Kenel_radius = %d\n", hole_rad);
		}

		pixelwiseOrGPU(d_2, d_1, d_Tmp, imageW, imageH, imageZ);
		swap(d_1, d_2);

		if(hole_rad >= 6){
			setSphereMaskKernel6(6);
			for(int i = 0; i < hole_rad / 6; i++){
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", 6);
			}
			if(hole_rad % 6 > 0){
				setSphereMaskKernel6(hole_rad % 6);
				minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
				swap(d_1, d_2);
				//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", hole_rad % 6);
			}
		}
		else {
			setSphereMaskKernel6(hole_rad);
			minimumSphereFilterGPU6(d_2, d_1, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			swap(d_1, d_2);
			//printf("minimumSphereFilterGPU6(): Kenel_radius = %d\n", hole_rad);
		}

		pixelwiseOrGPU(d_2, d_1, d_Tmp, imageW, imageH, imageZ);
		swap(d_1, d_2);

		checkCudaErrors( cudaDeviceSynchronize() );

		if(d_1 == d_Buffer)checkCudaErrors( cudaMemcpy(d_Output, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(d_Tmp, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("fillingHolesGPU\nTime = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n", gpuTime, (imageW * imageH * imageZ), 1, 0);


	checkCudaErrors( cudaMemcpy(d_Output, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	checkCudaErrors( cudaMemcpy(dstdata, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	/*
	checkCudaErrors( cudaMemcpy(bufdata, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	for(int x = 0; x < imageW; x++){
		for(int y = 0; y < imageH; y++){
			for(int z = 0; z < imageZ; z++){
				if(bufdata[z*imageH*imageW + y*imageW + x] > 0.0f && dstdata[z*imageH*imageW + y*imageW + x] <= 0.0f)dstdata[z*imageH*imageW + y*imageW + x] = 0.3f;
			}
		}
	}
	*/
	checkCudaErrors( cudaFree(d_buffer2) );
	
}

void Filter3D::fillingHolesSimple(float *data, int hole_rad)
{
	if(isEmpty)return;
	if(hole_rad <= 0)return;

	printf("<<fillingHolesSimple: hole_rad = %d>>\n", hole_rad);

	cudaEvent_t start, stop;
	Box3D bb;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	
	bb.x = 0;
	bb.y = 0;
	bb.z = 0;
	bb.width  = imageW - bb.x;
	bb.height = imageH - bb.y;
	bb.depth  = imageZ - bb.z;
	
	maximumSphereFilter(hole_rad, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
	minimumSphereFilter(hole_rad, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("fillingHolesSimple\nTime = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n", gpuTime, (imageW * imageH * imageZ), 1, 0);
	
	checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
}

int Filter3D::estimateWallThickness(int st_size, float th)
{
	if(isEmpty)return -1;
	printf("<<estimateWallThickness: st_size = %d>>\n", st_size);
	Box3D bb;

	bb.x = 0;
	bb.y = 0;
	bb.z = 0;
	bb.width  = imageW - bb.x;
	bb.height = imageH - bb.y;
	bb.depth  = imageZ - bb.z;

	float *d_1, *d_2;

	d_1 = d_Output;
	d_2 = d_Buffer;

	checkCudaErrors( cudaMemcpy(d_Tmp, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	
	int src_sum, dst_sum;
	src_sum = cublasSasum(imageW*imageH*imageZ, d_Tmp, 1);
	printf("[Src] Sum: %d\n", src_sum);

	initCuda_Suf3D(d_Output, imageW, imageH, imageZ);

	int size;
	if(st_size > 1)size = st_size;
	else size = 1;
	do{
		copyToSuf3DFromDeviceMem(d_Tmp, imageW, imageH, imageZ);
		
		if(size >= 4){
			checkCudaErrors( cudaDeviceSynchronize() );
			for(int i = 0; i < size / 4; i++){
				erodeSurfaceSphereGPU(4, imageW, imageH, imageZ);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
			if(size % 4 > 0){
				erodeSurfaceSphereGPU(size % 4, imageW, imageH, imageZ);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
		}
		else {
			erodeSurfaceSphereGPU(size, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}

		if(size >= 4){
			checkCudaErrors( cudaDeviceSynchronize() );
			for(int i = 0; i < size / 4; i++){
				dilateSurfaceSphereGPU(4, imageW, imageH, imageZ);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
			if(size % 4 > 0){
				dilateSurfaceSphereGPU(size % 4, imageW, imageH, imageZ);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
		}
		else {
			dilateSurfaceSphereGPU(size, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		
		copyToDeviceMemFromSuf3D(d_Output, imageW, imageH, imageZ); 
		checkCudaErrors( cudaDeviceSynchronize() );

		dst_sum = cublasSasum(imageW*imageH*imageZ, d_Output, 1);
		printf("Size: %d  Sum: %d\n", size, dst_sum);
		size++;

	}while(dst_sum > src_sum*th);

	destructCuda_Suf3D();

	size--;
	printf("[Size = %d]\n",size);
	checkCudaErrors( cudaMemcpy(d_Output, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
	/*
	for(int x = 0; x < imageW; x++){
		for(int y = 0; y < imageH; y++){
			for(int z = 0; z < imageZ; z++){
				if(bufdata[z*imageH*imageW + y*imageW + x] > 0.0f && dstdata[z*imageH*imageW + y*imageW + x] > 0.0f)dstdata[z*imageH*imageW + y*imageW + x] = 0.7f;
				else if(dstdata[z*imageH*imageW + y*imageW + x] > 0.0f)dstdata[z*imageH*imageW + y*imageW + x] = 0.3f;
			}
		}
	}
	*/
	return size*2;
}

//segment num == 1が終了条件
//Validation時: Opening半径上昇 (Validation対象のOpening後反転)∩(Segmentation済み領域)が空集合でない場合も終了する(この場合は最後にsegmentの数が変化したOpening半径を返す)
//Openingでsegmentの数が増えるのことについては要検証
int Filter3D::estimateWallThickness_v2(int st_size)
{
	if(isEmpty)return -1;
	printf("<<estimateWallThickness_v2: st_size = %d>>\n", st_size);
	Box3D bb;

	bb.x = 0;
	bb.y = 0;
	bb.z = 0;
	bb.width  = imageW - bb.x;
	bb.height = imageH - bb.y;
	bb.depth  = imageZ - bb.z;

	int *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
	
	float *tmpdata = new float[imageW * imageH * imageZ];

	int *tmpsegdata = new int[imageW * imageH * imageZ];
	memcpy(tmpsegdata, segdata, imageW * imageH * imageZ * sizeof(int));
	std::vector<Box3D> *tmp_seg_bbox = new std::vector<Box3D>;
	swap(tmp_seg_bbox, seg_bbox);
	std::vector< std::vector<cv::Point3i> > *tmp_segments = new std::vector< std::vector<cv::Point3i> >;
	swap(tmp_segments, segments);
	
	checkCudaErrors( cudaMemcpy(d_Tmp, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	if(tmp_segments->size() == 0)fillIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
	setValToSegmentedRegionROIGPU(d_Tmp, d_Seg, BINALIZE_UPPER_VAL, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
	checkCudaErrors( cudaMemcpy(tmpdata, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	
	binarySegmentationLow(tmpdata, 0.1f, SEGMENT_CONNECT6, 0, true);
	int *first_segdata = new int[imageW * imageH * imageZ];
	memcpy(first_segdata, segdata, imageW * imageH * imageZ * sizeof(int));

	int src_sum, dst_sum, tmp_sum;
	src_sum = segments->size();
	printf("[Src] Sum: %d\n", src_sum);

	int size, tmp_size;
	if(st_size > 1)size = st_size;
	else size = 1;
	tmp_size = size;
	tmp_sum = INT_MAX;
	do{
		checkCudaErrors( cudaMemcpy(d_Output, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		
		minimumSphereFilter(size, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		maximumSphereFilter(size, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaMemcpy(tmpdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		
		fillFloatGPU(d_Buffer, imageW, imageH, imageZ, 0.0f);
		setValToSegmentedRegionROIGPU(d_Buffer, d_Seg, BINALIZE_UPPER_VAL, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		inverseGPU(d_Output, d_Output, BINALIZE_UPPER_VAL, BINALIZE_LOWER_VAL, imageW, imageH, imageZ);
		pixelwiseAndGPU(d_Buffer, d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		
		float and_sum = cublasSasum(imageW*imageH*imageZ, d_Buffer, 1);
		if(and_sum > BINALIZE_UPPER_VAL){
			size = tmp_size;
			printf("Size: %d  AND_Sum: %f\n", size - 1, and_sum);
			break;
		}

		binarySegmentationLow(tmpdata, 0.1f, SEGMENT_CONNECT6, 0, true, first_segdata);
		dst_sum = segments->size();
		if(dst_sum < tmp_sum){
			tmp_sum = dst_sum;
			tmp_size = size + 1;
		}
		printf("Size: %d  Sum: %d  AND_Sum: %f\n", size, dst_sum, and_sum);
		size++;

	}while(dst_sum > 1);

	size--;
	if(st_size > size)size = st_size;
	printf("[Size = %d]\n",size);
	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	memcpy(segdata, tmpsegdata, imageW * imageH * imageZ * sizeof(int));

	swap(tmp_seg_bbox, seg_bbox);
	swap(tmp_segments, segments);
	
	delete [] tmpdata;
	delete [] tmpsegdata;
	delete [] first_segdata;
	delete tmp_seg_bbox;
	delete tmp_segments;

	checkCudaErrors( cudaFree(d_Seg) );
	
	return size*2;
}

int _imid = 0;
int Filter3D::estimateWallThickness_v2(int st_size, int currentX, int currentY, int currentZ, float bc_max, float bc_min)
{
	if(isEmpty)return -1;
	printf("<<estimateWallThickness_v2: st_size = %d>>\n", st_size);
	Box3D bb;

	const char *fnamebase = "estimateWallThickness_v2_Debug";
	
	bb.x = 0;
	bb.y = 0;
	bb.z = 0;
	bb.width  = imageW - bb.x;
	bb.height = imageH - bb.y;
	bb.depth  = imageZ - bb.z;

	int *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
	
	float *tmpdata = new float[imageW * imageH * imageZ];

	int *tmpsegdata = new int[imageW * imageH * imageZ];
	memcpy(tmpsegdata, segdata, imageW * imageH * imageZ * sizeof(int));
	std::vector<Box3D> *tmp_seg_bbox = new std::vector<Box3D>;
	swap(tmp_seg_bbox, seg_bbox);
	std::vector< std::vector<cv::Point3i> > *tmp_segments = new std::vector< std::vector<cv::Point3i> >;
	swap(tmp_segments, segments);
	
	checkCudaErrors( cudaMemcpy(d_Tmp, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	if(tmp_segments->size() == 0)fillIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
	setValToSegmentedRegionROIGPU(d_Tmp, d_Seg, BINALIZE_UPPER_VAL, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
	checkCudaErrors( cudaMemcpy(tmpdata, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

	saveOrthViewSimple_Numbered(tmpdata, fnamebase, _imid, currentX, currentY, currentZ, bc_max, bc_min);
	_imid++;
	
	binarySegmentationLow(tmpdata, 0.1f, SEGMENT_CONNECT6, 0, true);
	int *first_segdata = new int[imageW * imageH * imageZ];
	memcpy(first_segdata, segdata, imageW * imageH * imageZ * sizeof(int));

	int src_sum, dst_sum, tmp_sum;
	src_sum = segments->size();
	printf("[Src] Sum: %d\n", src_sum);

	int size, tmp_size;
	if(st_size > 1)size = st_size;
	else size = 1;
	//tmp_size = size;
	//tmp_sum = -1;
	do{
		checkCudaErrors( cudaMemcpy(d_Output, d_Tmp, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToDevice) );
		
		minimumSphereFilter(size, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		maximumSphereFilter(size, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaMemcpy(tmpdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

		saveOrthViewSimple_Numbered(tmpdata, fnamebase, _imid, currentX, currentY, currentZ, bc_max, bc_min);
		_imid++;
		
		fillFloatGPU(d_Buffer, imageW, imageH, imageZ, 0.0f);
		setValToSegmentedRegionROIGPU(d_Buffer, d_Seg, BINALIZE_UPPER_VAL, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		inverseGPU(d_Output, d_Output, BINALIZE_UPPER_VAL, BINALIZE_LOWER_VAL, imageW, imageH, imageZ);
		pixelwiseAndGPU(d_Buffer, d_Buffer, d_Output, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		
		float and_sum = cublasSasum(imageW*imageH*imageZ, d_Buffer, 1);
		if(and_sum > BINALIZE_UPPER_VAL){
			//size = tmp_size;
			printf("Size: %d  AND_Sum: %f\n", size - 1, and_sum);
			break;
		}

		binarySegmentationLow(tmpdata, 0.1f, SEGMENT_CONNECT6, 0, true, first_segdata);
		dst_sum = segments->size();
/*		if(dst_sum < tmp_sum){
			tmp_sum = dst_sum;
			tmp_size = size + 1;
		}
*/		printf("Size: %d  Sum: %d  AND_Sum: %f\n", size, dst_sum, and_sum);
		size++;

	}while(dst_sum > 1);

	size--;
	if(st_size > size)size = st_size;
	printf("[Size = %d]\n",size);
	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	memcpy(segdata, tmpsegdata, imageW * imageH * imageZ * sizeof(int));

	swap(tmp_seg_bbox, seg_bbox);
	swap(tmp_segments, segments);
	
	delete [] tmpdata;
	delete [] tmpsegdata;
	delete [] first_segdata;
	delete tmp_seg_bbox;
	delete tmp_segments;

	checkCudaErrors( cudaFree(d_Seg) );
	
	return size*2;
}

/*
void Filter3D::segmentation(int blocksize, int angle_d, float factor_C_Z, float minC, float maxC, float interval, int kernelType, int min_segVol, int minInvalidStructureArea, int hole_size)
{
	if(isEmpty)return;
	if(angle_d <= 0 || blocksize <= 0 || minC > maxC || interval <= 0 || hole_size < 0)return; 

	printf("<<Segmentation: blocksize = %d  angle_d = %d  factor_C_Z = %f  minC = %f  maxC = %f  interval = %f  kernelType = %d  min_segVol = %d  minInvalidStructureArea = %d>>\n",
			blocksize, angle_d, factor_C_Z, minC, maxC, interval, kernelType, min_segVol, minInvalidStructureArea);

	float curC = minC;
	int thickness = 0;
	bool isFirst = true;

	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	
	float *before_FH = new float[imageW*imageH*imageZ];
	float *new_ADTH = new float[imageW*imageH*imageZ];
	do{
		if(curC > maxC)curC = maxC;
		printf("\n[C = %f]\n", curC);
		allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, curC, curC * factor_C_Z, kernelType);
		
		if(!isFirst){
			checkCudaErrors( cudaMemcpy(new_ADTH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaMemcpy(d_Buffer, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
			pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
			checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		}

		thickness = estimateWallThickness(thickness/2);
		binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, isFirst, thickness/3, thickness, minInvalidStructureArea*thickness, false, true);//NOTSAVED領域はFillingHoles後にとりなおす
		
		if(!isFirst){
			checkCudaErrors( cudaMemcpy(d_Output, new_ADTH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		}
		fillingHoles(dstdata, thickness, hole_size/2, minInvalidStructureArea*thickness);
		if(!isFirst){
			checkCudaErrors( cudaMemcpy(d_Buffer, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
			pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
			checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		}

		isFirst = false;
		checkCudaErrors( cudaMemcpy(before_FH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, isFirst, thickness/3, thickness, minInvalidStructureArea*thickness, true, false);
		

		curC += interval;
		
	}while(curC < maxC + interval);

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("\nsegmentation: Time = %.5f s, Size = %u Pixels\n\n", gpuTime, (imageW * imageH * imageZ));

	//checkCudaErrors( cudaMemcpy(d_Output, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	delete [] before_FH;
	delete [] new_ADTH;
}
*/

void Filter3D::segmentation(int blocksize, int angle_d, float factor_C_Z, float minC, float maxC, float interval, int kernelType, int min_segVol, int minInvalidStructureArea, int hole_size)
{
	if(isEmpty)return;
	if(angle_d <= 0 || blocksize <= 0 || minC > maxC || interval <= 0 || hole_size < 0)return; 

	printf("<<Segmentation: blocksize = %d  angle_d = %d  factor_C_Z = %f  minC = %f  maxC = %f  interval = %f  kernelType = %d  min_segVol = %d  minInvalidStructureArea = %d>>\n",
			blocksize, angle_d, factor_C_Z, minC, maxC, interval, kernelType, min_segVol, minInvalidStructureArea);

	float curC = minC;
	int thickness = 0;
	
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	
	float *before_FH = new float[imageW*imageH*imageZ];
	

	printf("\n[C = %f]\n", curC);
	allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, curC, curC * factor_C_Z, kernelType);
	
	//最初はすべての部分でFillingHolesを適用
	binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, true);
	thickness = estimateWallThickness(0, 0.1f);
	swap(segments, invalid_segments);
	swap(seg_bbox, invalid_seg_bbox);
	fillingHoles(dstdata, thickness, hole_size/2, minInvalidStructureArea*thickness);
	checkCudaErrors( cudaMemcpy(before_FH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, true, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);
	curC = minC + interval;

	int loopcount = 1;
	while(curC < maxC + interval){
		if(curC > maxC)curC = maxC;
		printf("\n[C = %f]\n", curC);
		allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, curC, curC * factor_C_Z, kernelType);
		
		printf("<<Invalid Segments Update>>\n");
		for(int id = 0; id < invalid_segments->size(); id++){
			vector<Point3i>::iterator iseg_it = (*invalid_segments)[id].begin();
			int count = 0;
			while(iseg_it != invalid_segments->at(id).end()){
				if(dstdata[iseg_it->z*imageW*imageH + iseg_it->y*imageW + iseg_it->x] > 0.0f){
					iseg_it = (*invalid_segments)[id].erase(iseg_it);
					count++;
				}
				else iseg_it++;
			}
			printf("Invalid Segment %d: %d points deleted\n", id, count);
		}
		
		checkCudaErrors( cudaMemcpy(d_Buffer, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		thickness = estimateWallThickness(thickness/2, 0.1f);//測定用データは before_FH || new_ADTH
		
		memcpy(dstdata, before_FH, imageW * imageH * imageZ * sizeof(float));
		fillingHoles(dstdata, thickness, hole_size/2, minInvalidStructureArea*thickness);
				
		checkCudaErrors( cudaMemcpy(before_FH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, false, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);
		
		loopcount++;
		curC = minC + interval * loopcount;
		
	}

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("\nsegmentation: Time = %.5f s, Size = %u Pixels\n\n", gpuTime, (imageW * imageH * imageZ));

	//checkCudaErrors( cudaMemcpy(d_Output, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	delete [] before_FH;
	//delete [] new_ADTH;
}

void Filter3D::segmentation_SobelLikeADTH(int blocksize, int angle_d, float factor_C_Z, float minC, float maxC, float interval, int kernelType, int minInvalidStructureArea, int closing)
{
	if(isEmpty)return;
	if(angle_d <= 0 || blocksize <= 0 || minC > maxC || interval <= 0 || closing < 0)return; 

	printf("<<Segmentation_SobelLikeADTH: blocksize = %d  angle_d = %d  factor_C_Z = %f  minC = %f  maxC = %f  interval = %f  kernelType = %d  minInvalidStructureArea = %d>>\n",
			blocksize, angle_d, factor_C_Z, minC, maxC, interval, kernelType, minInvalidStructureArea);

	float curC = minC;
	int thickness = 0;
	
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	
	float *dsadth = new float[imageW*imageH*imageZ];
	float *argmin_lati = new float[imageW*imageH*imageZ];
	int *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	
	checkCudaErrors( cudaMemcpy(d_Input, imgdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );

	printf("\n[C = %f]\n", curC);
	
	//allAngleLinearConvolutionMinGPU(dsadth, imgdata, blocksize, angle_d, kernelType, argmin_lati, NULL);
	allAngleLinearConvolutionMinGPU_geodesic(dsadth, imgdata, blocksize, angle_d, kernelType, argmin_lati, NULL);

	checkCudaErrors( cudaMemcpy(d_Buffer, argmin_lati, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Output, dsadth, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	calcCorrectionTerms_DSADTH_GPU(d_Buffer, d_Buffer, curC, curC * factor_C_Z, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	addFloatGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
	checkCudaErrors( cudaDeviceSynchronize() );
	binarizeGPU(d_Output, d_Output, d_Input, imageW, imageH, imageZ, 0.0f);
	checkCudaErrors( cudaDeviceSynchronize() );
	
	maximumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	minimumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);

	checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	
	if(seg_bbox != NULL)std::vector<Box3D>().swap(*seg_bbox);
	clear2DVector(segments);
	fillIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
	checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );

	if(crop_isEnable){
		for(int z = 0; z < imageZ; z++){
			for(int y = 0; y < imageH; y++){
				for(int x = 0; x < imageW; x++){
					if(y < crop_border_xy_thickness || y >= imageH - crop_border_xy_thickness || x < crop_border_xy_thickness || x >= imageW - crop_border_xy_thickness)
						dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
					else if(crop_useHmap && !hmapEmpty){
						if(z < hmap[y*imageW + x] + crop_upper || z > hmap[y*imageW + x] + crop_lower)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
					}
					else{
						if(z < crop_upper || z > crop_lower)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
					}
				}
			}
		}
	}
	
	thickness = estimateWallThickness(0, 0.5f);
	//thickness = estimateWallThickness_v2(0);
	//if(thickness < closing*2)thickness = closing*2;

	if(crop_isEnable){
		for(int z = 0; z < imageZ; z++){
			for(int y = 0; y < imageH; y++){
				for(int x = 0; x < imageW; x++){
					if(y < crop_border_xy_thickness || y >= imageH - crop_border_xy_thickness || x < crop_border_xy_thickness || x >= imageW - crop_border_xy_thickness)
						dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
					else if(crop_useHmap && !hmapEmpty){
						if(z < hmap[y*imageW + x] + crop_upper || z > hmap[y*imageW + x] + crop_lower)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
					}
					else{
						if(z < crop_upper || z > crop_lower)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
					}
				}
			}
		}
	}

	binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, 0, true, 0, thickness, minInvalidStructureArea*thickness, true, true);
	curC = minC + interval;

	int loopcount = 1;
	while(curC < maxC + interval){
		if(curC > maxC)curC = maxC;
		printf("\n[C = %f]\n", curC);
		checkCudaErrors( cudaMemcpy(d_Buffer, argmin_lati, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_Output, dsadth, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
		calcCorrectionTerms_DSADTH_GPU(d_Buffer, d_Buffer, curC, curC * factor_C_Z, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		addFloatGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaDeviceSynchronize() );
		binarizeGPU(d_Output, d_Output, d_Input, imageW, imageH, imageZ, 0.0f);
		checkCudaErrors( cudaDeviceSynchronize() );
		
		maximumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		minimumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);

		checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
		setValToSegmentedRegionROIGPU(d_Output, d_Seg, 0.0f, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

		if(crop_isEnable){
			for(int z = 0; z < imageZ; z++){
				for(int y = 0; y < imageH; y++){
					for(int x = 0; x < imageW; x++){
						if(y < crop_border_xy_thickness || y >= imageH - crop_border_xy_thickness || x < crop_border_xy_thickness || x >= imageW - crop_border_xy_thickness)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
						else if(crop_useHmap && !hmapEmpty){
							if(z < hmap[y*imageW + x] + crop_upper || z > hmap[y*imageW + x] + crop_lower)
								dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
						}
						else{
							if(z < crop_upper || z > crop_lower)
								dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
						}
					}
				}
			}
		}

		thickness = estimateWallThickness(thickness/2, 0.5f);
		//thickness = estimateWallThickness_v2(0);//(thickness/2);
		//if(thickness < closing*2)thickness = closing*2;

		if(crop_isEnable){
			for(int z = 0; z < imageZ; z++){
				for(int y = 0; y < imageH; y++){
					for(int x = 0; x < imageW; x++){
						if(y < crop_border_xy_thickness || y >= imageH - crop_border_xy_thickness || x < crop_border_xy_thickness || x >= imageW - crop_border_xy_thickness)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
						else if(crop_useHmap && !hmapEmpty){
							if(z < hmap[y*imageW + x] + crop_upper || z > hmap[y*imageW + x] + crop_lower)
								dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
						}
						else{
							if(z < crop_upper || z > crop_lower)
								dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
						}
					}
				}
			}
		}

		if(minC + interval * (loopcount + 1) >= maxC + interval)binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, 0, false, 0, thickness, INT_MAX, true, true);
		else binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, 0, false, 0, thickness, minInvalidStructureArea*thickness, true, true);
		if(invalid_segments->size() == 0)break;
		loopcount++;
		curC = minC + interval * loopcount;
		
	}
	/*
	if(closing > 0){
		checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
		//d_Segを直接Dilationでもよいかも
		for(int i = 0; i < segments->size(); i++){
			Box3D bb;
			bb.x = ((*seg_bbox)[i].x - closing >= 0) ? (*seg_bbox)[i].x - closing : 0;
			bb.y = ((*seg_bbox)[i].y - closing >= 0) ? (*seg_bbox)[i].y - closing : 0;
			bb.z = ((*seg_bbox)[i].z - closing >= 0) ? (*seg_bbox)[i].z - closing : 0;
			bb.width  = ((*seg_bbox)[i].x + (*seg_bbox)[i].width  + closing < imageW) ? (*seg_bbox)[i].x + (*seg_bbox)[i].width  + closing - bb.x : imageW - bb.x;
			bb.height = ((*seg_bbox)[i].y + (*seg_bbox)[i].height + closing < imageH) ? (*seg_bbox)[i].y + (*seg_bbox)[i].height + closing - bb.y : imageH - bb.y;
			bb.depth  = ((*seg_bbox)[i].z + (*seg_bbox)[i].depth  + closing < imageZ) ? (*seg_bbox)[i].z + (*seg_bbox)[i].depth  + closing - bb.z : imageZ - bb.z;
			
			checkCudaErrors( cudaDeviceSynchronize() );
			fillFloatGPU(d_Output, imageW, imageH, imageZ, 0.0f);
			checkCudaErrors( cudaDeviceSynchronize() );
			copySingleSegmentROIGPU(d_Output, d_Seg, i, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			checkCudaErrors( cudaDeviceSynchronize() );
			maximumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			minimumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
			checkCudaErrors( cudaDeviceSynchronize() );
			overwriteSingleSegmentROIGPU(d_Seg, i, d_Output, BINALIZE_UPPER_VAL, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		}
		checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );

		for(int i = 0; i < segments->size(); i++){
			(*segments)[i].clear();
		}

		for(int z = 0; z < imageZ; z++){
			for(int y = 0; y < imageH; y++){
				for(int x = 0; x < imageW; x++){
					if(segdata[z*imageH*imageW + y*imageW + x] >= 0)(*segments)[segdata[z*imageH*imageW + y*imageW + x]].push_back(Point3i(x, y, z));
				}
			}
		}
	}
	*/
	checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
	setValToSegmentedRegionROIGPU(d_Output, d_Seg, BINALIZE_LOWER_VAL, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	
	memcpy(dmap, dstdata, imageW*imageH*imageZ*sizeof(float));

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("\nSegmentation_SobelLikeADTH: Time = %.5f s, Size = %u Pixels\n\n", gpuTime, (imageW * imageH * imageZ));

	//checkCudaErrors( cudaMemcpy(d_Output, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	delete [] dsadth;
	delete [] argmin_lati;
	checkCudaErrors( cudaFree(d_Seg) );
}

void Filter3D::saveRGBimageSeries(const char *fname, float bc_max, float bc_min)
{
	/*
	for(int i = 0; i < imageZ; i++){
		stringstream ss;
		ss << fname << i << ".tif";
		simpleProjection(imgdata, bc_max, bc_min, i, 0, true);
		imwrite(ss.str().c_str(), Mat(imageH, imageW, CV_8UC3, bufXY));
	}
	*/
	unsigned char *rgbimagestack = new unsigned char [imageW*imageH*imageZ*3];
	for(int i = 0; i < imageZ; i++){
		simpleProjection(imgdata, bc_max, bc_min, i, 0, true);
		memcpy(rgbimagestack + imageW*imageH*3*i, bufXY, imageW*imageH*3);
	}
	stringstream ss;
	ss << fname << "_rgb" << ".tif";
	multif::MultiTiffIO::SaveImageData(ss.str().c_str(), (char *)rgbimagestack, imageW, imageH, imageZ, sizeof(unsigned char)*8, 3);

	delete [] rgbimagestack;
}

void Filter3D::saveOrthViewSimple(float *data, const char *fname, int currentX, int currentY, int currentZ, float bc_max, float bc_min)
{
	string fnameXY = fname;
	fnameXY += ".tif";
	cout << fnameXY << endl;

	string fnameYZ = fname;
	fnameYZ += "YZ.tif";
	
	string fnameZX = fname;
	fnameZX += "ZX.tif";

	simpleProjection(data, bc_max, bc_min, currentZ, 0);
	setBufferYZ(data, currentX, bc_max, bc_min);
	setBufferZX(data, currentY, bc_max, bc_min);
	
	//imwrite(fnameXY, Mat(imageH, imageW, CV_8UC3, bufXY));
	//imwrite(fnameYZ, Mat(imageH, imageZ, CV_8UC3, bufYZ));
	imwrite(fnameZX, Mat(imageZ, imageW, CV_8UC3, bufZX));
	
}

void Filter3D::saveOrthViewSimple_Numbered(float *data, const char *fname, int id, int currentX, int currentY, int currentZ, float bc_max, float bc_min)
{
	stringstream ss;
	ss << fname << id;
	saveOrthViewSimple(data, ss.str().c_str(), currentX, currentY, currentZ, bc_max, bc_min);
}

void Filter3D::segmentation_SobelLikeADTH(int blocksize,
										  int angle_d,
										  float factor_C_Z,
										  float minC,
										  float maxC,
										  float interval,
										  int kernelType,
										  int minInvalidStructureArea,
										  int closing,
										  int currentX,
										  int currentY,
										  int currentZ,
										  float bc_max,
										  float bc_min)
{
/*
	if(isEmpty)return;
	if(angle_d <= 0 || blocksize <= 0 || minC > maxC || interval <= 0 || closing < 0)return; 

	printf("<<Segmentation_SobelLikeADTH: blocksize = %d  angle_d = %d  factor_C_Z = %f  minC = %f  maxC = %f  interval = %f  kernelType = %d  min_segVol = %d  minInvalidStructureArea = %d>>\n",
			blocksize, angle_d, factor_C_Z, minC, maxC, interval, kernelType, min_segVol, minInvalidStructureArea);

	float curC = minC;
	int thickness = 0;

	const char *fnamebase = "segmentationDebug";
	
	int imid = 0;
	
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	
	float *before_FH = new float[imageW*imageH*imageZ];
	

	printf("\n[C = %f]\n", curC);
	allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, curC, curC * factor_C_Z, kernelType);
	
	saveOrthViewSimple_Numbered(dstdata, fnamebase, imid, currentX, currentY, currentZ, bc_max, bc_min);
	imid++;

	thickness = estimateWallThickness(0, 0.1f);
	fillingHolesSimple(dstdata, closing);

	saveOrthViewSimple_Numbered(dstdata, fnamebase, imid, currentX, currentY, currentZ, bc_max, bc_min);
	imid++;

	checkCudaErrors( cudaMemcpy(before_FH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, true, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);

	saveOrthViewSimple_Numbered(dstdata, fnamebase, imid, currentX, currentY, currentZ, bc_max, bc_min);
	imid++;

	curC = minC + interval;

	int loopcount = 1;
	while(curC < maxC + interval){
		if(curC > maxC)curC = maxC;
		printf("\n[C = %f]\n", curC);
		allAngleLinearAdaptiveThresholdGPU(blocksize, angle_d, curC, curC * factor_C_Z, kernelType);

		saveOrthViewSimple_Numbered(dstdata, fnamebase, imid, currentX, currentY, currentZ, bc_max, bc_min);
		imid++;
		
		checkCudaErrors( cudaMemcpy(d_Buffer, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		for(int id = 0; id < segments->size(); id++){
			for(int i = 0; i < (*segments)[id].size(); i++){
				dstdata[(*segments)[id][i].z*imageW*imageH + (*segments)[id][i].y*imageW + (*segments)[id][i].x] = 0.0f;
			}
		}
		thickness = estimateWallThickness(thickness/2, 0.1f);
		
		fillingHolesSimple(dstdata, closing);
		checkCudaErrors( cudaMemcpy(before_FH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

		saveOrthViewSimple_Numbered(dstdata, fnamebase, imid, currentX, currentY, currentZ, bc_max, bc_min);
		imid++;

		binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, false, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);

		saveOrthViewSimple_Numbered(dstdata, fnamebase, imid, currentX, currentY, currentZ, bc_max, bc_min);
		imid++;
		
		loopcount++;
		curC = minC + interval * loopcount;
		
	}

	for(int x = 0; x < imageW; x++){
		for(int y = 0; y < imageH; y++){
			for(int z = 0; z < imageZ; z++){
				if(segdata[z*imageH*imageW + y*imageW + x] >= 0)dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
			}
		}
	}

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("\nSegmentation_SobelLikeADTH: Time = %.5f s, Size = %u Pixels\n\n", gpuTime, (imageW * imageH * imageZ));

	//checkCudaErrors( cudaMemcpy(d_Output, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	delete [] before_FH;
	//delete [] new_ADTH;
	*/
	if(isEmpty)return;
	if(angle_d <= 0 || blocksize <= 0 || minC > maxC || interval <= 0 || closing < 0)return; 

	printf("<<Segmentation_SobelLikeADTH: blocksize = %d  angle_d = %d  factor_C_Z = %f  minC = %f  maxC = %f  interval = %f  kernelType = %d  minInvalidStructureArea = %d>>\n",
			blocksize, angle_d, factor_C_Z, minC, maxC, interval, kernelType, minInvalidStructureArea);

	float curC = minC;
	int thickness = 0;
	
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	
	float *dsadth = new float[imageW*imageH*imageZ];
	float *argmin_lati = new float[imageW*imageH*imageZ];
	int *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	
	checkCudaErrors( cudaMemcpy(d_Input, imgdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );

	printf("\n[C = %f]\n", curC);
	
	allAngleLinearConvolutionMinGPU(dsadth, imgdata, blocksize, angle_d, kernelType, argmin_lati, NULL);

	checkCudaErrors( cudaMemcpy(d_Buffer, argmin_lati, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Output, dsadth, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	calcCorrectionTerms_DSADTH_GPU(d_Buffer, d_Buffer, curC, curC * factor_C_Z, imageW, imageH, imageZ);
	addFloatGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
	binarizeGPU(d_Output, d_Output, d_Input, imageW, imageH, imageZ, 0.0f);
	
	maximumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
	minimumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);

	checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	if(seg_bbox != NULL)std::vector<Box3D>().swap(*seg_bbox);
	clear2DVector(segments);
	fillIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
	checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	thickness = estimateWallThickness_v2(0, currentX, currentY, currentZ, bc_max, bc_min);
	
	binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, 0, true, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);
	curC = minC + interval;

	int loopcount = 1;
	while(curC < maxC + interval){
		if(curC > maxC)curC = maxC;
		printf("\n[C = %f]\n", curC);
		checkCudaErrors( cudaMemcpy(d_Buffer, argmin_lati, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_Output, dsadth, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
		calcCorrectionTerms_DSADTH_GPU(d_Buffer, d_Buffer, curC, curC * factor_C_Z, imageW, imageH, imageZ);
		addFloatGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		binarizeGPU(d_Output, d_Output, d_Input, imageW, imageH, imageZ, 0.0f);
		
		maximumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		minimumSphereFilter(closing, d_Output, d_Buffer, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);

		checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
		setValToSegmentedRegionROIGPU(d_Output, d_Seg, 0.0f, imageW, imageH, imageZ, 0, 0, 0, imageW, imageH, imageZ);
		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		
		thickness = estimateWallThickness_v2(0, currentX, currentY, currentZ, bc_max, bc_min);//(thickness/2);
		
		binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, 0, false, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);
		if(invalid_segments->size() == 0)break;
		loopcount++;
		curC = minC + interval * loopcount;
		
	}

	for(int x = 0; x < imageW; x++){
		for(int y = 0; y < imageH; y++){
			for(int z = 0; z < imageZ; z++){
				if(segdata[z*imageH*imageW + y*imageW + x] >= 0)dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
			}
		}
	}

	memcpy(dmap, dstdata, imageW*imageH*imageZ*sizeof(float));

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("\nSegmentation_SobelLikeADTH: Time = %.5f s, Size = %u Pixels\n\n", gpuTime, (imageW * imageH * imageZ));

	//checkCudaErrors( cudaMemcpy(d_Output, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	delete [] dsadth;
	delete [] argmin_lati;
	checkCudaErrors( cudaFree(d_Seg) );
}


void Filter3D::segmentation_hMinimaTransform(float minh, float maxh, float interval, int min_segVol, int minInvalidStructureArea, int closing)
{
	if(isEmpty)return;
	if(minh > maxh || interval <= 0 || closing < 0)return; 

	printf("<<Segmentation_hMinimaTransform: minh = %f  maxh = %f  interval = %f  min_segVol = %d  minInvalidStructureArea = %d>>\n",
			minh, maxh, interval, min_segVol, minInvalidStructureArea);

	float curh = maxh;
	int thickness = 0;
	
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	
	float *before_FH = new float[imageW*imageH*imageZ];
	

	printf("\n[h = %f]\n", curh);
	hMinimaTransform3D_GPU(maxh, 10, true);
	thickness = estimateWallThickness(0, 0.3f);
	fillingHolesSimple(dstdata, closing);
	checkCudaErrors( cudaMemcpy(before_FH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, true, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);
	curh = maxh - interval;

	int loopcount = 1;
	while(curh > minh - interval){
		if(curh < minh)curh = minh;
		printf("\n[h = %f]\n", curh);
		hMinimaTransform3D_GPU(curh, 10, true);
		
		checkCudaErrors( cudaMemcpy(d_Buffer, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
		pixelwiseOrGPU(d_Output, d_Output, d_Buffer, imageW, imageH, imageZ);
		checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
		for(int id = 0; id < segments->size(); id++){
			for(int i = 0; i < (*segments)[id].size(); i++){
				dstdata[(*segments)[id][i].z*imageW*imageH + (*segments)[id][i].y*imageW + (*segments)[id][i].x] = 0.0f;
			}
		}
		thickness = estimateWallThickness(thickness/2, 0.3f);
		
		fillingHolesSimple(dstdata, closing);
		checkCudaErrors( cudaMemcpy(before_FH, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

		binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, min_segVol, false, thickness/3, thickness, minInvalidStructureArea*thickness, true, true);
		
		loopcount++;
		curh = maxh - interval * loopcount;
		
	}

	for(int x = 0; x < imageW; x++){
		for(int y = 0; y < imageH; y++){
			for(int z = 0; z < imageZ; z++){
				if(segdata[z*imageH*imageW + y*imageW + x] >= 0)dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
			}
		}
	}

	checkCudaErrors( cudaDeviceSynchronize() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double gpuTime = 0.001 * elapsedTime;
	printf("\nSegmentation_hMinimaTransform: Time = %.5f s, Size = %u Pixels\n\n", gpuTime, (imageW * imageH * imageZ));

	//checkCudaErrors( cudaMemcpy(d_Output, before_FH, imageW * imageH * imageZ * sizeof(float), cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dstdata, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	delete [] before_FH;
	//delete [] new_ADTH;
}

void Filter3D::AdaptiveThreshold3D_CPU(int blocksize, float constC, int thresholdType)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	convolutionRowCPU(bufdata, imgdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	convolutionColumnCPU(dstdata, bufdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	convolutionZColumnCPU(bufdata, dstdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	binarizeCPU(dstdata, bufdata, imgdata, imageW*imageH*imageZ, constC);

	memcpy(h_tmp, dstdata, imageW*imageH*imageZ*sizeof(float));
	
	delete [] h_Kernel;
}

void Filter3D::threshold3D_CPU(int blocksize, float constC, int thresholdType)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	convolutionRowCPU(bufdata, h_tmp, h_Kernel, imageW, imageH, imageZ, blocksize);
	convolutionColumnCPU(dstdata, bufdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	convolutionZColumnCPU(bufdata, dstdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	thresholdCPU(dstdata, bufdata, imageW*imageH*imageZ, constC);
	
	delete [] h_Kernel;
}

void Filter3D::AdaptiveThreshold2D_CPU(int blocksize, float constC, int thresholdType)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	convolutionRowCPU(dstdata, imgdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	convolutionColumnCPU(bufdata, dstdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	binarizeCPU(dstdata, bufdata, imgdata, imageW*imageH*imageZ, constC);

	memcpy(h_tmp, dstdata, imageW*imageH*imageZ*sizeof(float));
	
	delete [] h_Kernel;
}

void Filter3D::threshold2D_CPU(int blocksize, float constC, int thresholdType)
{
	int kernel_length = blocksize * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel = new float[kernel_length];

	if(thresholdType == FILTER_GAUSSIAN)getGaussianFilter1D(h_Kernel, kernel_length);
	if(thresholdType == FILTER_MEAN)getMeanFilter1D(h_Kernel, kernel_length);

	convolutionRowCPU(dstdata, h_tmp, h_Kernel, imageW, imageH, imageZ, blocksize);
	convolutionColumnCPU(bufdata, dstdata, h_Kernel, imageW, imageH, imageZ, blocksize);
	thresholdCPU(dstdata, bufdata, imageW*imageH*imageZ, constC);
	
	delete [] h_Kernel;
}

void Filter3D::getGaussianFilter1D(float filter[], int ksize)
{
	double sigma = 0.3*(ksize/2 - 1) + 0.8;
	double denominator = 2.0*sigma*sigma;
	double sum;
	double xx, d;
	int x;

	sum = 0.0;
	for(x = 0; x < ksize; x++){
		xx = x - (ksize - 1)/2;
		d = xx*xx;
		filter[x] = (float)exp(-1.0*d/denominator);
		sum += filter[x];
	}

	for(x = 0; x < ksize; x++)filter[x] /= (float)sum;

}

void Filter3D::getMeanFilter1D(float filter[], int ksize)
{
	int x;

	for(x = 0; x < ksize; x++){
		filter[x] = 1.0f/(float)ksize;
	}
}

void Filter3D::generateHeightMapGPU(int blocksize_xy, int blocksize_z, float th, int thresholdType, int smoothLv)
{
	int kernel_length_xy = blocksize_xy * 2 + 1;
	int kernel_length_z = blocksize_z * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel_xy = new float[kernel_length_xy];
	float *h_Kernel_z = new float[kernel_length_z];

	if(thresholdType == FILTER_GAUSSIAN){
		getGaussianFilter1D(h_Kernel_xy, kernel_length_xy);
		getGaussianFilter1D(h_Kernel_z, kernel_length_z);
	}
	if(thresholdType == FILTER_MEAN){
		getMeanFilter1D(h_Kernel_xy, kernel_length_xy);
		getMeanFilter1D(h_Kernel_z, kernel_length_z);
	}

	checkCudaErrors( cudaMemcpy(d_Output, dstdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	
	if(blocksize_z <= 8){
		setConvolutionKernel8(h_Kernel_z, blocksize_z);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU8(d_Buffer, d_Output, imageW, imageH, imageZ);
		
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize_z <= 16){
		setConvolutionKernel16(h_Kernel_z, blocksize_z);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU16(d_Buffer, d_Output, imageW, imageH, imageZ);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize_z <= 32){
		setConvolutionKernel32(h_Kernel_z, blocksize_z);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU32(d_Buffer, d_Output, imageW, imageH, imageZ);

		checkCudaErrors( cudaDeviceSynchronize() );
	}
	else if(blocksize_z <= 64){
		setConvolutionKernel64(h_Kernel_z, blocksize_z);
		checkCudaErrors( cudaDeviceSynchronize() );

		convolutionZColumnsGPU64(d_Buffer, d_Output, imageW, imageH, imageZ);

		checkCudaErrors( cudaDeviceSynchronize() );
	}

	checkCudaErrors( cudaMemcpy(bufdata, d_Buffer, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );

	unsigned int x, y, z;

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			hmap[y*imageW + x] = 0.0f;
			for(z = 0; z < imageZ; z++){
				if(bufdata[z*imageH*imageW + y*imageW + x] > th){
					if(z == 0){
						hmap[y*imageW + x] = 0.0f;
						break;
					}
					else{
						hmap[y*imageW + x] = (float)(z - 1) + (th - bufdata[(z - 1)*imageH*imageW + y*imageW + x]) / (bufdata[z*imageH*imageW + y*imageW + x] - bufdata[(z - 1)*imageH*imageW + y*imageW + x]);
						break;
					}
				}
			}
			if(z == imageZ){
				hmap[y*imageW + x] = 0.0f;//(float)(z - 1);
			}
		}
	}

	checkCudaErrors( cudaMemcpy(d_Output, hmap, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );

	for(int i = 0; i < smoothLv; i++){
		if(blocksize_xy <= 8){
			setConvolutionKernel8(h_Kernel_xy, blocksize_xy);
			checkCudaErrors( cudaDeviceSynchronize() );

			convolutionRowsGPU8(d_Buffer, d_Output, imageW, imageH, 1);
			checkCudaErrors( cudaDeviceSynchronize() );
			convolutionColumnsGPU8(d_Output, d_Buffer, imageW, imageH, 1);

			checkCudaErrors( cudaDeviceSynchronize() );
		}
		else if(blocksize_xy <= 16){
			setConvolutionKernel16(h_Kernel_xy, blocksize_xy);
			checkCudaErrors( cudaDeviceSynchronize() );

			convolutionRowsGPU16(d_Buffer, d_Output, imageW, imageH, 1);
			checkCudaErrors( cudaDeviceSynchronize() );
			convolutionColumnsGPU16(d_Output, d_Buffer, imageW, imageH, 1);

			checkCudaErrors( cudaDeviceSynchronize() );
		}
		else if(blocksize_xy <= 32){
			setConvolutionKernel32(h_Kernel_xy, blocksize_xy);
			checkCudaErrors( cudaDeviceSynchronize() );

			convolutionRowsGPU32(d_Buffer, d_Output, imageW, imageH, 1);
			checkCudaErrors( cudaDeviceSynchronize() );
			convolutionColumnsGPU32(d_Output, d_Buffer, imageW, imageH, 1);

			checkCudaErrors( cudaDeviceSynchronize() );
		}
		else if(blocksize_xy <= 64){
			setConvolutionKernel64(h_Kernel_xy, blocksize_xy);
			checkCudaErrors( cudaDeviceSynchronize() );

			convolutionRowsGPU64(d_Buffer, d_Output, imageW, imageH, 1);
			checkCudaErrors( cudaDeviceSynchronize() );
			convolutionColumnsGPU64(d_Output, d_Buffer, imageW, imageH, 1);

			checkCudaErrors( cudaDeviceSynchronize() );
		}
	}

	checkCudaErrors( cudaMemcpy(hmap, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
	
	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			hmap_normalized[y*imageW + x] = hmap[y*imageW + x] / (float)(imageZ - 1);
			normals[y*imageW + x] = Point3f(0, 0, 0);
		}
	}

	/* culculate cross */
	Point3f crvec;
	float crnorm;
	for(y = 0; y < imageH - 1; y++){
		for(x = 0; x < imageW - 1; x++){
			normals[y*imageW + x] += Point3f(-(hmap[y*imageW + (x+1)] - hmap[y*imageW + x]), -(hmap[(y+1)*imageW + x] - hmap[y*imageW + x]), 1);
		}
	}

	for(y = 0; y < imageH - 1; y++){
		for(x = 1; x < imageW; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x-1)] - hmap[y*imageW + x], -(hmap[(y+1)*imageW + x] - hmap[y*imageW + x]), 1);
		}
	}

	for(y = 1; y < imageH; y++){
		for(x = 1; x < imageW; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x-1)] - hmap[y*imageW + x], hmap[(y-1)*imageW + x] - hmap[y*imageW + x], 1);
		}
	}

	for(y = 1; y < imageH; y++){
		for(x = 0; x < imageW - 1; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x+1)] - hmap[y*imageW + x], hmap[(y-1)*imageW + x] - hmap[y*imageW + x], 1);
		}
	}

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			crnorm = sqrt(normals[y*imageW + x].x * normals[y*imageW + x].x + normals[y*imageW + x].y * normals[y*imageW + x].y + normals[y*imageW + x].z * normals[y*imageW + x].z);
			normals[y*imageW + x] = Point3f(normals[y*imageW + x].x / crnorm, normals[y*imageW + x].y / crnorm, normals[y*imageW + x].z / crnorm);
		}
	}
	/*
	Mat hmp_mat, hmp_col;
	Mat(imageH, imageW, CV_32FC1, hmap_normalized).clone().convertTo(hmp_mat, CV_8U, 255);
	cvtColor(hmp_mat, hmp_col, CV_GRAY2RGB);
	imwrite("hmap.tif", hmp_col);
		
	for(y = 0; y < imageH; y += 10){
		for(x = 0; x < imageW; x += 10){
			line(hmp_col, Point2i(x, y), Point2f((float)x + normals[y*imageW + x].x*20.0f, (float)y + normals[y*imageW + x].y*20.0f), Scalar(0, 255, 255));
		}
	}
	imshow("hmp", hmp_col);
	imwrite("hmap_nor.tif", hmp_col);
	*/
	delete[] h_Kernel_xy;
	delete[] h_Kernel_z;

	hmapEmpty = false;
}

void Filter3D::generateHeightMapCPU(int blocksize_xy, int blocksize_z, float th, int thresholdType)
{
	int kernel_length_xy = blocksize_xy * 2 + 1;
	int kernel_length_z = blocksize_z * 2 + 1;

	if(isEmpty)return;

	float *h_Kernel_xy = new float[kernel_length_xy];
	float *h_Kernel_z = new float[kernel_length_z];

	if(thresholdType == FILTER_GAUSSIAN){
		getGaussianFilter1D(h_Kernel_xy, kernel_length_xy);
		getGaussianFilter1D(h_Kernel_z, kernel_length_z);
	}
	if(thresholdType == FILTER_MEAN){
		getMeanFilter1D(h_Kernel_xy, kernel_length_xy);
		getMeanFilter1D(h_Kernel_z, kernel_length_z);
	}

	convolutionRowCPU(bufdata, dstdata, h_Kernel_xy, imageW, imageH, imageZ, blocksize_xy);
	convolutionColumnCPU(h_tmp, bufdata, h_Kernel_xy, imageW, imageH, imageZ, blocksize_xy);
	convolutionZColumnCPU(bufdata, h_tmp, h_Kernel_z, imageW, imageH, imageZ, blocksize_z);

	unsigned int x, y, z;

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			hmap[y*imageW + x] = 0.0f;
			hmap_normalized[y*imageW + x] = 0.0f;
			for(z = 0; z < imageZ; z++){
				if(bufdata[z*imageH*imageW + y*imageW + x] > th){
					if(z == 0){
						hmap[y*imageW + x] = 0.0f;
						hmap_normalized[y*imageW + x] = 0.0f;
						break;
					}
					else{
						hmap[y*imageW + x] = (float)(z - 1) + (th - bufdata[(z - 1)*imageH*imageW + y*imageW + x]) / (bufdata[z*imageH*imageW + y*imageW + x] - bufdata[(z - 1)*imageH*imageW + y*imageW + x]);
						hmap_normalized[y*imageW + x] = hmap[y*imageW + x] / (float)(imageZ - 1);
						break;
					}
				}
			}
			if(z == imageZ){
				hmap[y*imageW + x] = (float)(z - 1);
				hmap_normalized[y*imageW + x] = hmap[y*imageW + x] / (float)(imageZ - 1);
			}
		}
	}

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			normals[y*imageW + x] = Point3f(0, 0, 0);
		}
	}

	/* culculate cross */
	Point3f crvec;
	float crnorm;
	for(y = 0; y < imageH - 1; y++){
		for(x = 0; x < imageW - 1; x++){
			normals[y*imageW + x] += Point3f(-(hmap[y*imageW + (x+1)] - hmap[y*imageW + x]), -(hmap[(y+1)*imageW + x] - hmap[y*imageW + x]), 1);
		}
	}

	for(y = 0; y < imageH - 1; y++){
		for(x = 1; x < imageW; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x-1)] - hmap[y*imageW + x], -(hmap[(y+1)*imageW + x] - hmap[y*imageW + x]), 1);
		}
	}

	for(y = 1; y < imageH; y++){
		for(x = 1; x < imageW; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x-1)] - hmap[y*imageW + x], hmap[(y-1)*imageW + x] - hmap[y*imageW + x], 1);
		}
	}

	for(y = 1; y < imageH; y++){
		for(x = 0; x < imageW - 1; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x+1)] - hmap[y*imageW + x], hmap[(y-1)*imageW + x] - hmap[y*imageW + x], 1);
		}
	}

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			crnorm = sqrt(normals[y*imageW + x].x * normals[y*imageW + x].x + normals[y*imageW + x].y * normals[y*imageW + x].y + normals[y*imageW + x].z * normals[y*imageW + x].z);
			normals[y*imageW + x] = Point3f(normals[y*imageW + x].x / crnorm, normals[y*imageW + x].y / crnorm, normals[y*imageW + x].z / crnorm);
		}
	}
	/*
	Mat hmp_mat, hmp_col;
	Mat(imageH, imageW, CV_32FC1, hmap_normalized).clone().convertTo(hmp_mat, CV_8U, 255);
	cvtColor(hmp_mat, hmp_col, CV_GRAY2RGB);
	imwrite("hmap.tif", hmp_col);
		
	for(y = 0; y < imageH; y += 10){
		for(x = 0; x < imageW; x += 10){
			line(hmp_col, Point2i(x, y), Point2f((float)x + normals[y*imageW + x].x*20.0f, (float)y + normals[y*imageW + x].y*20.0f), Scalar(0, 255, 255));
		}
	}
	imshow("hmp", hmp_col);
	imwrite("hmap_nor.tif", hmp_col);
	*/
	delete[] h_Kernel_xy;
	delete[] h_Kernel_z;

	hmapEmpty = false;
}

bool Filter3D::saveHeightMap(const char filename[])
{
	if(isEmpty)return false;
	if(hmapEmpty)return false;

	ofstream fout;

	fout.open(filename, ios::out|ios::binary|ios::trunc);
	if(!fout){
		cout << "Failed in fstream.open (saveHeightMap)" << endl;
		return false;
	}

	int tmp;
	tmp = 120;	//header
	fout.write((char *) &tmp, sizeof(int));
	tmp = 240;	//header
	fout.write((char *) &tmp, sizeof(int));

	tmp = imageW;
	fout.write((char *) &tmp, sizeof(int));
	tmp = imageH;
	fout.write((char *) &tmp, sizeof(int));

	float ftmp;
	for(int y = 0; y < imageH; y++){
		for(int x = 0; x < imageW; x++){
			ftmp = hmap[y*imageW + x];
			fout.write((char *) &ftmp, sizeof(float));
		}
	}

	cout << "HeightMap SAVED" << endl;
	return true;
}

bool Filter3D::readHeightMap(const char filename[])
{
	if(isEmpty)return false;

	ifstream fin;
	int size_x, size_y;
	unsigned int x, y;
	
	fin.open(filename, ios::in|ios::binary);
	if(!fin){
		cout << "Failed in fstream.open (readHeightMap)" << endl;
		return false;
	}

	int chk;
	fin.read((char *) &chk, sizeof(int));
	if(chk != 120)return false;
	fin.read((char *) &chk, sizeof(int));
	if(chk != 240)return false;

	fin.read((char *) &size_x, sizeof(int));
	if(size_x != imageW)return false;
	fin.read((char *) &size_y, sizeof(int));
	if(size_y != imageH)return false;
	
	float ftmp;
	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			fin.read((char *) &ftmp, sizeof(float));
			hmap[y*imageW + x] = ftmp;
		}
	}

	/* culculate cross */
	Point3f crvec;
	float crnorm;
	for(y = 0; y < imageH - 1; y++){
		for(x = 0; x < imageW - 1; x++){
			normals[y*imageW + x] += Point3f(-(hmap[y*imageW + (x+1)] - hmap[y*imageW + x]), -(hmap[(y+1)*imageW + x] - hmap[y*imageW + x]), 1);
		}
	}

	for(y = 0; y < imageH - 1; y++){
		for(x = 1; x < imageW; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x-1)] - hmap[y*imageW + x], -(hmap[(y+1)*imageW + x] - hmap[y*imageW + x]), 1);
		}
	}

	for(y = 1; y < imageH; y++){
		for(x = 1; x < imageW; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x-1)] - hmap[y*imageW + x], hmap[(y-1)*imageW + x] - hmap[y*imageW + x], 1);
		}
	}

	for(y = 1; y < imageH; y++){
		for(x = 0; x < imageW - 1; x++){
			normals[y*imageW + x] += Point3f(hmap[y*imageW + (x+1)] - hmap[y*imageW + x], hmap[(y-1)*imageW + x] - hmap[y*imageW + x], 1);
		}
	}

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			crnorm = sqrt(normals[y*imageW + x].x * normals[y*imageW + x].x + normals[y*imageW + x].y * normals[y*imageW + x].y + normals[y*imageW + x].z * normals[y*imageW + x].z);
			normals[y*imageW + x] = Point3f(normals[y*imageW + x].x / crnorm, normals[y*imageW + x].y / crnorm, normals[y*imageW + x].z / crnorm);
		}
	}

	cout << "HeightMap READ" << endl;
	hmapEmpty = false;
	return true;
}

void Filter3D::heightMapBasedProjection(float *data, float offset, float depth, int range, float th)
{
	if(isEmpty)return;

	unsigned int x, y;
	float d;

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			float proj_max = -1.0f;
			for(d = depth; d <= depth + range; d += 1.0f){
				Point3f p;
				Point3i rp;
				float proj_tmp;

				p.x = (float)x + normals[y*imageW + x].x * d;
				p.y = (float)y + normals[y*imageW + x].y * d;
				p.z = (float)hmap[y*imageW + x] + normals[y*imageW + x].z * d + offset;

				if(p.x < 0 || p.y < 0 || p.z < 0 || p.x > imageW-1 || p.y > imageH-1 || p.z > imageZ-1){
					if(proj_max < 0.0f)proj_max = 0.0f;
					break;
				}

				rp.x = (int)(p.x);
				rp.y = (int)(p.y);
				rp.z = (int)(p.z);

				if(rp.x == imageW - 1)rp.x--;
				if(rp.y == imageH - 1)rp.y--;
				if(rp.z == imageZ - 1)rp.z--;

				p.x -= rp.x;
				p.y -= rp.y;
				p.z -= rp.z;

				float c000 = data[rp.z*imageH*imageW + rp.y*imageW + rp.x];
				float c100 = data[rp.z*imageH*imageW + rp.y*imageW + (rp.x + 1)];
				float c010 = data[rp.z*imageH*imageW + (rp.y + 1)*imageW + rp.x];
				float c001 = data[(rp.z + 1)*imageH*imageW + rp.y*imageW + rp.x];
				float c101 = data[(rp.z + 1)*imageH*imageW + rp.y*imageW + (rp.x + 1)];
				float c011 = data[(rp.z + 1)*imageH*imageW + (rp.y + 1)*imageW + rp.x];
				float c110 = data[rp.z*imageH*imageW + (rp.y + 1)*imageW + (rp.x + 1)];
				float c111 = data[(rp.z + 1)*imageH*imageW + (rp.y + 1)*imageW + (rp.x + 1)];
				proj_tmp =	c000*(1.0f-p.x)*(1.0f-p.y)*(1.0f-p.z) + 
							c100*p.x*(1.0f-p.y)*(1.0f-p.z) + 
							c010*(1.0f-p.x)*p.y*(1.0f-p.z) +
							c001*(1.0f-p.x)*(1.0f-p.y)*p.z +
							c101*p.x*(1.0f-p.y)*p.z +
							c011*(1.0f-p.x)*p.y*p.z +
							c110*p.x*p.y*(1.0f-p.z) +
							c111*p.x*p.y*p.z;

				if(th > 0.0f){
					if(proj_tmp > th){
						proj_max = 1.0f;
						break;
					}
					else proj_max = 0.0f;
				}
				else{
					if(proj_tmp > proj_max)proj_max = proj_tmp;
				}
			}
			bufXY[y*imageW*3 + x*3    ] = (unsigned int)(proj_max * 255);
			bufXY[y*imageW*3 + x*3 + 1] = (unsigned int)(proj_max * 255);
			bufXY[y*imageW*3 + x*3 + 2] = (unsigned int)(proj_max * 255);
		}
	}

}

void Filter3D::heightMapSimpleProjection(float *data, float offset, float depth, int range, bool isBinarize, float th, bool dmap_isEnable, int d_range)
{
	if(isEmpty)return;

	unsigned int x, y;
	float d;

	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			float proj_max = -1.0f;
			float proj_depth = FLT_MAX;
			for(d = depth; d <= depth + range; d += 1.0f){
				Point3f p;
				Point3i rp;
				float proj_tmp;
				float d_tmp;

				p.x = (float)x;
				p.y = (float)y;
				p.z = (float)hmap[y*imageW + x] + 1.0f * d + offset;

				if(p.z < 0 || p.z > imageZ-1){
					if(proj_max < 0.0f)proj_max = 0.0f;
					break;
				}

				rp.x = (int)(p.x);
				rp.y = (int)(p.y);
				rp.z = (int)(p.z);

				if(rp.z == imageZ - 1)rp.z--;

				p.x -= rp.x;
				p.y -= rp.y;
				p.z -= rp.z;

				float c0 = data[rp.z*imageH*imageW + rp.y*imageW + rp.x];
				float c1 = data[(rp.z+1)*imageH*imageW + rp.y*imageW + rp.x];
				proj_tmp =	c0*(1.0f-p.z) + c1*p.z;

				if(th > 0.0f && isBinarize){
					if(proj_tmp > th){
						if(dmap_isEnable){
							p.z = (float)hmap[y*imageW + x] + 1.0f * d;
							if(p.z < 0){
								proj_max = 1.0f;
								break;
							}
							else if(p.z > imageZ-1){
								proj_max = 0.0f;
								break;
							}
							rp.z = (int)(p.z);
							if(rp.z == imageZ - 1)rp.z--;
							p.z -= rp.z;
							float d0 = dmap[rp.z*imageH*imageW + rp.y*imageW + rp.x];
							float d1 = dmap[(rp.z+1)*imageH*imageW + rp.y*imageW + rp.x];
							d_tmp =	d0*(1.0f-p.z) + d1*p.z;
							proj_max = 1.0f * (1.0f - d_tmp / (float)d_range);
							if(proj_max < 0.0f)proj_max = 0.0f;
						}
						else proj_max = 1.0f;

						break;
					}
					else proj_max = 0.0f;
				}
				else{
					if(proj_tmp > proj_max){
						proj_max = proj_tmp;

						if(dmap_isEnable){
							p.z = (float)hmap[y*imageW + x] + 1.0f * d;
							if(p.z < 0){
								proj_depth = 0.0f;
							}
							else if(p.z > imageZ-1){
								proj_depth = FLT_MAX;
							}
							else{
								rp.z = (int)(p.z);
								if(rp.z == imageZ - 1)rp.z--;
								p.z -= rp.z;
								float d0 = dmap[rp.z*imageH*imageW + rp.y*imageW + rp.x];
								float d1 = dmap[(rp.z+1)*imageH*imageW + rp.y*imageW + rp.x];
								proj_depth = d0*(1.0f-p.z) + d1*p.z;
							}
						}

					}
				}
				
			}
			//hmbproj[y*imageW + x] = proj_max;
			if(th > 0.0f && isBinarize){
				bufXY[y*imageW*3 + x*3    ] = (unsigned int)(proj_max * 255);
				bufXY[y*imageW*3 + x*3 + 1] = (unsigned int)(proj_max * 255);
				bufXY[y*imageW*3 + x*3 + 2] = (unsigned int)(proj_max * 255);
			}
			else{
				if(proj_max < th)proj_max = 0.0f;
				if(dmap_isEnable){
					float h;
					unsigned int rmax, gmax, bmax;
					h = proj_depth / d_range * 270.0f;
					if(h > 270.0f)h = 270.0f;
					hsv2rgb(h, 1.0f, proj_max, rmax, gmax, bmax);
					bufXY[y*imageW*3 + x*3    ] = rmax;
					bufXY[y*imageW*3 + x*3 + 1] = gmax;
					bufXY[y*imageW*3 + x*3 + 2] = bmax;
				}
				else{
					bufXY[y*imageW*3 + x*3    ] = (unsigned int)(proj_max * 255);
					bufXY[y*imageW*3 + x*3 + 1] = (unsigned int)(proj_max * 255);
					bufXY[y*imageW*3 + x*3 + 2] = (unsigned int)(proj_max * 255);
				}
			}
		}
	}

}

void Filter3D::simpleProjection(float *data, float bc_max, float bc_min, int slice_id, int range, bool seg_show, int seg_minVol, bool dmap_isEnable, float d_coefficient, float d_order, bool zbc_isEnable, int zbc_channel, float zbc_coefficient, float zbc_order)
{
	if(isEmpty)return;
	if(slice_id >= imageZ || slice_id < 0)return;

	float *zbc_data;
	if(zbc_channel >= 0 && zbc_channel < nChannels)zbc_data = rawdata_resized + imageW*imageH*imageZ*zbc_channel;
	else zbc_data = rawdata_resized;

	if(dmap_isEnable && !hmapEmpty){
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageW; j++){
				float imax = 0.0f;
				float zbc_sum = 0.0f;
				if(zbc_isEnable)for(unsigned int s = 0; s < slice_id; s++)zbc_sum += zbc_data[s*imageW*imageH + i*imageW + j]; 

				for(unsigned int k = slice_id; (k < imageZ) && (k <= slice_id + range); k++){
					float d = k - hmap[i*imageW + j];
					if(d < 0.0f)d = 0.0f;
					float v = data[k*imageW*imageH + i*imageW + j] * pow(1.0f + d_coefficient * d / imageZ, d_order);
					if(zbc_isEnable){
						zbc_sum += zbc_data[k*imageW*imageH + i*imageW + j];
						v *= pow(1.0f + zbc_coefficient * zbc_sum / imageZ, zbc_order);
					}
					imax = max(imax, v);
				}
				imax = range_adjustment(imax, bc_max, bc_min);
				bufXY[i*imageW*3 + j*3    ] = (unsigned char)(imax * 255);
				bufXY[i*imageW*3 + j*3 + 1] = (unsigned char)(imax * 255);
				bufXY[i*imageW*3 + j*3 + 2] = (unsigned char)(imax * 255);
			}
		}
	}
	else{
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageW; j++){
				float imax = 0.0f;
				float zbc_sum = 0.0f;
				if(zbc_isEnable)for(unsigned int s = 0; s < slice_id; s++)zbc_sum += zbc_data[s*imageW*imageH + i*imageW + j]; 
				
				for(unsigned int k = slice_id; (k < imageZ) && (k <= slice_id + range); k++){
					float v = data[k*imageW*imageH + i*imageW + j];
					if(zbc_isEnable){
						zbc_sum += zbc_data[k*imageW*imageH + i*imageW + j];
						v *= pow(1.0f + zbc_coefficient * zbc_sum / imageZ, zbc_order);
					}
					imax = max(imax, v);
				}
				imax = range_adjustment(imax, bc_max, bc_min);
				bufXY[i*imageW*3 + j*3    ] = (unsigned char)(imax * 255);
				bufXY[i*imageW*3 + j*3 + 1] = (unsigned char)(imax * 255);
				bufXY[i*imageW*3 + j*3 + 2] = (unsigned char)(imax * 255);
			}
		}
	}

	if(seg_colors->size() == segments->size() && seg_colors->size() > 0 && seg_show){
		unsigned int r, g, b;
		vector<char> seg_isSelected(segments->size(), 0);

		for(unsigned int i = 0; i < selected_segment->size(); i++)seg_isSelected[(*selected_segment)[i]] = 1;
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageW; j++){
				int id = segdata[slice_id*imageW*imageH + i*imageW + j];
				if(id >= 0){
					if((*segments)[id].size() > seg_minVol){
						hsv2rgb((*seg_colors)[id], 1.0f, 1.0f, r, g, b);
						if(seg_isSelected[id] == 1){
							r = r + 50 <= 255 ? r + 50: 255;
							g = g + 50 <= 255 ? g + 50: 255;
							b = b + 50 <= 255 ? b + 50: 255;
						}
						else {r /= 3; g /= 3; b /= 3;}
						bufXY[i*imageW*3 + j*3    ] = (unsigned char)r;
						bufXY[i*imageW*3 + j*3 + 1] = (unsigned char)g;
						bufXY[i*imageW*3 + j*3 + 2] = (unsigned char)b;
					}
				}
			}
		}
	}

}

void Filter3D::setBufferYZ(float *data, int currentX, float bc_max, float bc_min, bool seg_show, int seg_minVol, bool hmap_isVisible, int hmap_offset, int hmap_range, bool dmap_isEnable, float d_coefficient, float d_order, bool zbc_isEnable, int zbc_channel, float zbc_coefficient, float zbc_order)
{
	if(isEmpty)return;

	if(dmap_isEnable && !hmapEmpty){
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageZ; j++){
				//float v = data[j*imageW*imageH + i*imageW + currentX] * pow(1.0f + d_coefficient * dmap[j*imageW*imageH + i*imageW + currentX] / dmax, d_order);
				float d = j - hmap[i*imageW + currentX];
				if(d < 0.0f)d = 0.0f;
				float v = data[j*imageW*imageH + i*imageW + currentX] * pow(1.0f + d_coefficient * d / imageZ, d_order);
				v = range_adjustment(v, bc_max, bc_min);
				bufYZ[i*imageZ*3 + j*3    ] = (unsigned char)(v * 255);
				bufYZ[i*imageZ*3 + j*3 + 1] = (unsigned char)(v * 255);
				bufYZ[i*imageZ*3 + j*3 + 2] = (unsigned char)(v * 255);
			}
		}
	}
	else{
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageZ; j++){
				float v = range_adjustment(data[j*imageW*imageH + i*imageW + currentX], bc_max, bc_min);
				bufYZ[i*imageZ*3 + j*3    ] = (unsigned char)(v * 255);
				bufYZ[i*imageZ*3 + j*3 + 1] = (unsigned char)(v * 255);
				bufYZ[i*imageZ*3 + j*3 + 2] = (unsigned char)(v * 255);
			}
		}
	}

	if(hmap_isVisible && !hmapEmpty){
		for(unsigned int i = 0; i < imageH; i++){
			int z = (int)(hmap[i*imageW + currentX] + hmap_offset + hmap_range);
			if(z < 0)z = 0;
			if(z > imageZ-1)z = imageZ-1;
			bufYZ[i*imageZ*3 + z*3    ] = 0;
			bufYZ[i*imageZ*3 + z*3 + 1] = 255;
			bufYZ[i*imageZ*3 + z*3 + 2] = 0;

			z = (int)(hmap[i*imageW + currentX] + hmap_offset);
			if(z < 0)z = 0;
			if(z > imageZ-1)z = imageZ-1;
			bufYZ[i*imageZ*3 + z*3    ] = 255;
			bufYZ[i*imageZ*3 + z*3 + 1] = 0;
			bufYZ[i*imageZ*3 + z*3 + 2] = 255;
		}
	}

	if(seg_colors->size() == segments->size() && seg_colors->size() > 0 && seg_show){
		unsigned int r, g, b;
		vector<char> seg_isSelected(segments->size(), 0);

		for(unsigned int i = 0; i < selected_segment->size(); i++)seg_isSelected[(*selected_segment)[i]] = 1;
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageZ; j++){
				int id = segdata[j*imageW*imageH + i*imageW + currentX];
				if(id >= 0){
					if((*segments)[id].size() > seg_minVol){
						hsv2rgb((*seg_colors)[id], 1.0f, 1.0f, r, g, b);
						if(seg_isSelected[id] == 1){
							r = r + 50 <= 255 ? r + 50: 255;
							g = g + 50 <= 255 ? g + 50: 255;
							b = b + 50 <= 255 ? b + 50: 255;
						}
						else {r /= 3; g /= 3; b /= 3;}
						bufYZ[i*imageZ*3 + j*3    ] = (unsigned char)r;
						bufYZ[i*imageZ*3 + j*3 + 1] = (unsigned char)g;
						bufYZ[i*imageZ*3 + j*3 + 2] = (unsigned char)b;
					}
				}
			}
		}
	}
}

void Filter3D::setBufferZX(float *data, int currentY, float bc_max, float bc_min, bool seg_show, int seg_minVol, bool hmap_isVisible, int hmap_offset, int hmap_range, bool dmap_isEnable, float d_coefficient, float d_order, bool zbc_isEnable, int zbc_channel, float zbc_coefficient, float zbc_order)
{
	if(isEmpty)return;

	float *zbc_data;
	if(zbc_channel >= 0 && zbc_channel < nChannels)zbc_data = rawdata_resized + imageW*imageH*imageZ*zbc_channel;
	else zbc_data = rawdata_resized;

	if(dmap_isEnable && !hmapEmpty){
		for(unsigned int j = 0; j < imageW; j++){
			float zbc_sum = 0.0f;
			for(unsigned int i = 0; i < imageZ; i++){
				float d = i - hmap[currentY*imageW + j];
				if(d < 0.0f)d = 0.0f;
				float v = data[i*imageW*imageH + currentY*imageW + j] * pow(1.0f + d_coefficient * d / imageZ, d_order);
				if(zbc_isEnable){
					zbc_sum += v;//zbc_data[i*imageW*imageH + currentY*imageW + j];
					v *= pow(1.0f + zbc_coefficient * zbc_sum / imageZ, zbc_order);
				}
				v = range_adjustment(v, bc_max, bc_min);
				bufZX[i*imageW*3 + j*3    ] = (unsigned char)(v * 255);
				bufZX[i*imageW*3 + j*3 + 1] = (unsigned char)(v * 255);
				bufZX[i*imageW*3 + j*3 + 2] = (unsigned char)(v * 255);
			}
		}
	}
	else{
		for(unsigned int j = 0; j < imageW; j++){
			float zbc_sum = 0.0f;
			for(unsigned int i = 0; i < imageZ; i++){
				float v = data[i*imageW*imageH + currentY*imageW + j];
				if(zbc_isEnable){
					zbc_sum += v;//zbc_data[i*imageW*imageH + currentY*imageW + j];
					v *= pow(1.0f + zbc_coefficient * zbc_sum / imageZ, zbc_order);
				}
				v = range_adjustment(v, bc_max, bc_min);
				bufZX[i*imageW*3 + j*3    ] = (unsigned char)(v * 255);
				bufZX[i*imageW*3 + j*3 + 1] = (unsigned char)(v * 255);
				bufZX[i*imageW*3 + j*3 + 2] = (unsigned char)(v * 255);
			}
		}
	}

	if(hmap_isVisible && !hmapEmpty){
		for(unsigned int i = 0; i < imageW; i++){
			int z = (int)(hmap[currentY*imageW + i] + hmap_offset + hmap_range);
			if(z < 0)z = 0;
			if(z > imageZ-1)z = imageZ-1;
			bufZX[z*imageW*3 + i*3    ] = 0;
			bufZX[z*imageW*3 + i*3 + 1] = 255;
			bufZX[z*imageW*3 + i*3 + 2] = 0;

			z = (int)(hmap[currentY*imageW + i] + hmap_offset);
			if(z < 0)z = 0;
			if(z > imageZ-1)z = imageZ-1;
			bufZX[z*imageW*3 + i*3    ] = 255;
			bufZX[z*imageW*3 + i*3 + 1] = 0;
			bufZX[z*imageW*3 + i*3 + 2] = 255;
		}
	}
	
	if(seg_colors->size() == segments->size() && seg_colors->size() > 0 && seg_show){
		unsigned int r, g, b;
		vector<char> seg_isSelected(segments->size(), 0);

		for(unsigned int i = 0; i < selected_segment->size(); i++)seg_isSelected[(*selected_segment)[i]] = 1;
		for(unsigned int i = 0; i < imageZ; i++){
			for(unsigned int j = 0; j < imageW; j++){
				int id = segdata[i*imageW*imageH + currentY*imageW + j];
				if(id >= 0){
					if((*segments)[id].size() > seg_minVol){
						hsv2rgb((*seg_colors)[id], 1.0f, 1.0f, r, g, b);
						if(seg_isSelected[id] == 1){
							r = r + 50 <= 255 ? r + 50: 255;
							g = g + 50 <= 255 ? g + 50: 255;
							b = b + 50 <= 255 ? b + 50: 255;
						}
						else {r /= 3; g /= 3; b /= 3;}
						bufZX[i*imageW*3 + j*3    ] = (unsigned char)r;
						bufZX[i*imageW*3 + j*3 + 1] = (unsigned char)g;
						bufZX[i*imageW*3 + j*3 + 2] = (unsigned char)b;
					}
				}
			}
		}
	}
	
}

void Filter3D::setHeightMapToBufferXY()
{
	int x, y;

	if(!hmapEmpty){
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageW; j++){
				bufXY[i*imageW*3 + j*3    ] = (unsigned char)(hmap_normalized[i*imageW + j] * 255);
				bufXY[i*imageW*3 + j*3 + 1] = (unsigned char)(hmap_normalized[i*imageW + j] * 255);
				bufXY[i*imageW*3 + j*3 + 2] = (unsigned char)(hmap_normalized[i*imageW + j] * 255);
			}
		}
		Mat hmp_col = Mat(imageH, imageW, CV_8UC3, bufXY);
		for(y = 0; y < imageH; y += 10){
			for(x = 0; x < imageW; x += 10){
				line(hmp_col, Point2i(x, y), Point2f((float)x + normals[y*imageW + x].x*20.0f, (float)y + normals[y*imageW + x].y*20.0f), Scalar(255, 255, 0));
			}
		}
	}
	else{
		for(unsigned int i = 0; i < imageH; i++){
			for(unsigned int j = 0; j < imageW; j++){
				bufXY[i*imageW*3 + j*3    ] = 0;
				bufXY[i*imageW*3 + j*3 + 1] = 0;
				bufXY[i*imageW*3 + j*3 + 2] = 0;
			}
		}
	}

}

void Filter3D::generateGraphTemplate()
{
	float gridy = 5;
	int gridx = 20;
	int i;
	int margin = 3;
	int scalefac = 2;
	Size textsize;
	stringstream ss;
	
	graphW = max(max(imageW, imageH), imageZ) * scalefac + GRAPH_TEXT_AREA_WIDTH;
	graphH = GRAPH_DATA_AREA_HEIGHT + GRAPH_TEXT_AREA_HEIGHT;

	graph_tmpX = new unsigned char[graphW*graphH*3];
	graph_tmpY = new unsigned char[graphW*graphH*3];
	graph_tmpZ = new unsigned char[graphW*graphH*3];
	graph = new unsigned char[graphW*graphH*3];

	for(i = 0; i < graphH*graphW*3; i++){
		graph_tmpX[i] = 0;
		graph_tmpY[i] = 0;
		graph_tmpZ[i] = 0;
	}
	
	Mat gtmatx = Mat(graphH, graphW, CV_8UC3, graph_tmpX);
	Mat gtmaty = Mat(graphH, graphW, CV_8UC3, graph_tmpY);
	Mat gtmatz = Mat(graphH, graphW, CV_8UC3, graph_tmpZ);

	for(i = 0; i <= gridy; i++){
		line(gtmatx,
			 Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y + i*(GRAPH_DATA_AREA_HEIGHT / gridy)),
			 Point2i(GRAPH_DATA_AREA_COORD_X + graphW - GRAPH_TEXT_AREA_WIDTH, GRAPH_DATA_AREA_COORD_Y + i*(GRAPH_DATA_AREA_HEIGHT / gridy)),
			 Scalar(50, 50, 50)
			 );
		ss.str("");
		ss << std::fixed << std::setprecision(1) << 1.0f - i * 1.0f / (float)gridy;
		textsize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.4, 1, NULL);
		putText(gtmatx, ss.str(), Point2i(GRAPH_DATA_AREA_COORD_X - textsize.width - margin, GRAPH_DATA_AREA_COORD_Y + i*(GRAPH_DATA_AREA_HEIGHT / gridy) + textsize.height/2), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255));
	}

	memcpy(graph_tmpY, graph_tmpX, sizeof(unsigned char)*graphW*graphH*3);
	memcpy(graph_tmpZ, graph_tmpX, sizeof(unsigned char)*graphW*graphH*3);

	for(i = 0; i <= imageW/gridx; i++){
		line(gtmatx,
			 Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageW, GRAPH_DATA_AREA_COORD_Y),
			 Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageW, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			 Scalar(50, 50, 50)
			 );
		ss.str("");
		ss << std::fixed << std::setprecision(1) << i*gridx;
		textsize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.4, 1, NULL);
		putText(gtmatx, ss.str(), Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageW - textsize.width/2, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT + textsize.height + margin), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255));
	}
	
	for(i = 0; i <= imageH/gridx; i++){
		line(gtmaty,
			 Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageH, GRAPH_DATA_AREA_COORD_Y),
			 Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageH, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			 Scalar(50, 50, 50)
			 );
		ss.str("");
		ss << std::fixed << std::setprecision(1) << i*gridx;
		textsize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.4, 1, NULL);
		putText(gtmaty, ss.str(), Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageH - textsize.width/2, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT + textsize.height + margin), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255));
	}

	gtmatz = Mat(graphH, graphW, CV_8UC3, graph_tmpZ);
	for(i = 0; i <= imageZ/gridx; i++){
		line(gtmatz,
			 Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageZ, GRAPH_DATA_AREA_COORD_Y),
			 Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageZ, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			 Scalar(50, 50, 50)
			 );
		ss.str("");
		ss << std::fixed << std::setprecision(1) << i*gridx;
		textsize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.4, 1, NULL);
		putText(gtmatz, ss.str(), Point2i(GRAPH_DATA_AREA_COORD_X + i*gridx*(graphW - GRAPH_TEXT_AREA_WIDTH)/imageZ - textsize.width/2, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT + textsize.height + margin), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255));
	}

	rectangle(gtmatx,
			  Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y),
			  Point2i(GRAPH_DATA_AREA_COORD_X + graphW - GRAPH_TEXT_AREA_WIDTH, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			  Scalar(255, 255, 255)
			  );
	
	rectangle(gtmaty,
			  Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y),
			  Point2i(GRAPH_DATA_AREA_COORD_X + graphW - GRAPH_TEXT_AREA_WIDTH, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			  Scalar(255, 255, 255)
			  );

	rectangle(gtmatz,
			  Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y),
			  Point2i(GRAPH_DATA_AREA_COORD_X + graphW - GRAPH_TEXT_AREA_WIDTH, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			  Scalar(255, 255, 255)
			  );
}

void Filter3D::setXPlotGraph(int x, int y, int z, float bc_max, float bc_min, bool depthBC, float d_coefficient, float d_order)
{
	if(x < 0 || x >= imageW || y < 0 || y >= imageH || z < 0 || z >= imageZ) return;
	
	int dataAreaW = graphW - GRAPH_TEXT_AREA_WIDTH;
	float val1, val2;
	Point2i g_origin = Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT);

	Mat gtmat = Mat(graphH, graphW, CV_8UC3, graph_tmpX);
	Mat graphmat = Mat(graphH, graphW, CV_8UC3, graph);
	
	Mat src_data = Mat::zeros(graphH, graphW, CV_8UC3);
	Mat dst_data = Mat::zeros(graphH, graphW, CV_8UC3);

	
	if(!hmapEmpty && depthBC){
		for(int i = 0; i < imageW - 1; i++){
			float d = z - hmap[y*imageW + i];
			if(d < 0.0f)d = 0.0f;
			val1 = imgdata[z*imageW*imageH + y*imageW + i] * pow(1.0f + d_coefficient * d / imageZ, d_order);
			val1 = range_adjustment(val1, bc_max, bc_min);
			
			d = z - hmap[y*imageW + i + 1];
			if(d < 0.0f)d = 0.0f;
			val2 = imgdata[z*imageW*imageH + y*imageW + i + 1] * pow(1.0f + d_coefficient * d / imageZ, d_order);
			val2 = range_adjustment(val2, bc_max, bc_min);
			line(src_data, g_origin + Point2i(i*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 0, 0));
		}
	}
	else if(bc_max != 1.0f || bc_min != 0.0f){
		for(int i = 0; i < imageW - 1; i++){
			val1 = range_adjustment(imgdata[z*imageW*imageH + y*imageW + i], bc_max, bc_min);
			val2 = range_adjustment(imgdata[z*imageW*imageH + y*imageW + i + 1], bc_max, bc_min);
			line(src_data, g_origin + Point2i(i*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 0, 0));
		}
	}

	for(int i = 0; i < imageW - 1; i++){
		val1 = imgdata[z*imageW*imageH + y*imageW + i];
		val2 = imgdata[z*imageW*imageH + y*imageW + i + 1];
		line(src_data, g_origin + Point2i(i*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 255, 0));
	}

	for(int i = 0; i < imageW - 1; i++){
		val1 = dstdata[z*imageW*imageH + y*imageW + i];
		val2 = dstdata[z*imageW*imageH + y*imageW + i + 1];
		line(dst_data, g_origin + Point2i(i*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageW, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(0, 255, 255));
	}

	add(src_data, dst_data, src_data);
	add(src_data, gtmat, graphmat);
	
	rectangle(graphmat,
			  Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y),
			  Point2i(GRAPH_DATA_AREA_COORD_X + graphW - GRAPH_TEXT_AREA_WIDTH, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			  Scalar(255, 255, 255)
			  );
	
	circle(graphmat, g_origin + Point2i(x*dataAreaW/imageW, 0), 3, Scalar(255, 0, 255), CV_FILLED);

}

void Filter3D::setYPlotGraph(int x, int y, int z, float bc_max, float bc_min, bool depthBC, float d_coefficient, float d_order)
{
	if(x < 0 || x >= imageW || y < 0 || y >= imageH || z < 0 || z >= imageZ) return;

	int dataAreaW = graphW - GRAPH_TEXT_AREA_WIDTH;
	float val1, val2;
	Point2i g_origin = Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT);

	Mat gtmat = Mat(graphH, graphW, CV_8UC3, graph_tmpY);
	Mat graphmat = Mat(graphH, graphW, CV_8UC3, graph);
	
	Mat src_data = Mat::zeros(graphH, graphW, CV_8UC3);
	Mat dst_data = Mat::zeros(graphH, graphW, CV_8UC3);

	
	if(!hmapEmpty && depthBC){
		for(int i = 0; i < imageH - 1; i++){
			float d = z - hmap[i*imageW + x];
			if(d < 0.0f)d = 0.0f;
			val1 = imgdata[z*imageW*imageH + i*imageW + x] * pow(1.0f + d_coefficient * d / imageZ, d_order);
			val1 = range_adjustment(val1, bc_max, bc_min);
			
			d = z - hmap[(i+1)*imageW + x];
			if(d < 0.0f)d = 0.0f;
			val2 = imgdata[z*imageW*imageH + (i+1)*imageW + x] * pow(1.0f + d_coefficient * d / imageZ, d_order);
			val2 = range_adjustment(val2, bc_max, bc_min);
			line(src_data, g_origin + Point2i(i*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 0, 0));
		}
	}
	else if(bc_max != 1.0f || bc_min != 0.0f){
		for(int i = 0; i < imageH - 1; i++){
			val1 = range_adjustment(imgdata[z*imageW*imageH + i*imageW + x], bc_max, bc_min);
			val2 = range_adjustment(imgdata[z*imageW*imageH + (i+1)*imageW + x], bc_max, bc_min);
			line(src_data, g_origin + Point2i(i*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 0, 0));;
		}
	}

	for(int i = 0; i < imageH - 1; i++){
		val1 = imgdata[z*imageW*imageH + i*imageW + x];
		val2 = imgdata[z*imageW*imageH + (i+1)*imageW + x];
		line(src_data, g_origin + Point2i(i*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 255, 0));
	}

	for(int i = 0; i < imageH - 1; i++){
		val1 = dstdata[z*imageW*imageH + i*imageW + x];
		val2 = dstdata[z*imageW*imageH + (i+1)*imageW + x];
		line(dst_data, g_origin + Point2i(i*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageH, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(0, 255, 255));
	}

	add(src_data, dst_data, src_data);
	add(src_data, gtmat, graphmat);
	
	rectangle(graphmat,
			  Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y),
			  Point2i(GRAPH_DATA_AREA_COORD_X + graphW - GRAPH_TEXT_AREA_WIDTH, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			  Scalar(255, 255, 255)
			  );
	
	circle(graphmat, g_origin + Point2i(y*dataAreaW/imageH, 0), 3, Scalar(255, 0, 255), CV_FILLED);

}

void Filter3D::setZPlotGraph(int x, int y, int z, float bc_max, float bc_min, bool depthBC, float d_coefficient, float d_order)
{
	if(x < 0 || x >= imageW || y < 0 || y >= imageH || z < 0 || z >= imageZ) return;

	int dataAreaW = graphW - GRAPH_TEXT_AREA_WIDTH;
	float val1, val2;
	Point2i g_origin = Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT);

	Mat gtmat = Mat(graphH, graphW, CV_8UC3, graph_tmpZ);
	Mat graphmat = Mat(graphH, graphW, CV_8UC3, graph);
	
	Mat src_data = Mat::zeros(graphH, graphW, CV_8UC3);
	Mat dst_data = Mat::zeros(graphH, graphW, CV_8UC3);

	if(!hmapEmpty && depthBC){
		for(int i = 0; i < imageZ - 1; i++){
			float d = i - hmap[y*imageW + x];
			if(d < 0.0f)d = 0.0f;
			val1 = imgdata[i*imageW*imageH + y*imageW + x] * pow(1.0f + d_coefficient * d / imageZ, d_order);
			val1 = range_adjustment(val1, bc_max, bc_min);
			
			d = (i + 1) - hmap[y*imageW + x];
			if(d < 0.0f)d = 0.0f;
			val2 = imgdata[(i+1)*imageW*imageH + y*imageW + x] * pow(1.0f + d_coefficient * d / imageZ, d_order);
			val2 = range_adjustment(val2, bc_max, bc_min);
			line(src_data, g_origin + Point2i(i*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 0, 0));
		}
	}
	else if(bc_max != 1.0f || bc_min != 0.0f){
		for(int i = 0; i < imageZ - 1; i++){
			val1 = range_adjustment(imgdata[i*imageW*imageH + y*imageW + x], bc_max, bc_min);
			val2 = range_adjustment(imgdata[(i+1)*imageW*imageH + y*imageW + x], bc_max, bc_min);
			line(src_data, g_origin + Point2i(i*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 0, 0));
		}
	}

	for(int i = 0; i < imageZ - 1; i++){
		val1 = imgdata[i*imageW*imageH + y*imageW + x];
		val2 = imgdata[(i+1)*imageW*imageH + y*imageW + x];
		line(src_data, g_origin + Point2i(i*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(255, 255, 0));
	}

	for(int i = 0; i < imageZ - 1; i++){
		val1 = dstdata[i*imageW*imageH + y*imageW + x];
		val2 = dstdata[(i+1)*imageW*imageH + y*imageW + x];
		line(dst_data, g_origin + Point2i(i*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val1), g_origin + Point2i((i+1)*dataAreaW/imageZ, -GRAPH_DATA_AREA_HEIGHT * val2), Scalar(0, 255, 255));
	}

	add(src_data, dst_data, src_data);
	add(src_data, gtmat, graphmat);
	
	rectangle(graphmat,
			  Point2i(GRAPH_DATA_AREA_COORD_X, GRAPH_DATA_AREA_COORD_Y),
			  Point2i(GRAPH_DATA_AREA_COORD_X + graphW - GRAPH_TEXT_AREA_WIDTH, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT),
			  Scalar(255, 255, 255)
			  );

	circle(graphmat, g_origin + Point2i(z*dataAreaW/imageZ, 0), 3, Scalar(255, 0, 255), CV_FILLED);

	if(seg_colors->size() == segments->size() && seg_colors->size() > 0){
		stringstream ss;
		ss.str("");
		ss << "Segment: " << segdata[z*imageH*imageW + y*imageW + x] << "  ";
		if(segdata[z*imageH*imageW + y*imageW + x] >= 0)ss << "BoundingBox: " << seg_bbox->at(segdata[z*imageH*imageW + y*imageW + x]).x << " "
																			  << seg_bbox->at(segdata[z*imageH*imageW + y*imageW + x]).y << " "
																			  << seg_bbox->at(segdata[z*imageH*imageW + y*imageW + x]).z << " "
																			  << seg_bbox->at(segdata[z*imageH*imageW + y*imageW + x]).width  << " "
																			  << seg_bbox->at(segdata[z*imageH*imageW + y*imageW + x]).height << " "
																			  << seg_bbox->at(segdata[z*imageH*imageW + y*imageW + x]).depth  << "  "
																			  << "Size: " << segments->at(segdata[z*imageH*imageW + y*imageW + x]).size();

		Size textsize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.4, 1, NULL);
		putText(graphmat, ss.str(), Point2i(10, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT + textsize.height*2 + 13), FONT_HERSHEY_SIMPLEX, 0.40, Scalar(255,255,0));
	}
	else{
		stringstream ss;
		ss.str("");
		ss << "Src Intensity: " << imgdata[z*imageH*imageW + y*imageW + x] << "  Dst Intensity: " << dstdata[z*imageH*imageW + y*imageW + x];
		Size textsize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.4, 1, NULL);
		putText(graphmat, ss.str(), Point2i(10, GRAPH_DATA_AREA_COORD_Y + GRAPH_DATA_AREA_HEIGHT + textsize.height*2 + 13), FONT_HERSHEY_SIMPLEX, 0.40, Scalar(255,255,0));
	}
}

void Filter3D::applyBC(float bc_max, float bc_min, bool dmap_isEnable, float d_coefficient, float d_order)
{
	if(dmap_isEnable && !hmapEmpty){
		for(unsigned int z = 0; z < imageZ; z++){
			for(unsigned int i = 0; i < imageH; i++){
				for(unsigned int j = 0; j < imageW; j++){
					//float v = imgdata[z*imageW*imageH + i*imageW + j] * pow(1.0f + d_coefficient * dmap[z*imageW*imageH + i*imageW + j] / dmax, d_order);
					float d = z - hmap[i*imageW + j];
					if(d < 0.0f)d = 0.0f;
					float v = imgdata[z*imageW*imageH + i*imageW + j] * pow(1.0f + d_coefficient * d / imageZ, d_order);
					imgdata[z*imageW*imageH + i*imageW + j] = range_adjustment(v, bc_max, bc_min);
				}
			}
		}
	}
	else{
		for(unsigned int z = 0; z < imageZ; z++){
			for(unsigned int i = 0; i < imageH; i++){
				for(unsigned int j = 0; j < imageW; j++){
					imgdata[z*imageW*imageH + i*imageW + j] = range_adjustment(imgdata[z*imageW*imageH + i*imageW + j], bc_max, bc_min);
				}
			}
		}
	}
	memcpy(dstdata, imgdata, imageW*imageH*imageZ*sizeof(float));
	memcpy(rawdata_resized + imageW*imageH*imageZ*curCh, imgdata, imageW*imageH*imageZ*sizeof(float));

	if(isEnableGPU)checkCudaErrors( cudaMemcpy(d_Input, imgdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
}

void Filter3D::generateDepthMapGPU()
{
	if(isEmpty)return;

	unsigned int x, y, z;
	
	cout << "generateDepthMapGPU():please wait..." << endl;
	
	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			for(z = 0; z < imageZ; z++){
				if(z > (unsigned int)hmap[y*imageW + x])dmap[z*imageH*imageW + y*imageW + x] = FLT_MAX;
				else dmap[z*imageH*imageW + y*imageW + x] = 0.0f;
			}
		}
	}

	checkCudaErrors( cudaMemcpy(d_Output, dmap, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_Buffer, hmap, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaDeviceSynchronize() );

	depthMapGPU(d_Output, d_Buffer, imageW, imageH, imageZ);
	checkCudaErrors( cudaMemcpy(dmap, d_Output, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	cout << "depthMapGPU(): finish" << endl;

	dmax = 0.0f;
	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			for(z = 0; z < imageZ; z++){
				//dmap[z*imageH*imageW + y*imageW + x] = sqrt(dmap[z*imageH*imageW + y*imageW + x]);
				dmax = max(dmax, dmap[z*imageH*imageW + y*imageW + x]);
			}
		}
	}
	cout << "dmax: " << dmax << endl;
	dmapEmpty = false;
}

void Filter3D::generateDepthMap()
{
	if(isEmpty)return;

	unsigned int x, y, z, hx, hy;
	float d;

	cout << "generateDepthMap():please wait..." << endl;
	dmax = 0.0f;
	for(y = 0; y < imageH; y++){
		for(x = 0; x < imageW; x++){
			//cout << hmap[y*imageW + x] + 1 << endl;
			for(z = (unsigned int)hmap[y*imageW + x] + 1; z < imageZ; z++){
				d = FLT_MAX;
				for(hy = 0; hy < imageH; hy++){
					for(hx = 0; hx < imageW; hx++){
						d = min(d, (float)((x-hx)*(x-hx) + (y-hy)*(y-hy) + (z-hmap[hy*imageW + hx])*(z-hmap[hy*imageW + hx])));
					}
				}
				
				dmap[z*imageH*imageW + y*imageW + x] = sqrt(d);
				dmax = max(dmax, d);
			}
		}
		cout << y << endl;
	}
	dmax = sqrt(dmax);
	cout << "dmax: " << dmax << endl;
	dmapEmpty = false;
}

void Filter3D::depthCode(float *data, float offset, int range)
{

}

void Filter3D::hsv2rgb(float h, float s, float v, unsigned int &r, unsigned int &g, unsigned int &b)
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

float Filter3D::range_adjustment(float value, float max, float min)
{
	if(value >= max)return 1.0f;
	else if(value <= min)return 0.0f;

	return (value - min) / (max - min);
}
/*
void Filter3D::drawSelection(int brush_size, int cx, int cy, int cz, int mode)
{
	int x, y, z;
	for(x = cx - brush_size; x <= cx + brush_size; x++){
		for(z = cz - brush_size; z <= cz + brush_size; z++){
			if(x >= 0 && x < imageW && z >= 0 && z < imageZ){
				if(brush_size*brush_size >= (cx-x)*(cx-x) + (cz-z)*(cz-z)){
					selectdata[z*imageW*imageH + cy*imageW + x] = (mode == 0) ? 0.5f : 0.0f;
				}
			}
		}
	}
	//cout << "drawSelection: " << cx << " " << cy << " " << cz << endl;
}

void Filter3D::copySelectionSliceZX(int src, int dst)
{
	if(src < 0 || src >= imageH || dst < 0 || dst >= imageH)return;

	for(int z = 0; z < imageZ; z++){
		for(int x = 0; x < imageW; x++){
			selectdata[z*imageW*imageH + dst*imageW + x] = selectdata[z*imageW*imageH + src*imageW + x];
		}
	}
}
*/
void Filter3D::maskDst(bool inv)
{
	unsigned int x, y, z;

	if(inv){
		for(y = 0; y < imageH; y++){
			for(x = 0; x < imageW; x++){
				for(z = 0; z < imageZ; z++){
					if(dstdata[z*imageH*imageW + y*imageW + x] >= 1.0f)imgdata[z*imageH*imageW + y*imageW + x] = 0.0f;
				}
			}
		}
	}
	else{
		for(y = 0; y < imageH; y++){
			for(x = 0; x < imageW; x++){
				for(z = 0; z < imageZ; z++){
					if(dstdata[z*imageH*imageW + y*imageW + x] <= 0.0f)imgdata[z*imageH*imageW + y*imageW + x] = 0.0f;
				}
			}
		}
	}

	if(isEnableGPU){
		checkCudaErrors( cudaMemcpy(d_Input, imgdata, imageZ * imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );
	}
}

void Filter3D::calcHmapArea(int reso)
{
	double h1, h2, x, y, imax, s;
	int hmpx, hmpy;
	int i, j;
	Point3f *c_normals = new Point3f[(imageW - 1)*(imageH - 1)];
	
	for(hmpx = 0; hmpx < imageW - 1; hmpx++){
		for(hmpy = 0; hmpy < imageH - 1; hmpy++){
			c_normals[hmpy*(imageW-1) + hmpx] = normals[hmpy*imageW + hmpx] + normals[hmpy*imageW + hmpx + 1] + normals[(hmpy + 1)*imageW + hmpx] + normals[(hmpy + 1)*imageW + hmpx + 1];
			c_normals[hmpy*(imageW-1) + hmpx].x /= 4.0f;
			c_normals[hmpy*(imageW-1) + hmpx].y /= 4.0f;
			c_normals[hmpy*(imageW-1) + hmpx].z /= 4.0f;
		}
	}
	
	int m = reso; //x軸の刻み数
	int n = reso; //y軸の刻み数
	double x0 = 0.0 , x1 = 1.0; //xの積分範囲
	double y0 = 0.0 , y1 = 1.0; //yの積分範囲
	double **f= new double*[m+1];  // double型 m+1 個分の領域を動的確保
	for (i= 0; i<=m; ++i) f[i]= new double[n+1];  // double型 n+1 個分の領域を動的確保
	
	imax = 0.0;
	for(hmpx = 0; hmpx < imageW - 2; hmpx++){
		for(hmpy = 0; hmpy < imageH - 2; hmpy++){
			for(i = 0; i <= m; i++){
				for (j = 0; j <= n; j++)  {
					x = x0 + (x1-x0)/double(m) * double(i);
					y = y0 + (y1-y0)/double(n) * double(j);
					f[i][j] = F(x,
								y,
								c_normals[hmpy*(imageW - 1) + hmpx],
								c_normals[hmpy*(imageW - 1) + hmpx + 1],
								c_normals[(hmpy + 1)*(imageW - 1) + hmpx],
								c_normals[(hmpy + 1)*(imageW - 1) + hmpx + 1]
								);
				}
			}
			h1 = (x1-x0) / double(m);
			h2 = (y1-y0) / double(n);
			areas[(hmpy + 1)*imageW + hmpx + 1] = s = simpe2(f, m, n, h1, h2); // 計算
			if(imax < s)imax = s;
			//printf("  積分値 : %22.15e\n", s);
		}
	}
	
	for(hmpx = 0; hmpx < imageW; hmpx++){
		areas[hmpx] = 1.0f;
		areas[(imageH-1)*imageW + hmpx] = 1.0f;
	}
	for(hmpy = 1; hmpy < imageH - 2; hmpy++){
		areas[hmpy*imageH] = 1.0f;
		areas[hmpy*imageH + imageW - 1] = 1.0f;
	}
	
	for (i= 0; i<=m; ++i) delete[] f[i]; // 動的に確保した領域をそれぞれ解放
	delete[] f;
	delete[] c_normals;

	Mat area_mat, tmp;
	//Mat(imageH, imageW, CV_32FC1, areas).clone().convertTo(area_mat, CV_8U, 255);
	tmp = Mat(imageH, imageW, CV_32FC1, areas).clone();
	tmp -= 1.0f;
	tmp /= imax - 1.0f;
	tmp.convertTo(area_mat, CV_8U, 255);

	printf("max : %22.15e\n", imax);
	imshow("AreaMap", area_mat);
	imwrite("area_map.tif", area_mat);

	for(hmpx = 0; hmpx < imageW; hmpx++){
		areas[hmpx] = 0.0f;
		areas[(imageH-1)*imageW + hmpx] = 0.0f;
	}
	for(hmpy = 1; hmpy < imageH - 2; hmpy++){
		areas[hmpy*imageH] = 0.0f;
		areas[hmpy*imageH + imageW - 1] = 0.0f;
	}
	tifio->SaveImageDataSingle("area_map32.tif", (char *)areas, imageW, imageH, sizeof(float)*8, 1);
	return;
}

double Filter3D::F(double x, double y, Point3f p00, Point3f p10, Point3f p01, Point3f p11) //被積分関数
{
	double fx, fy, fz;

	fz = (double)p00.z*(1.0 - x)*(1.0 - y) + (double)p10.z*x*(1.0 - y) + (double)p01.z*(1.0 - x)*y + (double)p11.z*x*y;
	fx = ((double)p00.x*(1.0 - x)*(1.0 - y) + (double)p10.x*x*(1.0 - y) + (double)p01.x*(1.0 - x)*y + (double)p11.x*x*y) / fz;
	fy = ((double)p00.y*(1.0 - x)*(1.0 - y) + (double)p10.y*x*(1.0 - y) + (double)p01.y*(1.0 - x)*y + (double)p11.y*x*y) / fz;

	return sqrt(1 + fx*fx + fy*fy);
}

double Filter3D::simpe2(double **f, const int m, const int n, double h1, double h2)
{
  int i, j;
  double v;
  double *temp;
  temp = new double[m+1];  //double型 m+1 個分の領域を動的確保

  for(i = 0; i <= m; i++){
    v = - f[i][0] + f[i][n];
    for(j = 0 ; j < n - 1; j += 2)
      v += (2 * f[i][j] + 4 * f[i][j + 1]);
    temp[i] = v;
  }
  v = - temp[0] + temp[m];
  for(i = 0; i < m - 1; i += 2)  v += (2 * temp[i] + 4 * temp[i + 1]);
  delete [] temp;
  return v * h1 * h2 / 9.0;
}

bool Filter3D::isSelectedSegment(int id)
{
	if(isEmpty)return false;
	if(segments == NULL || seg_bbox == NULL || segdata == NULL || selected_segment == NULL)return false;
	if(segments->empty() || seg_bbox->empty())return false;

	vector<int>::iterator ite = find(selected_segment->begin(), selected_segment->end(), id);
	return ite != selected_segment->end() ? true : false;
}

void Filter3D::selectSegment(int x, int y, int z)
{
	if(isEmpty)return;
	if(imageW <= 0 || imageH <= 0 || imageZ <= 0)return;
	if(segments == NULL || seg_bbox == NULL || segdata == NULL || selected_segment == NULL)return;
	if(segments->empty() || seg_bbox->empty())return;

	if(x >= 0 && x < imageW && y >= 0 && y < imageH && z >= 0 && z < imageZ && segdata[z*imageH*imageW + y*imageW + x] >= 0){
		
		vector<int>::iterator ite = find(selected_segment->begin(), selected_segment->end(), segdata[z*imageH*imageW + y*imageW + x]);
		if(ite == selected_segment->end())selected_segment->push_back(segdata[z*imageH*imageW + y*imageW + x]);
		else selected_segment->erase(ite);

	}

	cout << selected_segment->size() << endl;
}

void Filter3D::selectAllSegments()
{
	if(isEmpty)return;
	if(segments == NULL || seg_bbox == NULL || segdata == NULL || selected_segment == NULL)return;
	if(segments->empty() || seg_bbox->empty())return;

	deselectSegment();
	for(int i = 0; i < segments->size(); i++)selected_segment->push_back(i);
	
}

void Filter3D::deselectSegment()
{
	if(isEmpty)return;
	if(selected_segment != NULL)std::vector<int>().swap(*selected_segment);
}

void Filter3D::autoSelectSegmentsTH(float th)
{
	if(isEmpty)return;
	if(imageW <= 0 || imageH <= 0 || imageZ <= 0)return;
	if(segments == NULL || seg_bbox == NULL || segdata == NULL || selected_segment == NULL)return;
	if(segments->empty() || seg_bbox->empty())return;

	for(int i = 0; i < segments->size(); i++){
		if(!isSelectedSegment(i) && (*segments)[i].size() > 0){
			double sum = 0.0;
			for(int j = 0; j < (*segments)[i].size(); j++){
				sum += imgdata[(*segments)[i][j].z*imageH*imageW + (*segments)[i][j].y*imageW + (*segments)[i][j].x];
			}
			if(sum / (*segments)[i].size() >= th)selected_segment->push_back(i);
		}
	}
}

void Filter3D::autoDeselectSegmentsTH(float th)
{
	if(isEmpty)return;
	if(imageW <= 0 || imageH <= 0 || imageZ <= 0)return;
	if(segments == NULL || seg_bbox == NULL || segdata == NULL || selected_segment == NULL)return;
	if(segments->empty() || seg_bbox->empty())return;

	vector<int>::iterator ite = selected_segment->begin();
	while(ite != selected_segment->end()){
		double sum = 0.0;
		for(int j = 0; j < (*segments)[*ite].size(); j++){
			sum += imgdata[(*segments)[*ite][j].z*imageH*imageW + (*segments)[*ite][j].y*imageW + (*segments)[*ite][j].x];
		}
		if(sum / (*segments)[*ite].size() >= th)ite = selected_segment->erase(ite);
		else ite++;
	}
}

void Filter3D::cropSegments()
{
	if(isEmpty)return;
	if(imageW <= 0 || imageH <= 0 || imageZ <= 0)return;
	if(segments == NULL || seg_bbox == NULL || segdata == NULL)return;
	if(segments->empty() || seg_bbox->empty())return;

	if(crop_isEnable){
		/*
		for(int z = 0; z < imageZ; z++){
			for(int y = 0; y < imageH; y++){
				for(int x = 0; x < imageW; x++){
					
					if(segdata[z*imageH*imageW + y*imageW + x] >= 0)dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_LOWER_VAL;
					else dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;

					if(crop_useHmap && !hmapEmpty){
						if(z < hmap[y*imageW + x] + crop_upper || z > hmap[y*imageW + x] + crop_lower)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
					}
					else{
						if(z < crop_upper || z > crop_lower)
							dstdata[z*imageH*imageW + y*imageW + x] = BINALIZE_UPPER_VAL;
					}

				}
			}
		}
		*/
		vector<int>::iterator ite = selected_segment->begin();
		while(ite != selected_segment->end()){
			bool inside = false;
			for(int i = 0; i < (*segments)[*ite].size(); i++){
				int x = (*segments)[*ite][i].x;
				int y = (*segments)[*ite][i].y;
				int z = (*segments)[*ite][i].z;
				if(crop_useHmap && !hmapEmpty){
					if(z >= hmap[y*imageW + x] + crop_upper && z <= hmap[y*imageW + x] + crop_lower){
						inside = true;
						break;
					}
				}
				else{
					if(z >= crop_upper && z <= crop_lower){
						inside = true;
						break;
					}
				}
			}
			if(inside)ite++;
			else ite = selected_segment->erase(ite);
		}
	}

	//binarySegmentationLow(dstdata, 0.1f, SEGMENT_CONNECT6, 0, true);
}

void Filter3D::joinSelectedSegments()
{
	if(isEmpty)return;
	if(selected_segment == NULL)return;
	if(selected_segment->size() <= 1)return;

	std::vector<cv::Point3i> joined;
	int newId = segments->size();
	int xmin = imageW, xmax = 0, ymin = imageH, ymax = 0, zmin = imageZ, zmax = 0;

	for(int i = 0; i < selected_segment->size(); i++){
		for(int j = 0; j < (*segments)[(*selected_segment)[i]].size(); j++){
			Point3i spt = (*segments)[(*selected_segment)[i]][j];
			joined.push_back(spt);
			segdata[spt.z*imageH*imageW + spt.y*imageW + spt.x] = newId;
		}
		std::vector<cv::Point3i>().swap((*segments)[(*selected_segment)[i]]);

		Box3D sbbox = (*seg_bbox)[(*selected_segment)[i]];
		if(xmin > sbbox.x)xmin = sbbox.x;
		if(ymin > sbbox.y)ymin = sbbox.y;
		if(zmin > sbbox.z)zmin = sbbox.z;
		if(xmax < sbbox.x + sbbox.width  - 1)xmax = sbbox.x + sbbox.width  - 1;
		if(ymax < sbbox.y + sbbox.height - 1)ymax = sbbox.y + sbbox.height - 1;
		if(zmax < sbbox.z + sbbox.depth  - 1)zmax = sbbox.z + sbbox.depth  - 1;
	}
	
	Box3D newbbox;
	newbbox.x = xmin;
	newbbox.y = ymin;
	newbbox.z = zmin;
	newbbox.width  = xmax - xmin + 1;
	newbbox.height = ymax - ymin + 1;
	newbbox.depth  = zmax - zmin + 1;

	if(newbbox.width <= 0 || newbbox.height <= 0 || newbbox.depth <= 0){
		for(int i = 0; i < joined.size(); i++)segdata[joined[i].z*imageH*imageW + joined[i].y*imageW + joined[i].x] = -1;
		return;
	}

	segments->push_back(joined);
	seg_bbox->push_back(newbbox);
	seg_colors->push_back(rand() % 360);
	std::vector<int>().swap(*selected_segment);
	selected_segment->push_back(newId);
}

void Filter3D::separateSelectedSegments()
{
	if(isEmpty)return;
	if(selected_segment == NULL)return;
	if(selected_segment->size() <= 0)return;

	float *bound = new float[imageW*imageH*imageZ];
	float *d_Bound;
	checkCudaErrors( cudaMalloc((void **)(&d_Bound) , imageZ * imageW * imageH * sizeof(float)) );
	
	int *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );

	fillFloatGPU(d_Bound, imageW, imageH, imageZ, BINALIZE_UPPER_VAL);
	checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
	setMinValIntGPU(d_Seg, imageW, imageH, imageZ, SEGMENT_BLANK);
	checkCudaErrors( cudaDeviceSynchronize() );
	for(int i = 0; i < selected_segment->size(); i++){
		Box3D bbox = (*seg_bbox)[(*selected_segment)[i]];
		copySingleSegmentROIGPU(d_Bound, d_Seg, (*selected_segment)[i], BINALIZE_LOWER_VAL, imageW, imageH, imageZ, bbox.x, bbox.y, bbox.z, bbox.width, bbox.height, bbox.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
		replacementIntROIGPU(d_Seg, (*selected_segment)[i], SEGMENT_BLANK, imageW, imageH, imageZ, bbox.x, bbox.y, bbox.z, bbox.width, bbox.height, bbox.depth);
	}
	checkCudaErrors( cudaMemcpy(bound, d_Bound, imageW * imageH * imageZ * sizeof(float), cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	
	int id = segments->size();
	int oldSegNum = segments->size();
	Box3D t_boundingBox;
	for(int i = 0; i < selected_segment->size(); i++){
		for(int j = 0; j < (*segments)[(*selected_segment)[i]].size(); j++){
			Point3i spt = (*segments)[(*selected_segment)[i]][j];
			if(bound[spt.z*imageH*imageW + spt.y*imageW + spt.x] <= BINALIZE_LOWER_VAL && segdata[spt.z*imageH*imageW + spt.y*imageW + spt.x] == SEGMENT_BLANK){
				binarySegmentatorLow(bound, t_boundingBox, BINALIZE_LOWER_VAL, SEGMENT_CONNECT6, spt.x, spt.y, spt.z, id);
				seg_bbox->push_back(t_boundingBox);
				seg_colors->push_back(rand() % 360);
				id++;
			}
		}
		std::vector<cv::Point3i>().swap((*segments)[(*selected_segment)[i]]);
	}

	std::vector<int>().swap(*selected_segment);
	for(int i = oldSegNum; i < segments->size(); i++)selected_segment->push_back(i);

	delete [] bound;
	checkCudaErrors( cudaFree(d_Bound) );
	checkCudaErrors( cudaFree(d_Seg) );
}

void Filter3D::dilateSphereSelectedSegments(int radius)
{
	if(isEmpty)return;
	if(selected_segment == NULL)return;
	if(selected_segment->size() <= 0)return;

	int diameter = radius*2 + 1;

	int *segroi = new int[imageW*imageH*imageZ];
	int *d_Seg, *d_SegOut;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	checkCudaErrors( cudaMalloc((void **)(&d_SegOut) , imageZ * imageW * imageH * sizeof(int)) );
	int *d_SegROI;
	checkCudaErrors( cudaMalloc((void **)(&d_SegROI) , imageZ * imageW * imageH * sizeof(int)) );

	checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_SegOut, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToDevice) );

	for(int i = 0; i < selected_segment->size(); i++){
		if((*segments)[(*selected_segment)[i]].size() <= 0)continue;

		Box3D sbbox = (*seg_bbox)[(*selected_segment)[i]];
		Box3D bb;
		bb.x = (sbbox.x - diameter >= 0) ? sbbox.x - diameter : 0;
		bb.y = (sbbox.y - diameter >= 0) ? sbbox.y - diameter : 0;
		bb.z = (sbbox.z - diameter >= 0) ? sbbox.z - diameter : 0;
		bb.width  = (sbbox.x + (sbbox.width  - 1) + diameter < imageW) ? sbbox.x + (sbbox.width  - 1) + diameter - bb.x + 1 : (imageW - 1) - bb.x + 1;
		bb.height = (sbbox.y + (sbbox.height - 1) + diameter < imageH) ? sbbox.y + (sbbox.height - 1) + diameter - bb.y + 1 : (imageH - 1) - bb.y + 1;
		bb.depth  = (sbbox.z + (sbbox.depth  - 1) + diameter < imageZ) ? sbbox.z + (sbbox.depth  - 1) + diameter - bb.z + 1 : (imageZ - 1) - bb.z + 1;

		cropIntGPU(d_SegROI, d_Seg, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
		initCuda_Suf3D_Int(d_SegROI, bb.width, bb.height, bb.depth);
		
		checkCudaErrors( cudaDeviceSynchronize() );
		if(radius >= 4){
			for(int i = 0; i < radius / 4; i++){
				dilateSegmentsSurfaceSphereGPU(4, bb.width, bb.height, bb.depth);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
			if(radius % 4 > 0){
				dilateSegmentsSurfaceSphereGPU(radius % 4, bb.width, bb.height, bb.depth);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
		}
		else {
			dilateSegmentsSurfaceSphereGPU(radius, bb.width, bb.height, bb.depth);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		
		copyToDeviceMemFromSuf3D_Int(d_SegROI, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
		
		checkCudaErrors( cudaMemcpy(segroi, d_SegROI, bb.width * bb.height * bb.depth * sizeof(int), cudaMemcpyDeviceToHost) );
		std::vector<cv::Point3i>().swap((*segments)[(*selected_segment)[i]]);
		int xmin = imageW, xmax = 0, ymin = imageH, ymax = 0, zmin = imageZ, zmax = 0;
		for(int z = 0; z < bb.depth; z++){
			for(int y = 0; y < bb.height; y++){
				for(int x = 0; x < bb.width; x++){
					if(segroi[z*bb.height*bb.width + y*bb.width + x] == (*selected_segment)[i]){
						(*segments)[(*selected_segment)[i]].push_back(Point3i(bb.x + x, bb.y + y, bb.z + z));
						if(xmin > bb.x + x)xmin = bb.x + x;
						if(ymin > bb.y + y)ymin = bb.y + y;
						if(zmin > bb.z + z)zmin = bb.z + z;
						if(xmax < bb.x + x)xmax = bb.x + x;
						if(ymax < bb.y + y)ymax = bb.y + y;
						if(zmax < bb.z + z)zmax = bb.z + z;
					}
				}
			}
		}

		(*seg_bbox)[(*selected_segment)[i]].x = xmin;
		(*seg_bbox)[(*selected_segment)[i]].y = ymin;
		(*seg_bbox)[(*selected_segment)[i]].z = zmin;
		(*seg_bbox)[(*selected_segment)[i]].width  = xmax - xmin + 1;
		(*seg_bbox)[(*selected_segment)[i]].height = ymax - ymin + 1;
		(*seg_bbox)[(*selected_segment)[i]].depth  = zmax - zmin + 1;

		destructCuda_Suf3D();
		copySingleSegmentIntOffsetGPU(d_SegOut, d_SegROI, (*selected_segment)[i], imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	checkCudaErrors( cudaMemcpy(segdata, d_SegOut, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );

	delete [] segroi;
	checkCudaErrors( cudaFree(d_Seg) );
	checkCudaErrors( cudaFree(d_SegOut) );
	checkCudaErrors( cudaFree(d_SegROI) );
}

void Filter3D::erodeSphereSelectedSegments(int radius, bool clamp)
{
	if(isEmpty)return;
	if(selected_segment == NULL)return;
	if(selected_segment->size() <= 0)return;

	int diameter = radius*2 + 1;

	int *segroi = new int[imageW*imageH*imageZ];
	int *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	int *d_SegROI;
	checkCudaErrors( cudaMalloc((void **)(&d_SegROI) , imageZ * imageW * imageH * sizeof(int)) );

	checkCudaErrors( cudaMemcpy(d_Seg, segdata, imageW * imageH * imageZ * sizeof(int), cudaMemcpyHostToDevice) );
	
	for(int i = 0; i < selected_segment->size(); i++){
		if((*segments)[(*selected_segment)[i]].size() <= 0)continue;

		Box3D sbbox = (*seg_bbox)[(*selected_segment)[i]];
		Box3D bb;
		bb.x = (sbbox.x - diameter >= 0) ? sbbox.x - diameter : 0;
		bb.y = (sbbox.y - diameter >= 0) ? sbbox.y - diameter : 0;
		bb.z = (sbbox.z - diameter >= 0) ? sbbox.z - diameter : 0;
		bb.width  = (sbbox.x + (sbbox.width  - 1) + diameter < imageW) ? sbbox.x + (sbbox.width  - 1) + diameter - bb.x + 1 : (imageW - 1) - bb.x + 1;
		bb.height = (sbbox.y + (sbbox.height - 1) + diameter < imageH) ? sbbox.y + (sbbox.height - 1) + diameter - bb.y + 1 : (imageH - 1) - bb.y + 1;
		bb.depth  = (sbbox.z + (sbbox.depth  - 1) + diameter < imageZ) ? sbbox.z + (sbbox.depth  - 1) + diameter - bb.z + 1 : (imageZ - 1) - bb.z + 1;

		//Box3D bb = (*seg_bbox)[(*selected_segment)[i]];
		cropIntGPU(d_SegROI, d_Seg, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
		replacementIntROIGPU(d_Seg, (*selected_segment)[i], SEGMENT_BLANK, imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
		initCuda_Suf3D_Int(d_SegROI, bb.width, bb.height, bb.depth);
		
		checkCudaErrors( cudaDeviceSynchronize() );
		if(radius >= 4){
			for(int i = 0; i < radius / 4; i++){
				if(clamp) erodeSegmentsSurfaceSphereGPU(4, bb.width, bb.height, bb.depth);
				else erodeSegmentsSurfaceSphereGPU_v2(4, bb.width, bb.height, bb.depth);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
			if(radius % 4 > 0){
				if(clamp) erodeSegmentsSurfaceSphereGPU(radius % 4, bb.width, bb.height, bb.depth);
				else erodeSegmentsSurfaceSphereGPU_v2(radius % 4, bb.width, bb.height, bb.depth);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
		}
		else {
			if(clamp) erodeSegmentsSurfaceSphereGPU(radius, bb.width, bb.height, bb.depth);
			else erodeSegmentsSurfaceSphereGPU_v2(radius, bb.width, bb.height, bb.depth);
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		
		copyToDeviceMemFromSuf3D_Int(d_SegROI, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
		
		checkCudaErrors( cudaMemcpy(segroi, d_SegROI, bb.width * bb.height * bb.depth * sizeof(int), cudaMemcpyDeviceToHost) );
		std::vector<cv::Point3i>().swap((*segments)[(*selected_segment)[i]]);
		int xmin = imageW, xmax = 0, ymin = imageH, ymax = 0, zmin = imageZ, zmax = 0;
		for(int z = 0; z < bb.depth; z++){
			for(int y = 0; y < bb.height; y++){
				for(int x = 0; x < bb.width; x++){
					if(segroi[z*bb.height*bb.width + y*bb.width + x] == (*selected_segment)[i]){
						(*segments)[(*selected_segment)[i]].push_back(Point3i(bb.x + x, bb.y + y, bb.z + z));
						if(xmin > bb.x + x)xmin = bb.x + x;
						if(ymin > bb.y + y)ymin = bb.y + y;
						if(zmin > bb.z + z)zmin = bb.z + z;
						if(xmax < bb.x + x)xmax = bb.x + x;
						if(ymax < bb.y + y)ymax = bb.y + y;
						if(zmax < bb.z + z)zmax = bb.z + z;
					}
				}
			}
		}

		if((*segments)[(*selected_segment)[i]].size() > 0){
			(*seg_bbox)[(*selected_segment)[i]].x = xmin;
			(*seg_bbox)[(*selected_segment)[i]].y = ymin;
			(*seg_bbox)[(*selected_segment)[i]].z = zmin;
			(*seg_bbox)[(*selected_segment)[i]].width  = xmax - xmin + 1;
			(*seg_bbox)[(*selected_segment)[i]].height = ymax - ymin + 1;
			(*seg_bbox)[(*selected_segment)[i]].depth  = zmax - zmin + 1;
		}

		destructCuda_Suf3D();
		copySingleSegmentIntOffsetGPU(d_Seg, d_SegROI, (*selected_segment)[i], imageW, imageH, imageZ, bb.x, bb.y, bb.z, bb.width, bb.height, bb.depth);
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	checkCudaErrors( cudaMemcpy(segdata, d_Seg, imageW * imageH * imageZ * sizeof(int), cudaMemcpyDeviceToHost) );
	
	vector<int>::iterator ss_ite = selected_segment->begin();
	while(ss_ite != selected_segment->end()){
		if((*segments)[*ss_ite].size() <= 0)ss_ite = selected_segment->erase(ss_ite);
		else ss_ite++;
	}
	
	delete [] segroi;
	checkCudaErrors( cudaFree(d_Seg) );
	checkCudaErrors( cudaFree(d_SegROI) );
}

void Filter3D::watershed3D(float stride, int seg_minVol)
{
	if(isEmpty)return;
	if(imageW <= 0 || imageH <= 0 || imageZ <= 0)return;
	if(segments == NULL || seg_bbox == NULL || segdata == NULL)return;

	cout << stride << endl;
	if(stride <= 0.0f)return;
	
	vector<char> seg_isSelected(segments->size(), 0);
	for(unsigned int i = 0; i < selected_segment->size(); i++)seg_isSelected[(*selected_segment)[i]] = 1;
	for(int i = 0; i < segments->size(); i++){
		if( (*segments)[i].size() < seg_minVol || seg_isSelected[i] == 0){
			for(int j = 0; j < (*segments)[i].size(); j++){
				int x = (*segments)[i][j].x;
				int y = (*segments)[i][j].y;
				int z = (*segments)[i][j].z;
				segdata[z*imageH*imageW + y*imageW + x] = SEGMENT_BLANK;
			}
		}
		(*segments)[i].clear();
	}
	
	memcpy(bufdata, imgdata, imageW*imageH*imageZ*sizeof(float));
	if(crop_isEnable){
		for(int z = 0; z < imageZ; z++){
			for(int y = 0; y < imageH; y++){
				for(int x = 0; x < imageW; x++){
					if(y < crop_border_xy_thickness || y >= imageH - crop_border_xy_thickness || x < crop_border_xy_thickness || x >= imageW - crop_border_xy_thickness)
						bufdata[z*imageH*imageW + y*imageW + x] = FLT_MAX;
					else if(crop_useHmap && !hmapEmpty){
						if(z < hmap[y*imageW + x] + crop_upper || z > hmap[y*imageW + x] + crop_lower)
							bufdata[z*imageH*imageW + y*imageW + x] = FLT_MAX;
					}
					else{
						if(z < crop_upper || z > crop_lower)
							bufdata[z*imageH*imageW + y*imageW + x] = FLT_MAX;
					}
				}
			}
		}
	}

	initCuda_Watershed(segdata, bufdata, imageW, imageH, imageZ);
	int *d_Seg_ref, *d_Seg;
	checkCudaErrors( cudaMalloc((void **)(&d_Seg_ref) , imageZ * imageW * imageH * sizeof(int)) );
	checkCudaErrors( cudaMalloc((void **)(&d_Seg) , imageZ * imageW * imageH * sizeof(int)) );
	
	int iteration = (int)ceil(1.0f / stride);
	vector<int> before_ites;
	int base_ite = 0;
	for(int i = 0; i < iteration; i++){
		float th = (i + 1) * stride;
		int inner_ite = 0;
		bool loop_end = false;
		do{
			if(inner_ite >= base_ite){
				copyToDeviceMemFromSuf3D_Int(d_Seg_ref, imageW, imageH, imageZ);
				checkCudaErrors( cudaDeviceSynchronize() );
			}
			
			watershed3dGPU(th, imageW, imageH, imageZ);
			checkCudaErrors( cudaDeviceSynchronize() );
			
			if(inner_ite >= base_ite){
				copyToDeviceMemFromSuf3D_Int(d_Seg, imageW, imageH, imageZ);
				checkCudaErrors( cudaDeviceSynchronize() );
				notEqualIntGPU(d_Buffer, d_Seg_ref, d_Seg, imageW, imageH, imageZ);
				checkCudaErrors( cudaDeviceSynchronize() );
				if(abs(cublasSasum(imageW * imageH * imageZ, d_Buffer, 1)) <= 0.5f)loop_end = true;
			}
			if(i == iteration - 1)loop_end = true;
			inner_ite++;
		}while(!loop_end);
		
		before_ites.push_back(inner_ite);
		if(i >= 10){
			int ite_sum10 = 0;
			float ite_mean10;
			float ite_v10 = 0.0f;
			float ite_stdev10;
			for(int j = 0; j < 10; j++)ite_sum10 += before_ites[i - j];
			ite_mean10 = ite_sum10 / 10.0f;
			for(int j = 0; j < 10; j++)ite_v10 += (before_ites[i - j] - ite_mean10) * (before_ites[i - j] - ite_mean10);
			ite_v10 /= 10.0f;
			ite_stdev10 = sqrt(ite_v10);

			int n = 10;
			for(int j = 0; j < 10; j++){
				float t = (before_ites[i - j] - ite_mean10) / ite_stdev10;
				if(t > 2){
					ite_sum10 -= before_ites[i - j];
					n--;
				}
			}
			if(n > 1)base_ite = ite_sum10 / n - 2;
			else base_ite = 0;
		}
		cout << th << " : " << inner_ite << " : " << base_ite << endl;
	}
	copyToHostMemFromSuf3D_Int(segdata, imageW, imageH, imageZ);

	destructCuda_Watershed();
	checkCudaErrors( cudaFree(d_Seg) );
	checkCudaErrors( cudaFree(d_Seg_ref) );

	for(int z = 0; z < imageZ; z++){
		for(int y = 0; y < imageH; y++){
			for(int x = 0; x < imageW; x++){
				if(segdata[z*imageH*imageW + y*imageW + x] >= 0)(*segments)[segdata[z*imageH*imageW + y*imageW + x]].push_back(Point3i(x, y, z));
			}
		}
	}
}

void Filter3D::saveBackupSegmentsData()
{
	copy1DVector(seg_colors_bk, seg_colors);
	copy1DVector(seg_bbox_bk, seg_bbox);
	copy2DVector(segments_bk, segments);
	memcpy(segdata_bk, segdata, imageW*imageH*imageZ*sizeof(int));
}

void Filter3D::loadBackupSegmentsData()
{
	if(seg_colors_bk->size() > 0 && seg_bbox_bk->size() > 0 && segments_bk->size() > 0 && seg_colors_bk->size() == seg_bbox_bk->size() && seg_bbox_bk->size() == segments_bk->size()){
		copy1DVector(seg_colors, seg_colors_bk);
		copy1DVector(seg_bbox, seg_bbox_bk);
		copy2DVector(segments, segments_bk);
		memcpy(segdata, segdata_bk, imageW*imageH*imageZ*sizeof(int));
		deselectSegment();
	}
}

void Filter3D::setCroppingParams(bool isEnable, bool useHmap, int upper, int lower, int border)
{
	crop_isEnable = isEnable;
	crop_useHmap = useHmap;
	crop_upper = upper;
	crop_lower = lower;
	crop_border_xy_thickness = border;
	//cout << crop_isEnable << " : " << crop_useHmap << " : " << crop_upper << " : " << crop_lower << endl;
}