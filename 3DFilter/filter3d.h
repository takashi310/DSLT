#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#pragma warning (push)
#pragma warning( disable : 4819 )
#pragma warning (pop)

#ifndef FILTER3D_H
#define FILTER3D_H

#ifdef DLL_FILTER3D
#define DECLSPEC_DLLPORT	__declspec(dllexport)
#else
#define DECLSPEC_DLLPORT	__declspec(dllimport)
#endif

#define FILTER_GAUSSIAN 0
#define FILTER_MEAN 1
#define C1_NORMAL_PROJECTION 0
#define C1_Z_PROJECTION 1
#define C1_SELECTION_VISIBLE 0
#define C1_SELECTION_HIDE 1
#define C1_SELECTION_CROP 2
#define FLT3D_EDITLOG_NUM 20
#define GRAPH_DATA_AREA_HEIGHT 200
#define GRAPH_DATA_AREA_COORD_X 40
#define GRAPH_DATA_AREA_COORD_Y 10
#define GRAPH_TEXT_AREA_WIDTH 50
#define GRAPH_TEXT_AREA_HEIGHT 50
#define SEGMENT_CONNECT6 0
#define SEGMENT_CONNECT18 1
#define SEGMENT_CONNECT26 2
#define SEGMENT_NOTSAVED -4
#define SEGMENT_INVALID -3
#define SEGMENT_TOOSMALL -2
#define SEGMENT_BLANK -1
#define SC_NONE 0
#define SC_AREA_AVE 1
#define SC_LANCZOS2 2
#define SC_LANCZOS3 3

namespace cv {

	template<typename _Tp> class Point3_;
	typedef Point3_<int> Point3i;
	typedef Point3_<float> Point3f;

}

namespace multif {
	class MultiTiffIO;
}

namespace TestGPUclass {

	struct Box3D{
		int x;
		int y;
		int z;
		int width;
		int height;
		int depth;
	};

	class DECLSPEC_DLLPORT Filter3D
	{
	private:
		float *rawdata;
		float *rawdata_resized;
		float *imgdata;
		float *bufdata;
		float *dstdata;
		float *dsltbk;
		float *dsltbk_lati;
		float dsltbk_rad;
		int dsltbk_d;
		int dslt_type;
		//float *selectdata;
		float *hmap;
		float *hmap_normalized;
		float *hmbproj;
		float *dmap;
		float dmax;
		int *segdata;
		int brush_size;
		cv::Point3f *normals;
		float *areas;
		unsigned char *bufXY;
		unsigned char *bufZX;
		unsigned char *bufYZ;
		unsigned int imageW;
		unsigned int imageH;
		unsigned int imageZ;
		unsigned int nSlices;
		unsigned int nChannels;
		unsigned char *graph_tmpX;
		unsigned char *graph_tmpY;
		unsigned char *graph_tmpZ;
		unsigned char *graph;
		unsigned int graphW;
		unsigned int graphH;
		bool isEmpty;
		bool hmapEmpty;
		bool dmapEmpty;
		bool isEnableGPU;
		int zScaling;
		float zScaleFactor;
		unsigned int curCh;

		multif::MultiTiffIO *tifio;
		float *d_Input;
		float *d_Output;
		float *d_Buffer;
		float *d_Tmp;
		float *h_tmp;
		std::vector<float> *seg_colors;
		std::vector<Box3D> *seg_bbox;
		std::vector<Box3D> *invalid_seg_bbox;
		std::vector< std::vector<cv::Point3i> > *segments;
		std::vector< std::vector<cv::Point3i> > *invalid_segments;
		std::vector<int> *selected_segment;

		std::vector<float> *seg_colors_bk;
		std::vector<Box3D> *seg_bbox_bk;
		std::vector< std::vector<cv::Point3i> > *segments_bk;
		int *segdata_bk;
		std::vector<int> *selected_segment_bk;

		bool crop_isEnable;
		bool crop_useHmap;
		int crop_upper;
		int crop_lower;
		int crop_border_xy_thickness;
		
		void clear();
		void binarySegmentatorLow(const float *data, Box3D &bbox, float th_val, int connect_type, int x, int y, int z, int id);
		void saveOrthViewSimple(float *data, const char *fname, int currentX, int currentY, int currentZ, float bc_max, float bc_min);
		void saveOrthViewSimple_Numbered(float *data, const char *fname, int id, int currentX, int currentY, int currentZ, float bc_max, float bc_min);
		void scalingAreaAveZ(float *dst, const float *src, int width, int height, int depth, float factor);
		void scalingLanczosZ(float *dst, const float *src, int width, int height, int depth, float factor, int n);
		void maximumSphereFilter(int radius, float *d_1, float *d_buf, int width, int height, int depth, int roi_x, int roi_y, int roi_z, int roi_w, int roi_h, int roi_d);
		void minimumSphereFilter(int radius, float *d_1, float *d_buf, int width, int height, int depth, int roi_x, int roi_y, int roi_z, int roi_w, int roi_h, int roi_d);
		bool isSelectedSegment(int id);

		double F(double x, double y, cv::Point3f p00, cv::Point3f p10, cv::Point3f p01, cv::Point3f p11);
		double simpe2(double **f, const int m, const int n, double h1, double h2);

	public:
		
		Filter3D();
		~Filter3D();

		bool empty(){return isEmpty;}
		bool isAvilableGPU(){return isEnableGPU;}
		bool hmap_empty(){return hmapEmpty;}
		bool dmap_empty(){return dmapEmpty;}
		unsigned int getWidth(){return imageW;}
		unsigned int getHeight(){return imageH;}
		unsigned int getDepth(){return imageZ;}
		unsigned char *getbufXY(){return bufXY;}
		unsigned char *getbufZX(){return bufZX;}
		unsigned char *getbufYZ(){return bufYZ;}
		unsigned char *getgraph(){return graph;}
		unsigned int getgraphW(){return graphW;}
		unsigned int getgraphH(){return graphH;}
		float *getImgData(){return imgdata;}
		float *getDstData(){return dstdata;}

		bool savebufXY(const char *fname);
		bool savebufYZ(const char *fname);
		bool savebufZX(const char *fname);

		void setDevice();
		static void getGaussianFilter1D(float filter[], int ksize);
		static void getMeanFilter1D(float filter[], int ksize);
		bool set3DImage_MultiTIFF(const char filename[], int channel = 0, int z_scaling = SC_NONE);
		void switchZScaling(int type);
		void switchChannel(int channel);
		int getChNum();
		void applyChanges();
		bool saveDst3DImage(const char filename[]);
		bool saveHeightMap(const char filename[]);
		bool readHeightMap(const char filename[]);
		//bool saveSrc2DImage(const char filename[]);
		//bool saveDst2DImage(const char filename[]);
		
		void AdaptiveThreshold3D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void threshold3D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void AdaptiveThreshold2D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void threshold2D_GPU(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void smoothing3D_GPU(int blocksize, int filterType, bool copyToHostMemory);
		void smoothing2D_GPU(int blocksize, int filterType, bool copyToHostMemory);
		void dilationCubic(int radius, bool copyToHostMemory);
		void erosionCubic(int radius, bool copyToHostMemory);
		void dilationSpherical(int radius, bool copyToHostMemory);
		void erosionSpherical(int radius, bool copyToHostMemory);
		void closingSpherical(int radius, bool copyToHostMemory);
		void openingSpherical(int radius, bool copyToHostMemory);
		void DoG(int blocksize, bool copyToHostMemory);
		
		void hMinimaTransform3D_GPU(float h, int chk_interval, bool copyToHostMemory);
		
		void AdaptiveThreshold3D_CPU(int blocksize, float constC, int thresholdType);
		void threshold3D_CPU(int blocksize, float constC, int thresholdType);
		void AdaptiveThreshold2D_CPU(int blocksize, float constC, int thresholdType);
		void threshold2D_CPU(int blocksize, float constC, int thresholdType);
		void allAngleLinearAdaptiveThresholdGPU(int blocksize, int angle_d, float constC_XY, float constC_Z, int thresholdType);
		void allAngleLinearConvolutionMinGPU(float *output, float *input, int blocksize, int angle_d, int FilterType, float *argmin_lati, float *argmin_longi);
		void allAngleLinearConvolutionMinGPU_geodesic(float *output, float *input, int radius, int level, int FilterType, float *argmin_lati, float *argmin_longi);
		void allAngleLinearConvolutionMinGPU_geodesic_v2(float *output, float *input, int radius, int level, int FilterType, float *argmin_lati, float *argmin_longi);
		void allAngleLinearConvolutionMinGPU_geodesic_v3(float *output, float *input, int radius, int level, int FilterType, float *argmin_lati, float *argmin_longi);

		void savitzky_golay_Z(const int npast, const int nfuture,const int deriv, const int poly_order);

		void binarySegmentationLow(const float *data, float th_val, int connect_type, int min_size, bool clearOldResult, const int *seed = NULL);
		void binarySegmentationLow(const float *data, float th_val, int connect_type, int min_size, bool clearOldResult, int noise_th, int wall_th, int minInvalidStructureVol, bool saveVaildSegments, bool saveInvalidSegments);
		void resetSegmentColors();
		void segmentsValidation(float *data, int noise_th, int wall_th, int minInvalidStructureVol);
		void fillingHoles(float *data, int wall_th, int hole_rad, int minInvalidStructureVol); 
		void fillingHolesSimple(float *data, int hole_rad); 
		int estimateWallThickness(int st_size, float th);
		int estimateWallThickness_v2(int st_size);
		int estimateWallThickness_v2(int st_size, int currentX, int currentY, int currentZ, float bc_max, float bc_min);
		void segmentation(int blocksize, int angle_d, float factor_C_Z, float minC, float maxC, float interval, int kernelType, int min_segVol, int minInvalidStructureArea, int hole_size);
		void segmentation_SobelLikeADTH(int blocksize, int angle_d, float factor_C_Z, float minC, float maxC, float interval, int kernelType, int minInvalidStructureArea, int closing);
		void segmentation_hMinimaTransform(float minh, float maxh, float interval, int min_segVol, int minInvalidStructureArea, int closing);
		void adjustAirspacesBrightness(int ar_dataCh, int blocksize, int angle_d, float ar_minC, float ar_maxC, int iteration, float targetC, float factor_C_Z, int kernelType);

		/*Debug—p*/
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
										  float bc_min);

		void generateHeightMapGPU(int blocksize_xy, int blocksize_z, float th, int thresholdType, int smoothLv);
		void generateHeightMapCPU(int blocksize_xy, int blocksize_z, float th, int thresholdType);
		void clearHeightMap();
		void heightMapBasedProjection(float *data, float offset, float depth, int range, float th);
		void heightMapSimpleProjection(float *data, float offset, float depth, int range, bool isBinarize, float th, bool dmap_isEnable, int d_range);
		void simpleProjection(float *data, float bc_max, float bc_min, int slice_id, int range, bool seg_show = false, int seg_minVol = 0, bool dmap_isEnable = false, float d_coefficient = 1.0f, float d_order = 1.0f, bool zbc_isEnable = false, int zbc_channel = 0, float zbc_coefficient = 1.0f, float zbc_order = 1.0f);
		void setBufferYZ(float *data, int currentX, float bc_max, float bc_min, bool seg_show = false, int seg_minVol = 0, bool hmap_isVisible = false, int hmap_offset = 0.0f, int hmap_range = 0.0f, bool dmap_isEnable = false, float d_coefficient = 1.0f, float d_order = 1.0f, bool zbc_isEnable = false, int zbc_channel = 0, float zbc_coefficient = 1.0f, float zbc_order = 1.0f);
		void setBufferZX(float *data, int currentY, float bc_max, float bc_min, bool seg_show = false, int seg_minVol = 0, bool hmap_isVisible = false, int hmap_offset = 0.0f, int hmap_range = 0.0f, bool dmap_isEnable = false, float d_coefficient = 1.0f, float d_order = 1.0f, bool zbc_isEnable = false, int zbc_channel = 0, float zbc_coefficient = 1.0f, float zbc_order = 1.0f);
		void setHeightMapToBufferXY();
		void generateGraphTemplate();
		void setXPlotGraph(int x, int y, int z, float bc_max = 1.0f, float bc_min = 0.0f, bool depthBC = false, float d_coefficient = 0.0f, float d_order = 0.0f);
		void setYPlotGraph(int x, int y, int z, float bc_max = 1.0f, float bc_min = 0.0f, bool depthBC = false, float d_coefficient = 0.0f, float d_order = 0.0f);
		void setZPlotGraph(int x, int y, int z, float bc_max = 1.0f, float bc_min = 0.0f, bool depthBC = false, float d_coefficient = 0.0f, float d_order = 0.0f);
		void applyBC(float bc_max, float bc_min, bool dmap_isEnable, float d_coefficient, float d_order);

		void generateDepthMap();
		void generateDepthMapGPU();
		void depthCode(float *data, float offset, int range);
		static void hsv2rgb(float h, float s, float v, unsigned int &r, unsigned int &g, unsigned int &b);
		static float range_adjustment(float value, float max, float min);
		void maskDst(bool inv);

		//void drawSelection(int brush_size, int cx, int cy, int cz, int mode);
		//void copySelectionSliceZX(int src, int dst);

		void calcHmapArea(int reso);
		
		void saveRGBimageSeries(const char *fname, float bc_max, float bc_min);

		void drawCircle(float longi, float lati, int radius);
		void drawLineSegments(int radius, int level);
		void drawIcoSphere(int radius, int level);

		void selectSegment(int x, int y, int z);
		void selectAllSegments();
		void deselectSegment();
		void autoSelectSegmentsTH(float th);
		void autoDeselectSegmentsTH(float th);
		void cropSegments();
		void joinSelectedSegments();
		void separateSelectedSegments();
		void dilateSphereSelectedSegments(int radius);
		void erodeSphereSelectedSegments(int radius, bool clamp);
		void watershed3D(float stride, int seg_minVol);

		void saveBackupSegmentsData();
		void loadBackupSegmentsData();

		void setCroppingParams(bool isEnable, bool useHmap, int upper, int lower, int border);
	};
}

#endif