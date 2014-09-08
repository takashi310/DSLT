// TestCVclass.h
#pragma unmanaged
#include "filter3d.h"
#pragma managed

#ifndef TEST_CV_CLASS_H
#define TEST_CV_CLASS_H

namespace TestCVclass {

	public ref class Class1
	{
		// TODO: このクラスの、ユーザーのメソッドをここに追加してください。
	internal:

		static float C_factor = 0.002f;
		
		TestGPUclass::Filter3D *flt3d;
		bool isEmpty;
		int currentX;
		int currentY;
		int currentZ;
		float bc_max;
		float bc_min;
		int selection_copy_src;
		int selection_display_mode;
		float hmapoffset;
		float hmap_depth;
		float hmap_range;
		bool hmap_isVisible;
		bool hmap_use;
		int proj_mode;
		bool dc_isEnable;
		float dmap_range;
		float dmap_coefficient;
		float dmap_order;
		float proj_th;
		bool zbc_isEnable;
		int zbc_channel;
		float zbc_coefficient;
		float zbc_order;
		bool seg_show;
		int seg_minVol;
		System::Drawing::Bitmap^ img_bmp;
	public:
		
		static int C1_FLT_SPHERE = 1;
		static int C1_FLT_CUBE = 0;
		static int C1_SEG_CONNECT6 = SEGMENT_CONNECT6;
		static int C1_SEG_CONNECT18 = SEGMENT_CONNECT18;
		static int C1_SEG_CONNECT26 = SEGMENT_CONNECT26;
		static int C1_SC_NONE = 0;
		static int C1_SC_AREA_AVE = 1;
		static int C1_SC_LANCZOS2 = 2;
		static int C1_SC_LANCZOS3 = 3;

		Class1();
		~Class1();
		!Class1();
		bool imageRead(System::String ^str);
		bool set3DImage_MultiTIFF(System::String ^filename, int channel, int scalingType);
		void setScalingType(int type);
		void setChannel(int ch);
		//System::Drawing::Bitmap^ getImageBitmap(void);
		bool empty();
		void setX(int value);
		void setY(int value);
		void setZ(int value);
		void setBCmax(float value);
		void setBCmin(float value);
		void setHmapVisibility(bool isVisible);
		void setHmapOffset(float value);
		void setHmapActivity(bool value);
		void setProjectionMode(int value);
		void setProjectionThreshold(float value);
		void setHmapProjDepth(float value);
		void setHmapProjRange(float value);
		void setDepthCodeEnable(bool isEnable);
		void setDmapRange(float value);
		void setDmapCoefficient(float value);
		void setDmapOrder(float value);
		void setZBCEnable(bool isEnable);
		void setZBCparams(int channel, float coef, float order);
		void setSegVisibility(bool isVisible);
		void setSegMinVol(int value);
		int getChannelNum();
		void applyChanges();
		void getImageSize([System::Runtime::InteropServices::Out] int %imgX,
						  [System::Runtime::InteropServices::Out] int %imgY,
						  [System::Runtime::InteropServices::Out] int %imgZ);
		unsigned long getImageDataArrayXY([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
										  [System::Runtime::InteropServices::Out] int %bytesperpixel,
										  [System::Runtime::InteropServices::Out] long %width,
										  [System::Runtime::InteropServices::Out] long %height);
		unsigned long getImageDataArrayYZ([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
										  [System::Runtime::InteropServices::Out] int %bytesperpixel,
										  [System::Runtime::InteropServices::Out] long %width,
										  [System::Runtime::InteropServices::Out] long %height);
		unsigned long getImageDataArrayZX([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
										  [System::Runtime::InteropServices::Out] int %bytesperpixel,
										  [System::Runtime::InteropServices::Out] long %width,
										  [System::Runtime::InteropServices::Out] long %height);
		unsigned long getDstImageDataArrayXY([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
											 [System::Runtime::InteropServices::Out] int %bytesperpixel,
											 [System::Runtime::InteropServices::Out] long %width,
											 [System::Runtime::InteropServices::Out] long %height);
		unsigned long getDstImageDataArrayYZ([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
											 [System::Runtime::InteropServices::Out] int %bytesperpixel,
											 [System::Runtime::InteropServices::Out] long %width,
											 [System::Runtime::InteropServices::Out] long %height);
		unsigned long getDstImageDataArrayZX([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
											 [System::Runtime::InteropServices::Out] int %bytesperpixel,
											 [System::Runtime::InteropServices::Out] long %width,
											 [System::Runtime::InteropServices::Out] long %height);
		unsigned long getHeightMapDataArrayXY([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
											  [System::Runtime::InteropServices::Out] int %bytesperpixel,
											  [System::Runtime::InteropServices::Out] long %width,
											  [System::Runtime::InteropServices::Out] long %height);
		unsigned long getXPlotGraph([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
									[System::Runtime::InteropServices::Out] int %bytesperpixel,
									[System::Runtime::InteropServices::Out] long %width,
									[System::Runtime::InteropServices::Out] long %height);
		unsigned long getYPlotGraph([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
									[System::Runtime::InteropServices::Out] int %bytesperpixel,
									[System::Runtime::InteropServices::Out] long %width,
									[System::Runtime::InteropServices::Out] long %height);
		unsigned long getZPlotGraph([System::Runtime::InteropServices::Out] System::IntPtr %ptr,
									[System::Runtime::InteropServices::Out] int %bytesperpixel,
									[System::Runtime::InteropServices::Out] long %width,
									[System::Runtime::InteropServices::Out] long %height);
		void AdaptiveThreshold3DLineKernels(int blocksize, int angle_d, float factor_C_Z, float constC, int kernelType);
		void segmentation_SobelLikeADTH(int blocksize, int angle_d, float factor_C_Z, float minC, float maxC, float interval, int kernelType, int min_segVol, int minInvalidStructureArea, int closing);
		void hMinimaTransform3D(float h, int chk_interval, bool copyToHostMemory);
		void segmentation_hMinimaTransform(float minh, float maxh, float interval, int min_segVol, int minInvalidStructureArea, int closing);
		void segmentation_FloodFill(float th_val, int connectType, int minSize);
		void segmentation_FloodFill(float th_val, int connectType, int minSize, int noise_th, int wall_th, int minInvalidStructureVol, bool saveValidSegments, bool saveInvalidSegments);
		void AdaptiveThreshold3D(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void AdaptiveThreshold2D(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void threshold3D(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void threshold2D(int blocksize, float constC, int thresholdType, bool copyToHostMemory);
		void smooth3D(int blocksize, int filterType, bool copyToHostMemory);
		void dilation3D(int radius, int filterShape, bool copyToHostMemory);
		void erosion3D(int radius, int filterShape, bool copyToHostMemory);
		void generateHeightMap(int blocksize_xy, int blocksize_z, float th, int thresholdType, int smoothLv);
		void inactivateHeightMap();
		bool saveDst3DImage(System::String ^filename);
		bool saveSrc2DImage(System::String ^filename);
		bool saveDst2DImage(System::String ^filename);
		bool saveHeightMap(System::String ^filename);
		bool readHeightMap(System::String ^filename);
		void generateDepthMap();
		void watershed3D(float stride);
		/*
		void drawSelectionZX(int brush_size, int x, int y, int z, int mode);
		void setCopySelectionSliceSrcZX(int slice_id);
		void copySelectionSliceZX(int src, int dst);
		void maskDstCall(bool inv);
		*/
		void calcHmapAreaCall(int reso);
		void applyBlightnessContrast();
		void selectSegment(int x, int y, int z);
		void selectAllSegments();
		void deselectSegment();
		void selectSegments_TH(float th);
		void deselectSegments_TH(float th);
		void cropSegments();
		void joinSelectedSegments();
		void separateSelectedSegments();
		void dilateSphereSelectedSegments(int radius);
		void erodeSphereSelectedSegments(int radius, bool clamp);
		void loadBackupSegmentsData();

		void setCroppingParams(bool isEnable, bool useHmap, int upper, int lower, int border);
		
		static void hsv2rgb(float h, float s, float v, unsigned int %r, unsigned int %g, unsigned int %b);
		
	};
}

#endif