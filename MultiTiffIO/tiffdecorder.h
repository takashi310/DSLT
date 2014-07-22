#ifndef TIFF_HNDL_H
#define TIFF_HNDL_H

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>
#include <errno.h>
#include <memory>

#pragma warning(push)
#pragma warning(disable : 4290)

#ifdef DLL_TIFF_HNDL
#define DECLSPEC_DLLPORT	__declspec(dllexport)
#else
#define DECLSPEC_DLLPORT	__declspec(dllimport)
#endif

namespace tiffhndl{

	typedef unsigned char byte;

	/** This class consists of public fields that describe an image file. */
	class DECLSPEC_DLLPORT FileInfo{
	private:

		void getType(char *tname);
		void clear();
		void init_vals();
		void init();
		void copy_body(const FileInfo& fi);

		FileInfo& copy(const FileInfo& fi);

	public:
		/** 8-bit unsigned integer (0-255). */
		static const int GRAY8 = 0;

		/**     16-bit signed integer (-32768-32767). Imported signed images
		are converted to unsigned by adding 32768. */
		static const int GRAY16_SIGNED = 1;

		/** 16-bit unsigned integer (0-65535). */
		static const int GRAY16_UNSIGNED = 2;

		/**     32-bit signed integer. Imported 32-bit integer images are
		converted to floating-point. */
		static const int GRAY32_INT = 3;

		/** 32-bit floating-point. */
		static const int GRAY32_FLOAT = 4;

		/** 8-bit unsigned integer with color lookup table. */
		static const int COLOR8 = 5;

		/** 24-bit interleaved RGB. Import/export only. */
		static const int RGB = 6;        

		/** 24-bit planer RGB. Import only. */
		static const int RGB_PLANAR = 7;

		/** 1-bit black and white. Import only. */
		static const int BITMAP = 8;

		/** 32-bit interleaved ARGB. Import only. */
		static const int ARGB = 9;

		/** 24-bit interleaved BGR. Import only. */
		static const int BGR = 10;

		/**     32-bit unsigned integer. Imported 32-bit integer images are
		converted to floating-point. */
		static const int GRAY32_UNSIGNED = 11;

		/** 48-bit interleaved RGB. */
		static const int RGB48 = 12;     

		/** 12-bit unsigned integer (0-4095). Import only. */
		static const int GRAY12_UNSIGNED = 13;   

		/** 24-bit unsigned integer. Import only. */
		static const int GRAY24_UNSIGNED = 14;   

		/** 32-bit interleaved BARG (MCID). Import only. */
		static const int BARG  = 15;     

		/** 64-bit floating-point. Import only.*/
		static const int GRAY64_FLOAT  = 16;     

		/** 48-bit planar RGB. Import only. */
		static const int RGB48_PLANAR = 17;      

		/** 32-bit interleaved ABGR. Import only. */
		static const int ABGR = 18;

		static const int CMYK = 19;

		// File formats
		static const int UNKNOWN = 0;
		static const int RAW = 1;
		static const int TIFF = 2;
		static const int GIF_OR_JPG = 3;
		static const int FITS = 4;
		static const int BMP = 5;
		static const int DICOM = 6;
		static const int ZIP_ARCHIVE = 7;
		static const int PGM = 8;
		static const int IMAGEIO = 9;

		// Compression modes
		static const int COMPRESSION_UNKNOWN = 0;
		static const int COMPRESSION_NONE = 1;
		static const int LZW = 2;
		static const int LZW_WITH_DIFFERENCING = 3;
		static const int JPEG = 4;
		static const int PACK_BITS = 5;
		static const int ZIP = 6;

		static const int TYPE_NAME_SIZE_MAX = 50;

		
		/* File format (TIFF, GIF_OR_JPG, BMP, etc.). Used by the File/Revert command */
		int fileFormat;

		/* File type (GRAY8, GRAY_16_UNSIGNED, RGB, etc.) */
		int fileType;

		char *fileName;
		char *directory;
		char *url;
		int width;
		int height;
		int offset;  // Use getOffset() to read //initialize with 0!
		int nImages;
		int gapBetweenImages;
		bool whiteIsZero;
		bool intelByteOrder;
		int compression;
		int nStripOffsets;
		int *stripOffsets;
		int nStripLengths;
		int *stripLengths;
		int rowsPerStrip;
		int lutSize;
		byte *reds;
		byte *greens;
		byte *blues;
		//Object pixels;   
		char *debugInfo;
		int nSliceLabels;
		char **sliceLabels;
		char *info;
		std::istream *inputStream; //InputStream
		//VirtualStack virtualStack;

		double pixelWidth;
		double pixelHeight;
		double pixelDepth;
		char *unit;
		int calibrationFunction;
		short nCoefficients;
		double *coefficients;
		char *valueUnit;
		double frameInterval;
		char *description;
		// Use <i>longOffset</i> instead of <i>offset</i> when offset>2147483647.
		long long longOffset;  // Use getOffset() to read
		// Extra metadata to be stored in the TIFF header
		int extraMetaDataEntries;
		int *metaDataTypes; // must be < 0xffffff
		char **metaData;
		int nDisplayRanges;
		double *displayRanges;
		int nChannelLuts;
		int *nLuts;
		byte **channelLuts;
		int nRoi;
		byte *roi;
		int nOverlay;
		int *overlaySize;
		byte **overlay;
		int samplesPerPixel;
		char *openNextDir, *openNextName;

		/** Creates a FileInfo object with all of its fields set to their default value. */
		FileInfo();
		FileInfo(const FileInfo& fi);
		~FileInfo();

		FileInfo& operator =(const FileInfo& fi); 

		/** Returns the offset as a long. */
		inline long long getOffset();

		/** Returns the number of bytes used per pixel. */
		int getBytesPerPixel();
		unsigned int getImageSize();

		void to_c_string(char *s);

	};

	class DECLSPEC_DLLPORT TiffDecoder {

	protected:
		std::fstream *in; //RandomAccessStream
		bool debugMode;

	private:
		//constants
		static const int UNSIGNED = 1;
		static const int SIGNED = 2;
		static const int FLOATING_POINT = 3;

		//field types
		static const int SHORT = 3;
		static const int LONG = 4;

		// metadata types
		static const int MAGIC_NUMBER = 0x494a494a;  // "IJIJ"
		static const int INFO = 0x696e666f;  // "info" (Info image property)
		static const int LABELS = 0x6c61626c;  // "labl" (slice labels)
		static const int RANGES = 0x72616e67;  // "rang" (display ranges)
		static const int LUTS = 0x6c757473;  // "luts" (channel LUTs)
		static const int ROI = 0x726f6920;  // "roi " (ROI)
		static const int OVERLAY = 0x6f766572;  // "over" (overlay)
		
		static const int NAME_SIZE_MAX = 100;
		static const int GAP_INFO_SIZE_MAX = 100;

		char *directory;
		char *name;
		char *url;
		
		bool littleEndian;
		char *dInfo;
		int ifdCount;
		int nMetaDataCounts;
		int *metaDataCounts;
		char *tiffMetadata;
		int photoInterp;

		FileInfo *fis;
		size_t nFInfo;

		inline int getInt() throw(std::ios_base::failure);
		inline int getShort() throw(std::ios_base::failure);
		inline float readFloat();
		inline unsigned long long readULongLong() throw(std::ios_base::failure);
		inline double readDouble();
		long OpenImageFileHeader();
		int getValue(int fieldType, int count);
		void getColorMap(long long offset, FileInfo &fi);
		bool getString(char *str,int count, long offset);
		void decodeNIHImageHeader(int offset, FileInfo &fi);
		void dumpTag(int tag, int count, int value, const FileInfo fi);
		void getName(char *buf, int tag);
		double getRational(long long loc);
		bool OpenIFD(FileInfo &fi);
		void getMetaData(long long loc, FileInfo &fi);
		void getInfoProperty(int first, FileInfo &fi);
		void getSliceLabels(int first, int last, FileInfo &fi);
		void getDisplayRanges(int first, FileInfo &fi);
		void getLuts(int first, int last, FileInfo &fi);
		void getRoi(int first, FileInfo &fi);
		void getOverlay(int first, int last, FileInfo &fi);
		void error(const char *message);
		void skipUnknownType(int first, int last);
		void getGapInfo(char *s, FileInfo *fi, size_t length);
		
		void tifdec_InvalidParameterHandler(const wchar_t* expression,
											const wchar_t* function, 
											const wchar_t* file, 
											unsigned int line, 
											uintptr_t pReserved);
		static bool illegal_operation_flag;

	public:
		// tags
		static const int NEW_SUBFILE_TYPE = 254;
		static const int IMAGE_WIDTH = 256;
		static const int IMAGE_LENGTH = 257;
		static const int BITS_PER_SAMPLE = 258;
		static const int COMPRESSION = 259;
		static const int PHOTO_INTERP = 262;
		static const int IMAGE_DESCRIPTION = 270;
		static const int STRIP_OFFSETS = 273;
		static const int ORIENTATION = 274;
		static const int SAMPLES_PER_PIXEL = 277;
		static const int ROWS_PER_STRIP = 278;
		static const int STRIP_BYTE_COUNT = 279;
		static const int X_RESOLUTION = 282;
		static const int Y_RESOLUTION = 283;
		static const int PLANAR_CONFIGURATION = 284;
		static const int RESOLUTION_UNIT = 296;
		static const int SOFTWARE = 305;
		static const int DATE_TIME = 306;
		static const int ARTEST = 315;
		static const int HOST_COMPUTER = 316;
		static const int PREDICTOR = 317;
		static const int COLOR_MAP = 320;
		static const int TILE_WIDTH = 322;
		static const int SAMPLE_FORMAT = 339;
		static const int JPEG_TABLES = 347;
		static const int METAMORPH1 = 33628;
		static const int METAMORPH2 = 33629;
		static const int IPLAB = 34122;
		static const int NIH_IMAGE_HDR = 43314;
		static const int META_DATA_BYTE_COUNTS = 50838; // private tag registered with Adobe
		static const int META_DATA = 50839; // private tag registered with Adobe


		TiffDecoder(const char *directory, const char *name);
		//TiffDecoder(std::istream *in, char *name);
		~TiffDecoder();

		void saveImageDescription(const char *description, FileInfo &fi);
		void saveMetadata(const char *name, const char *data);
		void enableDebugging();
		bool loadTiffInfo();
		size_t getFileInfoNum();
		bool getFileInfo(FileInfo* &dst);
				
	};

	union Double_ULL_Converter{ 
		double d;
		unsigned long long ull; 
	};

	union Float_Int_Converter{ 
		float f;
		int itr; 
	};
	
	class DECLSPEC_DLLPORT ImageReader{
	private:
		FileInfo *fi;
		unsigned long long skipCount;
		unsigned int bytesPerPixel, bufferSize, nPixels;
		unsigned long long byteCount;
		unsigned int width, height;
		double min, max; // readRGB48() calculates min/max pixel values
		int eofErrorCount;
		bool isEmpty;
				
		bool debugMode;

	public:
		ImageReader();
		ImageReader(const FileInfo &src_info);
		~ImageReader();
		void init(const FileInfo &src_info);
		void clear();
		bool empty();
		void eofError();
		void read8bitImage(std::fstream &in, tiffhndl::byte* data);
		void read16bitImage(std::fstream &in, short* data);
		void read32bitImage(std::fstream &in, float* data);
		void skip(std::fstream &in);
		bool readPixels(std::fstream &in, tiffhndl::byte* data);
		bool readPixels(std::fstream &in, tiffhndl::byte* data, unsigned long long skipCount);
		void setDebugMode(bool val);
//		bool LoadContiguousImagesAsStack(FileInfo info, size_t len);
	};

	
	struct Calibration{
		double pixelWidth;
		double pixelHeight;
		double pixelDepth;
		double xOrigin;
		double yOrigin;
		double zOrigin;
		double fps;
		bool loop;
		double frameInterval;
		int channels;
		int slices;
		int frames;
		bool hyperstack;
		int width;
		int height;
		
		bool calibrated;
		bool loaded;
	};

	class DECLSPEC_DLLPORT FileOpener{
	private:
		FileInfo *fi;
		int nInfo;
		bool isEmpty;
		byte *pixels;
		Calibration fcal;

		bool debugMode;
		void initVal();
	public:
		static const int STACK_XYZ	 = 0;
		static const int STACK_XYCZ	 = 1;
		static const int STACK_XYCZT = 2;
		
		FileOpener();
		FileOpener(const char *directory, const char *name);
		FileOpener(const char *path);
		FileOpener(const FileInfo *src_info, int info_num);
		~FileOpener();
		bool init(const FileInfo *src_info, int info_num);
		bool open(const char *directory, const char *name);
		void clear();
		bool empty();
		void decodeDescriptionString(FileInfo &fi);
		Calibration getCalibration(FileInfo &fi);
		Calibration getImageCalibration();
		bool allSameSizeAndType(FileInfo *info, size_t len);
		bool openStack();
		bool loadTiffStack();
		bool getXYZStackRaw(byte *dst, int channel = 0, int frame = 0);
		bool getXYZStackFloat(float *dst, bool normalize = false, int channel = 0, int frame = 0);

		
		//各ChannelのXYZスタックデータをコピーする関数とXYZスタックのサイズを取得する関数が必要
		//FileOpenerは各種類のファイル読み込み用
		//Openerは読み込むファイルタイプを取得してFileOpenerのどの関数を使うか決定する
		//Lutを取得する関数を作成

		void setDebugMode(bool val);
	};

}

#pragma warning(pop)

#endif