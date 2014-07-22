#include <tiffio.h>

#ifndef MULTI_TIFF_H
#define MULTI_TIFF_H

#ifdef DLL_MULTITIFFIO
#define DECLSPEC_DLLPORT	__declspec(dllexport)
#else
#define DECLSPEC_DLLPORT	__declspec(dllimport)
#endif

namespace multif{

	class DECLSPEC_DLLPORT MultiTiffIO{
	private:
		TIFF *image;
	public:
		MultiTiffIO(const char filename[]);
		~MultiTiffIO();
		int GetField(uint32 tag, ...);
		uint16 GetNumberOfDirectories();
		void GetChSeparatedImageData();
		unsigned long GetDataSize();
		bool GetImageData(char data[]);
		bool OpenTiff(const char filename[]);
		static bool SaveImageData(const char filename[], char data[], int width, int height, int nSlices, int bitspersamples, int samplesperpixel);
		static bool SaveImageDataSingle(const char filename[], char data[], int width, int height, int bitspersamples, int samplesperpixel);
		bool empty();
	};

}

#endif