#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <malloc.h>

#define DLL_MULTITIFFIO

#include "multi_tiff.h"

#include <tiffio.h>

using namespace multif;

MultiTiffIO::MultiTiffIO(const char filename[])
{
	image = TIFFOpen(filename, "r");
}

MultiTiffIO::~MultiTiffIO()
{
	if(image)TIFFClose(image);
}

int MultiTiffIO::GetField(unsigned int tag, ...)
{
	int return_val;
	va_list ap;

	if(image == NULL){
		printf("Image data is empty\n");
		return -1;
	}

	va_start(ap,tag);
	return_val = TIFFVGetField(image, tag, ap);
	va_end(ap);

	return return_val;
}

unsigned short MultiTiffIO::GetNumberOfDirectories()
{
	if(image == NULL){
		printf("Image data is empty\n");
		return 0;
	}
	return TIFFNumberOfDirectories(image);
}

unsigned long MultiTiffIO::GetDataSize()
{
	unsigned long img_size = 0L;
	
	if(image == NULL){
		printf("Image data is empty\n");
		return 0;
	}
	
	unsigned int dircount = TIFFNumberOfDirectories(image);
	unsigned int width, height, bitspersample;
	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bitspersample);
	/*
	for(int k = 0; k < dircount; k++)
	{
		if (!TIFFSetDirectory(image, k))
		{
			printf("ERROR: Can not set TIFF directory [TIFFSetDirectory] \n");
			return 0;
		}
		img_size += TIFFStripSize(image) * TIFFNumberOfStrips(image);
		printf("Stack[%d] StripSize: %lu NumberOfStrips: %lu\n", k, TIFFStripSize(image), TIFFNumberOfStrips(image));
	}
	printf("ImageDataSize: %lu\n", img_size);
	*/
	img_size = (unsigned long)width * (unsigned long)height * (unsigned long)dircount * (unsigned long)bitspersample / 8L;

	return img_size;
}

bool MultiTiffIO::OpenTiff(const char filename[])
{
	if(image)TIFFClose(image);
	image = TIFFOpen(filename, "r");
	if(image == NULL)return false;

	return true;
}

bool MultiTiffIO::GetImageData(char data[])
{
	assert(image != NULL);

	int		i, j;
	int		width, height, dircount;
	uint32	tifsize;
	uint16	samplesperpixel, bitspersample;
	uint16	photometric, fillorder, compression;
	uint16	orientation, resunit, planarconfig, sampleformat, datatype;
	tsize_t stripSize;
	unsigned long imageOffset, result;
	int stripMax, stripCount, scanlineSize;
	char temp1byte;
	uint16 temp2byte;
	unsigned long dataSize, count;

	if(data == NULL){
		printf("data == NULL\n");
		return false;
	}

	dircount = TIFFNumberOfDirectories(image);
	printf("TIFF Directories: %d \n", dircount);

	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bitspersample);
	TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel);
	TIFFGetField(image, TIFFTAG_COMPRESSION, &compression);
	TIFFGetField(image, TIFFTAG_PHOTOMETRIC, &photometric);
	TIFFGetField(image, TIFFTAG_FILLORDER, &fillorder);
	TIFFGetField(image, TIFFTAG_PLANARCONFIG, &planarconfig);
	TIFFGetField(image, TIFFTAG_ORIENTATION, &orientation);
	TIFFGetField(image, TIFFTAG_RESOLUTIONUNIT, &resunit);
	TIFFGetField(image, TIFFTAG_SAMPLEFORMAT, &sampleformat);
	 TIFFGetField(image, TIFFTAG_DATATYPE, &datatype);
	if(sampleformat == 0 && datatype == 0){
		printf("Invild sampleformat and datatype... use default datatype\n");
	}

	printf("width: %d \n", width);
	printf("height: %d \n", height);
	printf("bitspersample: %hd \n", bitspersample);
	printf("samplesperpixel: %hd \n", samplesperpixel);
	printf("compression: %hd \n", compression);
	printf("photometric: %hd \n", photometric);
	printf("fillorder: %hd \n", fillorder);
	printf("planarconfig: %hd \n", planarconfig);
	printf("resunit: %hd \n", resunit);
	printf("sampleformat: %hd \n", sampleformat);
	printf("datatype: %hd \n", datatype);
	printf("scanline size: %d \n", (int)TIFFScanlineSize(image));

	if(!(width > 0 && height > 0)){
		printf("Invild image size\n");
		return false;
	};
	
	if(!(samplesperpixel == 1)){
		printf("Invild number of samples per pixel\n");
		return false;
	};

	//support only 8bit_UINT, 16bit_UINT, 32bit_FLOAT
	if(TIFFGetField(image, TIFFTAG_SAMPLEFORMAT, &sampleformat) != 0){
		if(!(	(bitspersample == 8 && sampleformat == SAMPLEFORMAT_UINT) ||
				(bitspersample == 16 && sampleformat == SAMPLEFORMAT_UINT) ||
				(bitspersample == 32 && sampleformat == SAMPLEFORMAT_IEEEFP && photometric == PHOTOMETRIC_MINISBLACK) )){
			printf("Invild data type\n");
			return false;
		}
	}
	else if(TIFFGetField(image, TIFFTAG_DATATYPE, &datatype) != 0){
		if(!(	(bitspersample == 8 && datatype == 1) ||
				(bitspersample == 16 && datatype == 3) ||
				(bitspersample == 32 && datatype == 11 && photometric == PHOTOMETRIC_MINISBLACK) )){
			printf("Invild data type\n");
			return false;
		}
	}
	else{
		if(!(bitspersample == 8 || bitspersample == 16 || (bitspersample == 32 && photometric == PHOTOMETRIC_MINISBLACK))){
			printf("Invild data type\n");
			return false;
		}
	}

	
	dataSize = GetDataSize();
	
	imageOffset = 0;
	for(int k = 0; k < dircount; k++){
		if (!TIFFSetDirectory(image, k)){
			printf("ERROR: Can not set TIFF directory [TIFFSetDirectory] \n");
			return 0;
		}
		
		stripSize = TIFFStripSize(image);
		stripMax = TIFFNumberOfStrips(image);
		//printf("stripSize*stripMax: %d", stripSize*stripMax);
		for (stripCount = 0; stripCount < stripMax; stripCount++){
			if((result = TIFFReadEncodedStrip (image, stripCount, data + imageOffset, stripSize)) == -1){
				printf("Read error on input strip number %d\n", stripCount);
				return false;
			}
			imageOffset += result;
		}
	}

	if(TIFFGetField(image, TIFFTAG_PHOTOMETRIC, &photometric) == 0){
		printf("Image has an undefined photometric interpretation\n");
		return false;
	}
	if(photometric == PHOTOMETRIC_MINISWHITE){
		printf("Fixing the photometric interpretation\n");
		for(count = 0; count < dataSize; count++)
			data[count] = ~data[count];
	}

	//default: FILLORDER_MSB2LSB
	if( (TIFFGetField(image, TIFFTAG_FILLORDER, &fillorder) != 0) && fillorder != FILLORDER_MSB2LSB){
		if(bitspersample == 8){
			printf("Fixing the fillorder (bitspersample = 8)\n");
			for(count = 0; count < dataSize; count++){
				temp1byte = 0;
				if(data[count] & 128) temp1byte += 1;
				if(data[count] & 64) temp1byte += 2;
				if(data[count] & 32) temp1byte += 4;
				if(data[count] & 16) temp1byte += 8;
				if(data[count] & 8) temp1byte += 16;
				if(data[count] & 4) temp1byte += 32;
				if(data[count] & 2) temp1byte += 64;
				if(data[count] & 1) temp1byte += 128;
				data[count] = temp1byte;
			}
		}
		if(bitspersample == 16){
			printf("Fixing the fillorder (bitspersample = 16)\n");
			for(count = 0; count < (dataSize >> 1); count++){
				temp2byte = 0;
				if(((uint16 *)data)[count] & 0x8000) temp2byte += 0x0001;
				if(((uint16 *)data)[count] & 0x4000) temp2byte += 0x0002;
				if(((uint16 *)data)[count] & 0x2000) temp2byte += 0x0004;
				if(((uint16 *)data)[count] & 0x1000) temp2byte += 0x0008;
				if(((uint16 *)data)[count] & 0x0800) temp2byte += 0x0010;
				if(((uint16 *)data)[count] & 0x0400) temp2byte += 0x0020;
				if(((uint16 *)data)[count] & 0x0200) temp2byte += 0x0040;
				if(((uint16 *)data)[count] & 0x0100) temp2byte += 0x0080;
				if(((uint16 *)data)[count] & 0x0080) temp2byte += 0x0100;
				if(((uint16 *)data)[count] & 0x0040) temp2byte += 0x0200;
				if(((uint16 *)data)[count] & 0x0020) temp2byte += 0x0400;
				if(((uint16 *)data)[count] & 0x0010) temp2byte += 0x0800;
				if(((uint16 *)data)[count] & 0x0008) temp2byte += 0x1000;
				if(((uint16 *)data)[count] & 0x0004) temp2byte += 0x2000;
				if(((uint16 *)data)[count] & 0x0002) temp2byte += 0x4000;
				if(((uint16 *)data)[count] & 0x0001) temp2byte += 0x8000;
				((uint16 *)data)[count] = temp2byte;
			}
		}
	}
	
	if(bitspersample == 8){
		for(count = 0; count < dataSize && count <= 8; count++){
			printf("%02x", (unsigned char) data[count]);
			if((count + 1) % (width / 8) == 0)printf("\n");
			else printf(" ");
		}
		printf("\n");
	}
	if(bitspersample == 16){
		for(count = 0; count < (dataSize >> 1) && count <= 8; count++){
			printf("%04x", ((uint16 *)data)[count]);
			if(((count >> 1) + 1) % (width / 8) == 0)printf("\n");
			else printf(" ");
		}
		printf("\n");
	}

	return true;
}

bool MultiTiffIO::SaveImageData(const char filename[], char data[], int width, int height, int nSlices, int bitspersamples, int samplesperpixel)
{
	int32	tifsize;
	uint16	photometric, fillorder, compression;
	uint16	orientation, resunit, planarconfig, sampleformat, datatype;
	unsigned long imageOffset, result;
	TIFF *wtif;

	printf("--MultiTiffIO::SaveImageData--\n");

	if(data == NULL){
		printf("data == NULL\n");
		return false;
	}

	if(filename == NULL){
		printf("filename == NULL\n");
		return false;
	}

	if(samplesperpixel != 1 &&  samplesperpixel != 3){
		printf("samplesperpixel != 1 &&  samplesperpixel != 3\n");
		return false;
	}

	wtif = TIFFOpen(filename, "w");
	if(wtif == NULL){
		printf("TIFFOpen: error\n");
		return false;
	}

	printf("TIFF Directories: %d \n", nSlices);

	printf("width: %d \n", width);
	printf("height: %d \n", height);

	tifsize = width * height * (bitspersamples / 8) * samplesperpixel;
	printf("page size: %d \n", tifsize);

	imageOffset = 0;
	for(uint16 p_num = 0; p_num < nSlices; p_num++){
		TIFFSetField(wtif, TIFFTAG_SOFTWARE, filename);
		TIFFSetField(wtif, TIFFTAG_IMAGEWIDTH, width);
		TIFFSetField(wtif, TIFFTAG_IMAGELENGTH, height);
		TIFFSetField(wtif, TIFFTAG_BITSPERSAMPLE, bitspersamples);
		TIFFSetField(wtif, TIFFTAG_SAMPLESPERPIXEL, samplesperpixel);
		TIFFSetField(wtif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		TIFFSetField(wtif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
		TIFFSetField(wtif, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
		TIFFSetField(wtif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(wtif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
		TIFFSetField(wtif, TIFFTAG_ROWSPERSTRIP, height);
		//if(TIFFGetField(image, TIFFTAG_RESOLUTIONUNIT, &resunit) != 0)		TIFFSetField(wtif, TIFFTAG_RESOLUTIONUNIT, &resunit);
		if(bitspersamples == 32)TIFFSetField(wtif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
		if(bitspersamples == 16)TIFFSetField(wtif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
		if(bitspersamples == 8)TIFFSetField(wtif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
		//if(TIFFGetField(image, TIFFTAG_DATATYPE, &datatype) != 0)				TIFFSetField(wtif, TIFFTAG_DATATYPE, &datatype);

		if(p_num == 0)TIFFSetField(wtif, TIFFTAG_SUBFILETYPE, 0);
		else TIFFSetField(wtif, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
		TIFFSetField(wtif, TIFFTAG_PAGENUMBER, p_num, nSlices);
		
		if((result = TIFFWriteEncodedStrip(wtif, 0, data + imageOffset, tifsize)) == -1){
			printf("TIFF write error\n");
			return false;
		}
		imageOffset += result;
		TIFFWriteDirectory(wtif);
	}

	TIFFClose(wtif);

	printf("--Complete--\n");

	return true;
}

bool MultiTiffIO::SaveImageDataSingle(const char filename[], char data[], int width, int height, int bitspersamples, int samplesperpixel)
{
	int32	tifsize;
	uint16	photometric, fillorder, compression;
	uint16	orientation, resunit, planarconfig, sampleformat, datatype;
	unsigned long imageOffset, result;
	TIFF *wtif;

	printf("--MultiTiffIO::SaveImageDataSingle--\n");

	if(data == NULL){
		printf("data == NULL\n");
		return false;
	}

	if(filename == NULL){
		printf("filename == NULL\n");
		return false;
	}

	if(samplesperpixel != 1 &&  samplesperpixel != 3){
		printf("samplesperpixel != 1 &&  samplesperpixel != 3\n");
		return false;
	}

	wtif = TIFFOpen(filename, "w");
	if(wtif == NULL){
		printf("TIFFOpen: error\n");
		return false;
	}

	printf("width: %d \n", width);
	printf("height: %d \n", height);

	tifsize = width * height * (bitspersamples / 8) * samplesperpixel;
	printf("page size: %d \n", tifsize);

	//TIFFSetField(wtif, TIFFTAG_SOFTWARE, filename);
	TIFFSetField(wtif, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(wtif, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(wtif, TIFFTAG_BITSPERSAMPLE, bitspersamples);
	TIFFSetField(wtif, TIFFTAG_SAMPLESPERPIXEL, samplesperpixel);
	TIFFSetField(wtif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
	TIFFSetField(wtif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	TIFFSetField(wtif, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
	TIFFSetField(wtif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(wtif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	if(bitspersamples == 32)TIFFSetField(wtif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
	if(bitspersamples == 16)TIFFSetField(wtif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
	if(bitspersamples == 8)TIFFSetField(wtif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
	//TIFFSetField(wtif, TIFFTAG_ROWSPERSTRIP, height);
	//if(TIFFGetField(image, TIFFTAG_RESOLUTIONUNIT, &resunit) != 0)		TIFFSetField(wtif, TIFFTAG_RESOLUTIONUNIT, &resunit);
	//if(TIFFGetField(image, TIFFTAG_SAMPLEFORMAT, &sampleformat) != 0)		TIFFSetField(wtif, TIFFTAG_SAMPLEFORMAT, &sampleformat);
	//if(TIFFGetField(image, TIFFTAG_DATATYPE, &datatype) != 0)				TIFFSetField(wtif, TIFFTAG_DATATYPE, &datatype);

	if((result = TIFFWriteEncodedStrip(wtif, 0, data, tifsize)) == -1){
		printf("TIFF write error\n");
		return false;
	}
	TIFFWriteDirectory(wtif);

	TIFFClose(wtif);

	printf("--Complete--\n");

	return true;
}

void MultiTiffIO::GetChSeparatedImageData()
{
}

bool MultiTiffIO::empty()
{
	return (image == NULL) ? true : false;
}