/*
 * DSLT Demo
 *
 * Copyright (C) 2014 Kyoto University
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License or any
 * later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
  ImageJ

Copyright (c) 2009 - 2014, Board of Regents of the University of
Wisconsin-Madison, Broad Institute of MIT and Harvard, and Max Planck
Institute of Molecular Cell Biology and Genetics.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

//#include <tbb/tbb.h>
#define DLL_TIFF_HNDL
#include "tiffdecorder.h"

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

using namespace tiffhndl;
using namespace std;

#pragma warning(push)
#pragma warning(disable : 4290)

#define SAFE_DELETE(p) { if (p){ delete p; } p = NULL; }
#define SAFE_DELETE_ARRAY(p) { if (p){ delete [] p; } p = NULL; }

template <class T> inline void SAFE_DELETE_2D_ARRAY(T ** &p, int rows)
{
	if(p && rows > 0){
		for(int i = 0; i < rows; i++) delete [] p[i];
		delete [] p;
		p = NULL;
	}
}

inline void _reset_stringstream(stringstream& ss)
{
	static const string empty_string;

	ss.str(empty_string);
	ss.clear();
	ss << dec;     // clear()でも元に戻らないので、毎回指定する。
}

inline void _stringNewAndCopy(char * &dst, const char *src)
{
	if(src != NULL && strlen(src) > 0){
		dst = new char[strlen(src)+1];
		strcpy_s(dst, strlen(src)+1, src);
	}
}

inline void _2DstringNewAndCopy(char ** &dst, char ** const src, int rows)
{
	if(src != NULL && rows > 0){
		dst = new char *[rows];
		for(int i=0; i<rows; i++){
			if(src[i] != NULL){
				dst[i] = new char[strlen(src[i])+1];
				strcpy_s(dst[i], strlen(src[i])+1, src[i]);
			}
			else dst[i] = NULL;
		}
	}
}

template <class T> inline void _arrayNewAndCopy(T * &dst, size_t length, T *src)
{
	if(src != NULL && length > 0){
		dst = new T[length];
		memcpy_s(dst, sizeof(T)*length, src, sizeof(T)*length);
	}
}

template <class T> inline void _2DarrayNewAndCopy(T ** &dst, size_t rows, T **src, int *length)
{
	if(src != NULL && rows > 0 && length != NULL){
		dst = new T *[rows];
		for(int i=0; i < rows; i++){
			if(length[i] > 0 && src[i] != NULL){
				dst[i] = new T[length[i]];
				memcpy_s(dst[i], sizeof(T)*length[i], src[i], sizeof(T)*length[i]);
			}
			else dst[i] = NULL;
		}
	}
}

bool getPropertyBool(const char *prop, const char *name, bool default_val = false)
{
	if(prop == NULL)return default_val;

	bool b;
	string prop_str(prop);
	string name_str(name);
	name_str += '=';
	string::size_type index1 = prop_str.find(name);
	
	if (index1 != string::npos) {
		string::size_type index2 = prop_str.find_first_of("\n\r", index1);
		if (index2 == string::npos) index2 = prop_str.length()-1;//区切り文字が無い場合は末尾と判断
		index1 += name_str.length();

		string bool_str = prop_str.substr(index1, index2-index1);
		if(bool_str == "true")b = true;
		else if(bool_str == "false")b = false;
	}
	else b = default_val;

	return b;
}

int getPropertyInt(const char *prop, const char *name, int default_val = 0)
{
	if(prop == NULL)return default_val;

	int n;
	string prop_str(prop);
	string name_str(name);
	name_str += '=';
	string::size_type index1 = prop_str.find(name);
	
	if (index1 != string::npos) {
		string::size_type index2 = prop_str.find_first_of("\n\r", index1);
		if (index2 == string::npos) index2 = prop_str.length()-1;//区切り文字が無い場合は末尾と判断
		index1 += name_str.length();

		stringstream ss;
		ss << prop_str.substr(index1, index2-index1);
		ss >> n;
	}
	else n = default_val;

	return n;
}

double getPropertyDouble(const char *prop, const char *name, double default_val = 0.0)
{
	if(prop == NULL)return default_val;

	double d;
	string prop_str(prop);
	string name_str(name);
	name_str += '=';
	string::size_type index1 = prop_str.find(name);
	
	if (index1 != string::npos) {
		string::size_type index2 = prop_str.find_first_of("\n\r", index1);
		if (index2 == string::npos) index2 = prop_str.length()-1;//区切り文字が無い場合は末尾と判断
		index1 += name_str.length();

		stringstream ss;
		ss << prop_str.substr(index1, index2-index1);
		ss >> d;
	}
	else d = default_val;

	return d;
}

string getPropertyString(const char *prop, const char *name)
{
	string dst;
	if(prop == NULL)return dst;

	string prop_str(prop);
	string name_str(name);
	name_str += '=';
	string::size_type index1 = prop_str.find(name);
		
	if(index1 != string::npos){
		string::size_type index2 = prop_str.find_first_of("\n\r", index1);
		if (index2 == string::npos) index2 = prop_str.length()-1;//区切り文字が無い場合は末尾と判断
		index1 += name_str.length();
		dst = prop_str.substr(index1, index2-index1);
	}

	return dst;
}

TiffDecoder::TiffDecoder(const char *directory, const char *name){
	if(directory != NULL){
		this->directory = new char[strlen(directory)+1];
		strcpy_s(this->directory, strlen(directory)+1, directory);
	}
	if(name != NULL){
		this->name = new char [strlen(name)+1];
		strcpy_s(this->name, strlen(name)+1, name);
	}
	debugMode = false;
	url = NULL;
	dInfo = NULL;
	nMetaDataCounts = 0;
	ifdCount = 0;
	metaDataCounts = NULL;
	tiffMetadata = NULL;
	in = NULL;
	fis = NULL;
	nFInfo = 0;
}

TiffDecoder::~TiffDecoder(){
	if (in != NULL)in->close();
	SAFE_DELETE(in);
	SAFE_DELETE_ARRAY(directory);
	SAFE_DELETE_ARRAY(name);
	SAFE_DELETE_ARRAY(url);
	SAFE_DELETE_ARRAY(dInfo);
	SAFE_DELETE_ARRAY(metaDataCounts);
	SAFE_DELETE_ARRAY(tiffMetadata);
	SAFE_DELETE_ARRAY(fis);
}

inline int TiffDecoder::getInt() throw(ios_base::failure){
	int b1 = in->get();
	int b2 = in->get();
	int b3 = in->get();
	int b4 = in->get();
	if (littleEndian)
		return ((b4 << 24) + (b3 << 16) + (b2 << 8) + (b1 << 0));
	else
		return ((b1 << 24) + (b2 << 16) + (b3 << 8) + b4);
}

inline int TiffDecoder::getShort() throw(ios_base::failure){
	int b1 = in->get();
	int b2 = in->get();
	if (littleEndian)
		return ((b2<<8) + b1);
	else
		return ((b1<<8) + b2);
}

inline float TiffDecoder::readFloat() throw(ios_base::failure){
	Float_Int_Converter conv;
	conv.itr = getInt();
	return conv.f;
}

inline unsigned long long TiffDecoder::readULongLong() throw(ios_base::failure){
	if (littleEndian)
		return ((unsigned long long)getInt()&0xffffffffULL) + ((unsigned long long)getInt()<<32);
	else
		return ((unsigned long long)getInt()<<32) + ((unsigned long long)getInt()&0xffffffffULL);

}

inline double TiffDecoder::readDouble() throw(ios_base::failure){
	Double_ULL_Converter conv;
	conv.ull = readULongLong();
	return conv.d;
}

//magicNumberに格納している値はTiffのVersionNumber(常に42)
//4GBを超えるTiff(BigTiff)はこの値が43になっておりOffset値が64bitになっている
long TiffDecoder::OpenImageFileHeader() throw(ios_base::failure){
	// Open 8-byte Image File Header at start of file.
	// Returns the offset in bytes to the first IFD or -1
	// if this is not a valid tiff file.
	short byte1 = in->get();
	short byte2 = in->get();
	if ((byte2<<8)+byte1 == 0x4949) // "II"
		littleEndian = true;
	else if ((byte2<<8)+byte1 == 0x4d4d) // "MM"
		littleEndian = false;
	else {
		in->close();
		//cout << "Invalid Endian\n" << std::hex << (byte2<<8)+byte1 << std::dec << endl;
		return -1;
	}
	int magicNumber = getShort(); // 42
	long offset = ((long)getInt())&0xffffffffL;
	return offset;
}


int TiffDecoder::getValue(int fieldType, int count) throw(ios_base::failure){
	int value = 0;
	int unused;
	if (fieldType==SHORT && count==1) {
		value = getShort();
		unused = getShort();
	} else
		value = getInt();
	return value;
}	

void TiffDecoder::getColorMap(long long offset, FileInfo &fi) throw(ios_base::failure){
	const int tablesize = 768*2;
	byte *colorTable16 = new byte[768*2];
	long long saveLoc = in->tellg();
	in->seekg(offset, in->beg);
	in->read((char *)colorTable16, tablesize);
	in->seekg(saveLoc, in->beg);
	fi.lutSize = 256;
	SAFE_DELETE_ARRAY(fi.reds);
	SAFE_DELETE_ARRAY(fi.greens);
	SAFE_DELETE_ARRAY(fi.blues);
	fi.reds = new byte[fi.lutSize];
	fi.greens = new byte[fi.lutSize];
	fi.blues = new byte[fi.lutSize];
	int j = 0;
	if (littleEndian) j++;
	int sum = 0;
	for (int i=0; i<256; i++) {
		fi.reds[i] = colorTable16[j];
		sum += fi.reds[i];
		fi.greens[i] = colorTable16[512+j];
		sum += fi.greens[i];
		fi.blues[i] = colorTable16[1024+j];
		sum += fi.blues[i];
		j += 2;
	}
	if (sum!=0)
		fi.fileType = FileInfo::COLOR8;

	delete [] colorTable16;
}


bool TiffDecoder::getString(char *str, int count, long offset) throw(ios_base::failure){
	if(str == NULL)return false;
	//count--; // skip null byte at end of string
	if (count<=4)return false;
	long long saveLoc = in->tellg();
	in->seekg(offset, in->beg);
	in->read(str, count);// istream::get()は改行で止まるので不適
	if(in->bad())return false;
	in->seekg(saveLoc, in->beg);
	return true;
}


/** Save the image description in the specified FileInfo. ImageJ
saves spatial and density calibration data in this string. For
stacks, it also saves the number of images to avoid having to
decode an IFD for each image. */
void TiffDecoder::saveImageDescription(const char *description, FileInfo &fi) {
	if(description == NULL)return;
	
	string id(description);
		
	if (id.find("ImageJ") != 0){
		char namebuf[NAME_SIZE_MAX];
		getName(namebuf, IMAGE_DESCRIPTION);
		saveMetadata(namebuf, id.c_str());
	}
	if (id.length() < 7) return;
	
	SAFE_DELETE_ARRAY(fi.description);
	fi.description = new char[id.length()+1];
	strcpy_s(fi.description, id.length()+1, id.c_str()) ;

	int n = getPropertyInt(description, "images");
	if (n > 1) fi.nImages = n;
}


void TiffDecoder::saveMetadata(const char *name, const char *data) {
	if (data == NULL) return;
	string str(name+string(": ")+data+string("\n"));
	if (tiffMetadata == NULL){
		tiffMetadata = new char[str.length()+1];//+1は\0の分
		strcpy_s(tiffMetadata, str.length()+1, str.c_str());
	}
	else{
		string tmp(tiffMetadata + str);
		SAFE_DELETE_ARRAY(tiffMetadata);
		tiffMetadata = new char[tmp.length()+1];//+1は\0の分
		strcpy_s(tiffMetadata, tmp.length()+1, tmp.c_str());
	}
}

//Support in this DLL is limited to uncompressed, little-endian NIH image header
void TiffDecoder::decodeNIHImageHeader(int offset, FileInfo &fi) throw(ios_base::failure){
	long long saveLoc = in->tellg();

	in->seekg(offset+12, in->beg);
	short version = 0;
	version = getShort();

	in->seekg(offset+160, in->beg);
	double scale = 0.0;
	scale = readDouble();
	if (version > 106 && scale != 0.0) {
		fi.pixelWidth = 1.0/scale;
		fi.pixelHeight = fi.pixelWidth;
	}

	// spatial calibration
	in->seekg(offset+172, in->beg);
	short units = 0;
	units = getShort();
	if (version <= 153) units += 5;
	string ustr;
	switch (units) {
	case 5: ustr = "nanometer"; break;
	case 6: ustr = "micrometer"; break;
	case 7: ustr = "mm"; break;
	case 8: ustr = "cm"; break;
	case 9: ustr = "meter"; break;
	case 10: ustr = "km"; break;
	case 11: ustr = "inch"; break;
	case 12: ustr = "ft"; break;
	case 13: ustr = "mi"; break;
	}
	if(!ustr.empty()){
		SAFE_DELETE_ARRAY(fi.unit);
		fi.unit = new char[ustr.length()+1];
		strcpy_s(fi.unit, ustr.length()+1, ustr.c_str());
	}

	// density calibration
	in->seekg(offset+182, in->beg);
	byte fitType = 0;
	fitType = in->get();
	byte unused = 0;
	unused = in->get();
	fi.nCoefficients = 0;
	fi.nCoefficients = getShort();
	if (fitType==11) {
		fi.calibrationFunction = 21; //Calibration.UNCALIBRATED_OD
		string vustr("U. OD");
		SAFE_DELETE_ARRAY(fi.valueUnit);
		fi.valueUnit = new char[vustr.length()+1];
		strcpy_s(fi.valueUnit, vustr.length()+1, vustr.c_str());
	} else if (fitType>=0 && fitType<=8 && fi.nCoefficients>=1 && fi.nCoefficients<=5) {
		switch (fitType) {
		case 0: fi.calibrationFunction = 0; break; //Calibration.STRAIGHT_LINE
		case 1: fi.calibrationFunction = 1; break; //Calibration.POLY2
		case 2: fi.calibrationFunction = 2; break; //Calibration.POLY3
		case 3: fi.calibrationFunction = 3; break; //Calibration.POLY4
		case 5: fi.calibrationFunction = 4; break; //Calibration.EXPONENTIAL
		case 6: fi.calibrationFunction = 5; break; //Calibration.POWER
		case 7: fi.calibrationFunction = 6; break; //Calibration.LOG
		case 8: fi.calibrationFunction = 10; break; //Calibration.RODBARD2 (NIH Image)
		}
		SAFE_DELETE_ARRAY(fi.coefficients);
		fi.coefficients = new double[fi.nCoefficients];
		for (int i=0; i<fi.nCoefficients; i++) {
			fi.coefficients[i] = readDouble();
		}
		in->seekg(offset+234, in->beg);
		byte size = 0;
		size = in->get();
		SAFE_DELETE_ARRAY(fi.valueUnit);
		if (size>=1 && size<=16) {
			stringstream ss;
			char ch;
			for (int i=0; i<size; i++){
				ch = in->get();
				ss << ch;
			}
			fi.valueUnit = new char[ss.str().length()+1];
			strcpy_s(fi.valueUnit, ss.str().length()+1, ss.str().c_str());
		} else{
			fi.valueUnit = new char[2];
			strcpy_s(fi.valueUnit, 2, " ");
		}
	}

	in->seekg(offset+260, in->beg);
	short nImages = 1;
	nImages = getShort();
	if(nImages>=2 && (fi.fileType==FileInfo::GRAY8||fi.fileType==FileInfo::COLOR8)) {
		fi.nImages = nImages;
		fi.pixelDepth = readFloat();	//SliceSpacing
		short skip = 0;
		skip = getShort();				//CurrentSlice
		fi.frameInterval = readFloat();
	}

	in->seekg(offset+272, in->beg);
	float aspectRatio = 0.0f;
	aspectRatio = readFloat();
	if (version>140 && aspectRatio!=0.0f)
		fi.pixelHeight = fi.pixelWidth/aspectRatio;

	in->seekg(saveLoc, in->beg);
}

void TiffDecoder::getName(char *buf,int tag) {
	if(buf == NULL)return;
	string name;
	switch (tag) {
	case NEW_SUBFILE_TYPE: name="NewSubfileType"; break;
	case IMAGE_WIDTH: name="ImageWidth"; break;
	case IMAGE_LENGTH: name="ImageLength"; break;
	case STRIP_OFFSETS: name="StripOffsets"; break;
	case ORIENTATION: name="Orientation"; break;
	case PHOTO_INTERP: name="PhotoInterp"; break;
	case IMAGE_DESCRIPTION: name="ImageDescription"; break;
	case BITS_PER_SAMPLE: name="BitsPerSample"; break;
	case SAMPLES_PER_PIXEL: name="SamplesPerPixel"; break;
	case ROWS_PER_STRIP: name="RowsPerStrip"; break;
	case STRIP_BYTE_COUNT: name="StripByteCount"; break;
	case X_RESOLUTION: name="XResolution"; break;
	case Y_RESOLUTION: name="YResolution"; break;
	case RESOLUTION_UNIT: name="ResolutionUnit"; break;
	case SOFTWARE: name="Software"; break;
	case DATE_TIME: name="DateTime"; break;
	case ARTEST: name="Artest"; break;
	case HOST_COMPUTER: name="HostComputer"; break;
	case PLANAR_CONFIGURATION: name="PlanarConfiguration"; break;
	case COMPRESSION: name="Compression"; break; 
	case PREDICTOR: name="Predictor"; break; 
	case COLOR_MAP: name="ColorMap"; break; 
	case SAMPLE_FORMAT: name="SampleFormat"; break; 
	case JPEG_TABLES: name="JPEGTables"; break; 
	case NIH_IMAGE_HDR: name="NIHImageHeader"; break; 
	case META_DATA_BYTE_COUNTS: name="MetaDataByteCounts"; break; 
	case META_DATA: name="MetaData"; break; 
	default: name="???"; break;
	}
	strcpy_s(buf, NAME_SIZE_MAX, name.c_str());
}

void TiffDecoder::dumpTag(int tag, int count, int value, const FileInfo fi) {
	long long lvalue = ((long long)value)&0xffffffffL;
	char name[NAME_SIZE_MAX];
	stringstream ss, tss;

	getName(name, tag);
	
	if(count != 1)ss << ", count=" << count;
	else ss << "";

	if(dInfo != NULL){
		tss << dInfo;
		SAFE_DELETE_ARRAY(dInfo);
	}
	tss << "    " << tag << ", \"" << name << "\", value=" << lvalue << ss.str() << "\n";
	dInfo = new char[tss.str().length()+1];
	strcpy_s(dInfo, (tss.str()).length()+1, tss.str().c_str());
}

double TiffDecoder::getRational(long long loc) throw(ios_base::failure){
	long long saveLoc = in->tellg();
	in->seekg(loc, in->beg);
	int numerator = getInt();
	int denominator = getInt();
	in->seekg(saveLoc, in->beg);
	//System.out.println("numerator: "+numerator);
	//System.out.println("denominator: "+denominator);
	if (denominator!=0)
		return (double)numerator/denominator;
	else
		return 0.0;
}


bool TiffDecoder::OpenIFD(FileInfo &fi) throw(ios_base::failure){
	// Get Image File Directory data
	int tag, fieldType, count, value;
	stringstream ermsg;
	int nEntries = getShort();
	if (nEntries<1 || nEntries>1000)
		return false;
	ifdCount++;

	double xScale, yScale;
	long long saveLoc;
		
	for (int i=0; i<nEntries; i++) {
		tag = getShort();
		fieldType = getShort();
		count = getInt();
		value = getValue(fieldType, count);
		long long lvalue = ((long)value)&0xffffffffL;
		if (debugMode && ifdCount<10) dumpTag(tag, count, value, fi);
		
		switch (tag) {
		case IMAGE_WIDTH: 
			fi.width = value;
			fi.intelByteOrder = littleEndian;
			break;
		case IMAGE_LENGTH: 
			fi.height = value;
			break;
		case STRIP_OFFSETS:
			SAFE_DELETE_ARRAY(fi.stripOffsets);
			if (count==1)
				fi.stripOffsets = new int(value);
			else {
				saveLoc = in->tellg();
				in->seekg(lvalue, in->beg);
				fi.stripOffsets = new int[count];
				for (int c=0; c<count; c++)
					fi.stripOffsets[c] = getInt();
				in->seekg(saveLoc, in->beg);
			}
			fi.nStripOffsets = count;
			fi.offset = count>0?fi.stripOffsets[0]:value; //offsetはlong longの方が良い？
			if (count>1 && (((long long)fi.stripOffsets[count-1])&0xffffffffLL)<(((long long)fi.stripOffsets[0])&0xffffffffLL))
				fi.offset = fi.stripOffsets[count-1];
			break;
		case STRIP_BYTE_COUNT:
			SAFE_DELETE_ARRAY(fi.stripLengths);
			if (count==1)
				fi.stripLengths = new int(value);
			else {
				saveLoc = in->tellg();
				in->seekg(lvalue, in->beg);
				fi.stripLengths = new int[count];
				for (int c=0; c<count; c++) {
					if (fieldType==SHORT)
						fi.stripLengths[c] = getShort();
					else
						fi.stripLengths[c] = getInt();
				}
				fi.nStripLengths = count;
				in->seekg(saveLoc, in->beg);
			}
			break;
		case PHOTO_INTERP:
			photoInterp = value;
			fi.whiteIsZero = value==0;
			break;
		case BITS_PER_SAMPLE:
			if (count==1) {
				if (value==8)
					fi.fileType = FileInfo::GRAY8;
				else if (value==16)
					fi.fileType = FileInfo::GRAY16_UNSIGNED;
				else if (value==32)
					fi.fileType = FileInfo::GRAY32_INT;
				else if (value==12)
					fi.fileType = FileInfo::GRAY12_UNSIGNED;
				else if (value==1)
					fi.fileType = FileInfo::BITMAP;
				else{
					_reset_stringstream(ermsg);
					ermsg << "Unsupported BitsPerSample: " << value;
					error(ermsg.str().c_str());
				}
			} else if (count==3) {
				saveLoc = in->tellg();
				in->seekg(lvalue, in->beg);
				int bitDepth = getShort();
				if (!(bitDepth==8||bitDepth==16)){
					_reset_stringstream(ermsg);
					ermsg << "ImageJ can only open 8 and 16 bit/channel RGB images (" << bitDepth << ")";
					error(ermsg.str().c_str());
				}
				if (bitDepth==16)
					fi.fileType = FileInfo::RGB48;
				in->seekg(saveLoc, in->beg);
			}
			break;
		case SAMPLES_PER_PIXEL:
			fi.samplesPerPixel = value;
			if (value==3 && fi.fileType!=FileInfo::RGB48)
				fi.fileType = (fi.fileType == FileInfo::GRAY16_UNSIGNED) ? FileInfo::RGB48 : FileInfo::RGB;
			else if (value==4 && fi.fileType==FileInfo::GRAY8) {
				fi.fileType = (photoInterp == 5) ? FileInfo::CMYK : FileInfo::ARGB;
			}
			break;
		case ROWS_PER_STRIP:
			fi.rowsPerStrip = value;
			break;
		case X_RESOLUTION:
			xScale = getRational(lvalue); 
			if (xScale!=0.0) fi.pixelWidth = 1.0/xScale; 
			break;
		case Y_RESOLUTION:
			yScale = getRational(lvalue); 
			if (yScale!=0.0) fi.pixelHeight = 1.0/yScale; 
			break;
		case RESOLUTION_UNIT:
			if (value==1&&fi.unit==NULL){
				SAFE_DELETE_ARRAY(fi.unit);
				_stringNewAndCopy(fi.unit, " ");
			}
			else if (value==2) {
				if (fi.pixelWidth==1.0/72.0) {
					fi.pixelWidth = 1.0;
					fi.pixelHeight = 1.0;
				} else{
					SAFE_DELETE_ARRAY(fi.unit);
					_stringNewAndCopy(fi.unit, "inch");
				}
			} else if (value==3){
				SAFE_DELETE_ARRAY(fi.unit);
				_stringNewAndCopy(fi.unit, "cm");
			}
			break;
		case PLANAR_CONFIGURATION:  // 1=chunky, 2=planar
			if (value==2 && fi.fileType==FileInfo::RGB48)
				fi.fileType = FileInfo::GRAY16_UNSIGNED;
			else if (value==2 && fi.fileType==FileInfo::RGB)
				fi.fileType = FileInfo::RGB_PLANAR;
			else if (value==1 && fi.samplesPerPixel==4) {
				fi.fileType = (photoInterp == 5) ? FileInfo::CMYK : FileInfo::ARGB;
			} else if (value!=2 && !((fi.samplesPerPixel==1)||(fi.samplesPerPixel==3))) {
				_reset_stringstream(ermsg);
				ermsg << "Unsupported SamplesPerPixel: " << fi.samplesPerPixel;
				error(ermsg.str().c_str());
			}
			break;
		case COMPRESSION:
			if (value==5)  // LZW compression
				fi.compression = FileInfo::LZW;
			else if (value==32773)  // PackBits compression
				fi.compression = FileInfo::PACK_BITS;
			else if (value==32946 || value==8)
				fi.compression = FileInfo::ZIP;
			else if (value!=1 && value!=0 && !(value==7&&fi.width<500)) {
				// don't abort with Spot camera compressed (7) thumbnails
				// otherwise, this is an unknown compression type
				fi.compression = FileInfo::COMPRESSION_UNKNOWN;
				_reset_stringstream(ermsg);
				ermsg << "ImageJ cannot open TIFF files " << "compressed in this fashion (" << value << ")";
				error(ermsg.str().c_str());
			}
			break;
		case SOFTWARE: case DATE_TIME: case HOST_COMPUTER: case ARTEST:
			if (ifdCount==1) {
				char *bytes = new char[count];
				if(getString(bytes, count, lvalue)){
					char name[NAME_SIZE_MAX];
					getName(name, tag);
					saveMetadata(name, bytes);
				}
				delete [] bytes;
			}
			break;
		case PREDICTOR:
			if (value==2 && fi.compression==FileInfo::LZW)
				fi.compression = FileInfo::LZW_WITH_DIFFERENCING;
			break;
		case COLOR_MAP: 
			if (count==768 && fi.fileType==FileInfo::GRAY8)
				getColorMap(lvalue, fi);
			break;
		case TILE_WIDTH:
			error("ImageJ cannot open tiled TIFFs");
			break;
		case SAMPLE_FORMAT:
			if (fi.fileType==FileInfo::GRAY32_INT && value==FLOATING_POINT)
				fi.fileType = FileInfo::GRAY32_FLOAT;
			if (fi.fileType==FileInfo::GRAY16_UNSIGNED) {
				if (value==SIGNED)
					fi.fileType = FileInfo::GRAY16_SIGNED;
				if (value==FLOATING_POINT)
					error("ImageJ cannot open 16-bit float TIFFs");
			}
			break;
		case JPEG_TABLES:
			if (fi.compression==FileInfo::JPEG)
				error("Cannot open JPEG-compressed TIFFs with separate tables");
			break;
		case IMAGE_DESCRIPTION: 
			if (ifdCount==1){
				char *s = new char[count];
				if(getString(s, count, lvalue)){
					saveImageDescription(s, fi);
				}
				delete [] s;
			}
			break;
		case ORIENTATION:
			fi.nImages = 0; // file not created by ImageJ so look at all the IFDs
			break;
		case METAMORPH1: case METAMORPH2:
			if ((string(name).find(".STK")>0||string(name).find(".stk")>0) && fi.compression==FileInfo::COMPRESSION_NONE) {
				if (tag==METAMORPH2)
					fi.nImages=count;
				else
					fi.nImages=9999;
			}
			break;
		case IPLAB: 
			fi.nImages=value;
			break;
		case NIH_IMAGE_HDR: 
			if (count==256)
				decodeNIHImageHeader(value, fi);
			break;
		case META_DATA_BYTE_COUNTS:
			saveLoc = in->tellg();
			in->seekg(lvalue, in->beg);
			SAFE_DELETE_ARRAY(metaDataCounts);
			metaDataCounts = new int[count];
			for (int c=0; c<count; c++)
				metaDataCounts[c] = getInt();
			nMetaDataCounts = count;
			in->seekg(saveLoc, in->beg);
			break;
		case META_DATA: 
			getMetaData(value, fi);
			break;
		default:
			if (tag>10000 && tag<32768 && ifdCount>1)
				return false;
		}
	}
	fi.fileFormat = FileInfo::TIFF;
	SAFE_DELETE_ARRAY(fi.fileName);
	_stringNewAndCopy(fi.fileName, name);
	SAFE_DELETE_ARRAY(fi.directory);
	_stringNewAndCopy(fi.directory, directory);
	if (url != NULL){
		SAFE_DELETE_ARRAY(fi.url);
		_stringNewAndCopy(fi.url, url);
	}

	return true;
}


void TiffDecoder::getMetaData(long long loc, FileInfo &fi) throw(ios_base::failure){
	if (metaDataCounts==NULL || nMetaDataCounts==0)
		return;
	int maxTypes = 10;

	long long saveLoc = in->tellg();
	in->seekg(loc, in->beg);

	int n = nMetaDataCounts;

	int hdrSize = metaDataCounts[0];
	if (hdrSize<12 || hdrSize>804){in->seekg(saveLoc, in->beg); return;}

	int magicNumber = getInt();
	if (magicNumber!=MAGIC_NUMBER){in->seekg(saveLoc, in->beg); return;}
	
	int nTypes = (hdrSize-4)/8;
	int *types =  new int[nTypes];
	int *counts = new int[nTypes];

	string s;

	if (debugMode){
		s = dInfo;
		s += "Metadata:\n";
	}
	
	int extraMetaDataEntries = 0;
	for (int i=0; i<nTypes; i++) {
		types[i] = getInt();
		counts[i] = getInt();
		if (types[i]<0xffffff)
			extraMetaDataEntries += counts[i];
		if (debugMode) {
			string id("");
			if (types[i]==INFO) id = " (Info property)";
			if (types[i]==LABELS) id = " (slice labels)";
			if (types[i]==RANGES) id = " (display ranges)";
			if (types[i]==LUTS) id = " (luts)";
			if (types[i]==ROI) id = " (roi)";
			if (types[i]==OVERLAY) id = " (overlay)";
			stringstream ss;
			ss << "   " << i << " " << std::hex << types[i] << std::dec << " " << counts[i] << id << "\n";
			s += ss.str();
		}
	}
	if(extraMetaDataEntries > 0){
		SAFE_DELETE_ARRAY(fi.metaDataTypes);
		fi.metaDataTypes = new int[extraMetaDataEntries];
		SAFE_DELETE_2D_ARRAY(fi.metaData, fi.extraMetaDataEntries);
		fi.metaData = new char *[extraMetaDataEntries];
	}
	fi.extraMetaDataEntries = extraMetaDataEntries;
	int start = 1;
	int eMDindex = 0;
	for (int i=0; i<nTypes; i++) {
		if (types[i]==INFO)
			getInfoProperty(start, fi);
		else if (types[i]==LABELS)
			getSliceLabels(start, start+counts[i]-1, fi);
		else if (types[i]==RANGES)
			getDisplayRanges(start, fi);
		else if (types[i]==LUTS)
			getLuts(start, start+counts[i]-1, fi);
		else if (types[i]==ROI)
			getRoi(start, fi);
		else if (types[i]==OVERLAY)
			getOverlay(start, start+counts[i]-1, fi);
		else if (types[i]<0xffffff) {
			for (int j=start; j<start+counts[i]; j++) { 
				int len = metaDataCounts[j]; 
				fi.metaData[eMDindex] = new char[len];
				in->read(fi.metaData[eMDindex], len);
				fi.metaDataTypes[eMDindex] = types[i]; 
				eMDindex++;
			} 
		} else
			skipUnknownType(start, start+counts[i]-1);
		start += counts[i];
	}
	in->seekg(saveLoc, in->beg);

	if (debugMode){
		SAFE_DELETE_ARRAY(dInfo);
		_stringNewAndCopy(dInfo, s.c_str());
	}

	delete [] types;
	delete [] counts;
}

void TiffDecoder::getInfoProperty(int first, FileInfo &fi) throw(ios_base::failure){
	int len = metaDataCounts[first];
	byte* buffer = new byte[len];
	in->read((char *)buffer, len);
	len /= 2;
	char *chars = new char[len+1];
	if (littleEndian) {
		for (int j=0, k=0; j<len; j++)
			chars[j] = (char)(buffer[k++]&255 + ((buffer[k++]&255)<<8));
	} else {
		for (int j=0, k=0; j<len; j++)
			chars[j] = (char)(((buffer[k++]&255)<<8) + buffer[k++]&255);
	}
	chars[len] = '\0';
	SAFE_DELETE_ARRAY(fi.info);
	_stringNewAndCopy(fi.info, chars);

	delete [] buffer;
	delete [] chars;
}

void TiffDecoder::getSliceLabels(int first, int last, FileInfo &fi) throw(ios_base::failure){
	if(last-first+1 <= 0)return;
	SAFE_DELETE_2D_ARRAY(fi.sliceLabels, fi.nSliceLabels);
	fi.sliceLabels = new char *[last-first+1];
	for(int i=0; i<last-first+1; i++)fi.sliceLabels[i] = NULL;
	fi.nSliceLabels = last-first+1;
	int index = 0;
	byte *buffer = new byte[metaDataCounts[first]];
	for (int i=first; i<=last; i++) {
		int len = metaDataCounts[i];
		if (len>0) {
			if (len>metaDataCounts[first]){
				delete [] buffer;
				buffer = new byte[len];
			}
			in->read((char *)buffer, len);
			len /= 2;
			char *chars = new char[len+1];
			if (littleEndian) {
				for (int j=0, k=0; j<len; j++)
					chars[j] = buffer[2*k++];//(char)(buffer[k++]&255 + ((buffer[k++]&255)<<8));
			} else {
				for (int j=0, k=0; j<len; j++)
					chars[j] = buffer[2*(k++)+1];//(char)(((buffer[k++]&255)<<8) + buffer[k++]&255);
			}
			chars[len] = '\0';
			_stringNewAndCopy(fi.sliceLabels[index], chars);
			//if(fi.sliceLabels[index] && index < 10)cout << index << ": " << fi.sliceLabels[index] << endl;
			index++;
			delete [] chars;
		} else fi.sliceLabels[index++] = NULL;
	}
	delete [] buffer;
}

void TiffDecoder::getDisplayRanges(int first, FileInfo &fi) throw(ios_base::failure){
	int n = metaDataCounts[first]/8;
	fi.nDisplayRanges = n;
	SAFE_DELETE_ARRAY(fi.displayRanges);
	fi.displayRanges = new double[n];
	for (int i=0; i<n; i++)
		fi.displayRanges[i] = readDouble();
}

void TiffDecoder::getLuts(int first, int last, FileInfo &fi) throw(ios_base::failure){
	if(last-first+1 <= 0)return;
	SAFE_DELETE_2D_ARRAY(fi.channelLuts, fi.nChannelLuts);
	SAFE_DELETE_ARRAY(fi.nLuts);
	fi.channelLuts = new byte* [last-first+1];
	fi.nLuts = new int[last-first+1];
	fi.nChannelLuts = last-first+1;
	int index = 0;
	for (int i=first; i<=last; i++) {
		int len = metaDataCounts[i];
		fi.channelLuts[index] = new byte[len];
		fi.nLuts[index] = len;
		in->read((char *)fi.channelLuts[index], len);
		index++;
	}
	
}

void TiffDecoder::getRoi(int first, FileInfo &fi) throw(ios_base::failure){
	SAFE_DELETE_ARRAY(fi.roi);
	int len = metaDataCounts[first];
	fi.roi = new byte[len]; 
	in->read((char *)fi.roi, len);
}

void TiffDecoder::getOverlay(int first, int last, FileInfo &fi) throw(ios_base::failure){
	SAFE_DELETE_2D_ARRAY(fi.overlay, fi.nOverlay);
	SAFE_DELETE_ARRAY(fi.overlaySize);
	fi.overlay = new byte *[last-first+1];
	fi.nOverlay = last-first+1;
	fi.overlaySize = new int[last-first+1];
	int index = 0;
	for (int i=first; i<=last; i++) {
		int len = metaDataCounts[i];
		fi.overlaySize[index] = len;
		fi.overlay[index] = new byte[len];
		in->read((char *)fi.overlay[index], len);
		index++;
	}
}


void TiffDecoder::error(const char *message) throw(ios_base::failure){
	if (in != NULL) in->close();
	throw  ios_base::failure("TiffDecoder::error");
}

void TiffDecoder::skipUnknownType(int first, int last) throw(ios_base::failure){
	byte *buffer = new byte[metaDataCounts[first]];
	for (int i=first; i<=last; i++) {
		int len = metaDataCounts[i];
		if (len>metaDataCounts[first]){
			delete [] buffer;
			buffer = new byte[len];
		}
		in->read((char *)buffer, len);
	}

	delete [] buffer;
}

void TiffDecoder::enableDebugging(){
	debugMode = true;
}

//imageJが生成したTiffでは1つのIFDしか読み込まないためFileInfoは1つだけになる
//通常のTiffについては2次元画像1枚につき1つのIFDを読み込みそれぞれに対応するFileInfoを生成する
//
//nImagesはFileInfoが対応する2次元画像の数でありこれが2以上になるのは基本的にImageJのTiffのみ
//またImageJが生成するTiffは画像データが1つの連続したデータブロックになっているためgapBetweenImagesは0となる
//**通常のTiffでは2次元画像1枚につき1つのIFDを用いるのでgapBetweenImagesは不要であり**//
//**画像データの抽出にはFileInfoのoffsetを用いる									**//
bool TiffDecoder::loadTiffInfo() throw(ios_base::failure){
	long long ifdOffset;
	stringstream ss;
	ss << directory << "\\" << name;
	
	if(in == NULL) in = new fstream(ss.str().c_str(), ios::binary|ios::in);
	if(in->fail()){
		//cout << "File Open Error" << endl;
		return false;
	}
	vector<FileInfo> fivec;
	ifdOffset = OpenImageFileHeader();
	if (ifdOffset<0LL) {
		in->close();
		//cout << "OpenImageFileHeader Error" << endl;
		return false;
	}
	if (debugMode){
		_reset_stringstream(ss);
		ss << "\n" << name << ": opening\n";
		SAFE_DELETE_ARRAY(dInfo);
		_stringNewAndCopy(dInfo, ss.str().c_str());
	}
	while (ifdOffset>0LL) {
		in->seekg(ifdOffset, in->beg);
		FileInfo fi;
		bool isOpened = OpenIFD(fi);
		if (isOpened) {
			fivec.push_back(fi);
			ifdOffset = ((long long)getInt())&0xffffffffLL;
		} else
			ifdOffset = 0LL;
		if (debugMode && ifdCount<10){
			_reset_stringstream(ss);
			ss << dInfo << "  nextIFD=" << ifdOffset << "\n";
			SAFE_DELETE_ARRAY(dInfo);
			_stringNewAndCopy(dInfo, ss.str().c_str());
		}
		if (isOpened) {
			if (fi.nImages>1){ // ignore extra IFDs in ImageJ and NIH Image stacks
				ifdOffset = 0LL;
				fi.gapBetweenImages = 0;
			}
		}
	}
	if (fivec.size()==0) {
		in->close();
		return false;
	} else {
		SAFE_DELETE_ARRAY(fis);
		nFInfo = fivec.size();
		fis = new FileInfo[fivec.size()];
		for(int i=0; i<fivec.size(); i++)fis[i] = fivec[i];
		
		if (debugMode){
			SAFE_DELETE_ARRAY(fis[0].debugInfo);
			_stringNewAndCopy(fis[0].debugInfo, dInfo);
		}

		if (url!=NULL) {
			in->seekg(0, in->beg);
			fis[0].inputStream = in;
		} else{
			in->close();
		}

		if (fis[0].info==NULL)
			_stringNewAndCopy(fis[0].info, tiffMetadata);
		if (debugMode) {
			size_t n = fivec.size();
			char gapinfo[GAP_INFO_SIZE_MAX];
			getGapInfo(gapinfo, fis, n);
			_reset_stringstream(ss);
			ss << fis[0].debugInfo;
			ss << "number of IFDs: " << n << "\n";
			ss << "offset to first image: " << fis[0].getOffset() << "\n";
			ss << "gap between images: " << gapinfo << "\n";
			ss << "little-endian byte order: " << fis[0].intelByteOrder << "\n\n";

			if(fis[0].info)ss << "info:\n" << fis[0].info << "\n\n";
			
			for(int i=0; i<fis[0].nChannelLuts; i++){
				ss << "ChannelLut " << i << "\n"; 
				for(int j=255; j<fis[0].nLuts[i]; j+=256){
					ss << (int)(fis[0].channelLuts[i][j]) << " ";
				}
				ss << "\n";
			}
			SAFE_DELETE_ARRAY(fis[0].debugInfo);
			_stringNewAndCopy(fis[0].debugInfo, ss.str().c_str());
		}
		return true;
	}
	in->close();
	delete in;
	in = NULL;
}

void TiffDecoder::getGapInfo(char *s, FileInfo *fi, size_t length) {
	stringstream ss;

	if (length<2){
		strcpy_s(s, GAP_INFO_SIZE_MAX, "0");
		return;
	}
	long minGap = LONG_MAX;
	long maxGap = -LONG_MAX;
	for (int i=1; i<length; i++) {
		long gap = fi[i].getOffset()-fi[i-1].getOffset();
		if (gap<minGap) minGap = gap;
		if (gap>maxGap) maxGap = gap;
	}
	long imageSize = fi[0].width*fi[0].height*fi[0].getBytesPerPixel();
	minGap -= imageSize;
	maxGap -= imageSize;
	if (minGap==maxGap){
		ss << minGap;
		strcpy_s(s, GAP_INFO_SIZE_MAX, ss.str().c_str());
	}
	else {
		ss << "varies (" << minGap << " to " << maxGap << ")";
		strcpy_s(s, GAP_INFO_SIZE_MAX, ss.str().c_str());
	}
}

size_t TiffDecoder::getFileInfoNum(){
	return nFInfo;
}

bool TiffDecoder::getFileInfo(FileInfo* &dst){
	if(dst == NULL)return false;
	for(int i = 0; i < nFInfo; i++)dst[i] = fis[i];
	return true;
}

void TiffDecoder::tifdec_InvalidParameterHandler(const wchar_t* expression,
												 const wchar_t* function, 
												 const wchar_t* file, 
												 unsigned int line, 
												 uintptr_t pReserved)
{
	wprintf(L"Invalid parameter detected in function %s."
			L" File: %s Line: %d\n", function, file, line);
	wprintf(L"Expression: %s\n", expression);
}

void FileInfo::init_vals(){
	fileFormat			 = UNKNOWN;
	fileType			 = GRAY8;
	nImages				 = 1;
	compression			 = COMPRESSION_NONE;
	samplesPerPixel		 = 1;
	offset				 = 0;
	pixelWidth			 = 1.0;
	pixelHeight			 = 1.0;
	pixelDepth			 = 1.0;
	extraMetaDataEntries = 0;
	nSliceLabels		 = 0;
	nChannelLuts		 = 0;
	nOverlay			 = 0;
	nDisplayRanges		 = 0;
	nStripOffsets		 = 0;
	nStripLengths		 = 0;
	lutSize				 = 0;
	nCoefficients		 = 0;
	nRoi				 = 0;
	longOffset			 = 0LL;
	gapBetweenImages	 = 0;
}

void FileInfo::init(){
	
	init_vals();

	fileName		= NULL;
	directory		= NULL;
	url				= NULL;
	stripOffsets	= NULL;
	stripLengths	= NULL;
	reds			= NULL;
	greens			= NULL;
	blues			= NULL;
	debugInfo		= NULL;
	sliceLabels		= NULL;
	info			= NULL;
	inputStream		= NULL;
	unit			= NULL;
	coefficients	= NULL;
	valueUnit		= NULL;
	description		= NULL;
	metaDataTypes	= NULL;
	metaData		= NULL;
	displayRanges	= NULL;
	nLuts			= NULL;
	channelLuts		= NULL;
	roi				= NULL;
	overlaySize		= NULL;
	overlay			= NULL;
	openNextDir		= NULL;
	openNextName	= NULL;
	
}

void FileInfo::clear(){
	
	SAFE_DELETE_ARRAY(fileName);
	SAFE_DELETE_ARRAY(directory);
	SAFE_DELETE_ARRAY(url);
	SAFE_DELETE_ARRAY(stripOffsets);
	SAFE_DELETE_ARRAY(stripLengths);
	SAFE_DELETE_ARRAY(reds);
	SAFE_DELETE_ARRAY(greens);
	SAFE_DELETE_ARRAY(blues);
	SAFE_DELETE_ARRAY(debugInfo);
	SAFE_DELETE_2D_ARRAY(sliceLabels, nSliceLabels);
	SAFE_DELETE_ARRAY(info);
	SAFE_DELETE_ARRAY(inputStream);
	SAFE_DELETE_ARRAY(unit);
	SAFE_DELETE_ARRAY(coefficients);
	SAFE_DELETE_ARRAY(valueUnit);
	SAFE_DELETE_ARRAY(description);
	SAFE_DELETE_ARRAY(metaDataTypes);
	SAFE_DELETE_2D_ARRAY(metaData, extraMetaDataEntries);
	SAFE_DELETE_ARRAY(displayRanges);
	SAFE_DELETE_ARRAY(nLuts);
	SAFE_DELETE_2D_ARRAY(channelLuts, nChannelLuts);
	SAFE_DELETE_ARRAY(roi);
	SAFE_DELETE_ARRAY(overlaySize);
	SAFE_DELETE_2D_ARRAY(overlay, nOverlay);
	SAFE_DELETE_ARRAY(openNextDir);
	SAFE_DELETE_ARRAY(openNextName);

	init_vals();

}

void FileInfo::copy_body(const FileInfo& fi){
	fileFormat = fi.fileFormat;
	fileType = fi.fileType;
	_stringNewAndCopy(fileName, fi.fileName);
	_stringNewAndCopy(directory, fi.directory);
	_stringNewAndCopy(url, fi.url);
	width = fi.width;
	height = fi.height;
	offset = fi.offset;
	nImages = fi.nImages;
	gapBetweenImages = fi.gapBetweenImages;
	whiteIsZero = fi.whiteIsZero;
	intelByteOrder = fi.intelByteOrder;
	compression = fi.compression;
	nStripOffsets = fi.nStripOffsets;
	_arrayNewAndCopy(stripOffsets, fi.nStripOffsets, fi.stripOffsets);
	nStripLengths = fi.nStripLengths;
	_arrayNewAndCopy(stripLengths, fi.nStripLengths, fi.stripLengths);
	rowsPerStrip = fi.rowsPerStrip;
	lutSize = fi.lutSize;
	_arrayNewAndCopy(reds, fi.lutSize, fi.reds);
	_arrayNewAndCopy(greens, fi.lutSize, fi.greens);
	_arrayNewAndCopy(blues, fi.lutSize, fi.blues);
	_stringNewAndCopy(debugInfo, fi.debugInfo);
	nSliceLabels = fi.nSliceLabels;
	_2DstringNewAndCopy(sliceLabels, fi.sliceLabels, fi.nSliceLabels);
	_stringNewAndCopy(info, fi.info);
	pixelWidth = fi.pixelWidth;
	pixelHeight = fi.pixelHeight;
	pixelDepth = fi.pixelDepth;
	_stringNewAndCopy(unit, fi.unit);
	calibrationFunction = fi.calibrationFunction;
	nCoefficients = fi.nCoefficients;
	_arrayNewAndCopy(coefficients, fi.nCoefficients, fi.coefficients);
	_stringNewAndCopy(valueUnit, fi.valueUnit);
	frameInterval = fi.frameInterval;
	_stringNewAndCopy(description, fi.description);
	longOffset = fi.longOffset;
	extraMetaDataEntries = fi.extraMetaDataEntries;
	_arrayNewAndCopy(metaDataTypes, fi.extraMetaDataEntries, metaDataTypes);
	_2DstringNewAndCopy(metaData, fi.metaData, fi.extraMetaDataEntries);
	nDisplayRanges = fi.nDisplayRanges;
	_arrayNewAndCopy(displayRanges, nDisplayRanges, fi.displayRanges);
	nChannelLuts = fi.nChannelLuts;
	_arrayNewAndCopy(nLuts, fi.nChannelLuts, fi.nLuts);
	_2DarrayNewAndCopy(channelLuts, fi.nChannelLuts, fi.channelLuts, fi.nLuts);
	nRoi = fi.nRoi;
	_arrayNewAndCopy(roi, nRoi, fi.roi);
	nOverlay = fi.nOverlay;
	_arrayNewAndCopy(overlaySize, fi.nOverlay, fi.overlaySize);
	_2DarrayNewAndCopy(overlay, fi.nOverlay, fi.overlay, fi.overlaySize);
	samplesPerPixel = fi.samplesPerPixel;
	_stringNewAndCopy(openNextDir, fi.openNextDir);
	_stringNewAndCopy(openNextName, fi.openNextName);

}

FileInfo::FileInfo(){
	init();
}

FileInfo::FileInfo(const FileInfo& fi){
	
	init();
	copy_body(fi);
}

FileInfo::~FileInfo(){
	clear();
}

FileInfo& FileInfo::copy(const FileInfo& fi){
	if(this == &fi)return *this;
	clear();
	
	copy_body(fi);

	return *this;
}

FileInfo& FileInfo::operator =(const FileInfo& fi){
	return copy(fi);
}

/** Returns the offset as a long. */
inline long long FileInfo::getOffset() {
	return longOffset>0LL ? longOffset : ((long long)offset)&0xffffffffLL;
}

/** Returns the number of bytes used per pixel. */
int FileInfo::getBytesPerPixel() {
	switch (fileType) {
	case GRAY8: case COLOR8: case BITMAP: return 1;
	case GRAY16_SIGNED: case GRAY16_UNSIGNED: return 2;
	case GRAY32_INT: case GRAY32_UNSIGNED: case GRAY32_FLOAT: case ARGB: case GRAY24_UNSIGNED: case BARG: case ABGR: case CMYK: return 4;
	case RGB: case RGB_PLANAR: case BGR: return 3;
	case RGB48: case RGB48_PLANAR: return 6;
	case GRAY64_FLOAT : return 8;
	default: return 0;
	}
}

void FileInfo::to_c_string(char *s){
	char tname[TYPE_NAME_SIZE_MAX];
	getType(tname);
	stringstream ss;
	ss	<< "name=" << (fileName ? fileName : "NULL")
		<< ", dir=" << (directory ? directory : "NULL")
		<< ", url=" << (url ? url : "NULL") 
		<< ", width=" << width
		<< ", height=" << height
		<< ", nImages=" << nImages
		<< ", type=" << tname
		<< ", format=" << fileFormat
		<< ", offset=" << getOffset()
		<< ", whiteZero=" << (whiteIsZero?"t":"f")
		<< ", Intel=" << (intelByteOrder?"t":"f")
		<< ", lutSize=" << lutSize
		<< ", comp=" << compression
		<< ", ranges=" << (displayRanges!=NULL ? nDisplayRanges/2 : 0)
		<< ", samples=" << samplesPerPixel;
	strcpy_s(s, ss.str().length()+1, ss.str().c_str());
}


void FileInfo::getType(char *tname) {
	switch (fileType) {
	case GRAY8: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "byte"); return;
	case GRAY16_SIGNED: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "short"); return;
	case GRAY16_UNSIGNED: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "ushort"); return;
	case GRAY32_INT: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "int"); return;
	case GRAY32_UNSIGNED: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "uint"); return;
	case GRAY32_FLOAT: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "float"); return;
	case COLOR8: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "byte+lut"); return;
	case RGB: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "RGB"); return;
	case RGB_PLANAR: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "RGB(p)"); return;
	case RGB48: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "RGB48"); return;
	case BITMAP: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "bitmap"); return;
	case ARGB: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "ARGB"); return;
	case ABGR: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "ABGR"); return;
	case BGR: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "BGR"); return;
	case BARG: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "BARG"); return;
	case CMYK: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "CMYK"); return;
	case GRAY64_FLOAT: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "double"); return;
	case RGB48_PLANAR: strcpy_s(tname, TYPE_NAME_SIZE_MAX, "RGB48(p)"); return;
	default: strcpy_s(tname, TYPE_NAME_SIZE_MAX, ""); return;
	}
}

unsigned int FileInfo::getImageSize(){
	int imageSize = width*height*getBytesPerPixel();
	if (fileType==FileInfo::GRAY12_UNSIGNED) {
		imageSize = (int)(width*height*1.5);
		if ((imageSize&1)==1) imageSize++; // add 1 if odd
	} if (fileType==FileInfo::BITMAP) {
		int scan=(int)ceil(width/8.0);
		imageSize = scan*height;
	}
	return imageSize;
}

ImageReader::ImageReader(){
	isEmpty = true;
}

void ImageReader::clear(){
	SAFE_DELETE(fi);
	isEmpty = true;
}

void ImageReader::init(const FileInfo &src_info){
	
	fi = new FileInfo;
	*fi = src_info;
	
	width = fi->width;
	height = fi->height;
	skipCount = fi->getOffset();
	eofErrorCount = 0;

	debugMode = false;
	isEmpty = false;

	return;
}	

ImageReader::ImageReader(const FileInfo &src_info){
	init(src_info);
}

ImageReader::~ImageReader(){
	clear();
}

bool ImageReader::empty(){
	return isEmpty;
}

void ImageReader::eofError(){
	eofErrorCount++;
}

void ImageReader::setDebugMode(bool val){
	debugMode = val;
}

void ImageReader::read8bitImage(std::fstream &in, tiffhndl::byte* data){
	if(isEmpty)return;
	if (fi->compression>FileInfo::COMPRESSION_NONE)
		return;//readCompressed8bitImage(in);
	// assume contiguous strips
	int count, actuallyRead;
	int totalRead = 0;
	while (totalRead<byteCount) {
		if (totalRead+bufferSize>byteCount)
			count = (int)(byteCount-totalRead);
		else
			count = bufferSize;
		actuallyRead = in.read((char *)data + totalRead, count).gcount();
		if (actuallyRead == 0) {eofError(); break;}
		totalRead += actuallyRead;
	}
	return;
}

// Reads a 16-bit image. Signed pixels are converted to unsigned by adding 32768. //
void ImageReader::read16bitImage(std::fstream &in, short* data){
	if(isEmpty)return;
	if (fi->compression>FileInfo::COMPRESSION_NONE || (fi->stripOffsets!=NULL && fi->nStripOffsets>1))
		return;// readCompressed16bitImage(in);
	int pixelsRead;
	byte *buffer = new byte[bufferSize];
	long totalRead = 0L;
	int base = 0;
	int count, value;
	int bufferCount;

	while (totalRead<byteCount) {
		if ((totalRead+bufferSize)>byteCount)
			bufferSize = (int)(byteCount-totalRead);
		bufferCount = 0;
		while (bufferCount<bufferSize) { // fill the buffer
			count = in.read((char *)buffer + bufferCount, bufferSize-bufferCount).gcount();
			if (count == 0) {
				if (bufferCount>0)
					for (int i=bufferCount; i<bufferSize; i++) buffer[i] = 0;
				totalRead = byteCount;
				eofError();
				break;
			}
			bufferCount += count;
		}
		totalRead += bufferSize;
		pixelsRead = bufferSize/bytesPerPixel;
		if (fi->intelByteOrder) {
			if (fi->fileType==FileInfo::GRAY16_SIGNED)
				for (int i=base,j=0; i<(base+pixelsRead); i++,j+=2)
					//元のコードでは32768を加えて非符号化をしていたがこのプログラムではそのまま格納している
					data[i] = (short)((((buffer[j+1]&0xff)<<8) | (buffer[j]&0xff))+0);
			else
				for (int i=base,j=0; i<(base+pixelsRead); i++,j+=2)
					data[i] = (short)(((buffer[j+1]&0xff)<<8) | (buffer[j]&0xff));
		} else {
			if (fi->fileType==FileInfo::GRAY16_SIGNED)
				for (int i=base,j=0; i<(base+pixelsRead); i++,j+=2)
					//元のコードでは32768を加えて非符号化をしていたがこのプログラムではそのまま格納している
					data[i] = (short)((((buffer[j]&0xff)<<8) | (buffer[j+1]&0xff))+0);
			else
				for (int i=base,j=0; i<(base+pixelsRead); i++,j+=2)
					data[i] = (short)(((buffer[j]&0xff)<<8) | (buffer[j+1]&0xff));
		}
		base += pixelsRead;
	}
	delete [] buffer;
	return;
}

void ImageReader::read32bitImage(fstream &in, float* data){
	if(isEmpty)return;
	if (fi->compression > FileInfo::COMPRESSION_NONE)
		return;// readCompressed32bitImage(in);
	int pixelsRead;
	byte *buffer = new byte[bufferSize];
	long totalRead = 0L;
	int base = 0;
	int count, value;
	int bufferCount;
	int tmp;

	//cout << "read32bitImage" << endl;

	while (totalRead < byteCount) {
		if ((totalRead+bufferSize) > byteCount)
			bufferSize = (int)(byteCount-totalRead);
		bufferCount = 0;
		while (bufferCount<bufferSize) { // fill the buffer
			count = in.read((char *)buffer + bufferCount, bufferSize-bufferCount).gcount();
			if (count == 0) {
				if (bufferCount>0)
					for (int i=bufferCount; i<bufferSize; i++) buffer[i] = 0;
				totalRead = byteCount;
				eofError();
				break;
			}
			bufferCount += count;
		}
		totalRead += bufferSize;
		pixelsRead = bufferSize/bytesPerPixel;
		int pmax = base+pixelsRead;
		if (pmax>nPixels) pmax = nPixels;
		int j = 0;
		if (fi->intelByteOrder){
			for (int i=base; i<pmax; i++) {
				tmp = (int)(((buffer[j+3]&0xff)<<24) | ((buffer[j+2]&0xff)<<16) | ((buffer[j+1]&0xff)<<8) | (buffer[j]&0xff));
				if (fi->fileType==FileInfo::GRAY32_FLOAT){
					Float_Int_Converter conv;
					conv.itr = tmp;
					data[i] = conv.f;
				}
				else if (fi->fileType==FileInfo::GRAY32_UNSIGNED)
					data[i] = (float)(tmp&0xffffffffL);
				else
					data[i] = tmp;
				j += 4;
			}
		}
		else{
			for (int i=base; i<pmax; i++) {
				tmp = (int)(((buffer[j]&0xff)<<24) | ((buffer[j+1]&0xff)<<16) | ((buffer[j+2]&0xff)<<8) | (buffer[j+3]&0xff));
				if (fi->fileType==FileInfo::GRAY32_FLOAT){
					Float_Int_Converter conv;
					conv.itr = tmp;
					data[i] = conv.f;
				}
				else if (fi->fileType==FileInfo::GRAY32_UNSIGNED)
					data[i] = (float)(tmp&0xffffffffL);
				else
					data[i] = tmp;
				j += 4;
			}
		}
		base += pixelsRead;
	}
	delete [] buffer;
	return;
}

void ImageReader::skip(std::fstream &in){
	if(isEmpty)return;
	if (skipCount>0) {
		unsigned long bytesRead = 0;
		int skipAttempts = 0;
		unsigned long long count;
		while (bytesRead<skipCount) {
			count = in.ignore(skipCount-bytesRead).gcount();
			skipAttempts++;
			if (in.eof() || skipAttempts>5) break;
			bytesRead += count;
			//IJ.log("skip: "+skipCount+" "+count+" "+bytesRead+" "+skipAttempts);
		}
	}
	byteCount = ((unsigned long long)width)*height*bytesPerPixel;
	if (fi->fileType==FileInfo::BITMAP) {
		unsigned int scan=width/8, pad = width%8;
		if (pad>0) scan++;
		byteCount = scan*height;
	}
	nPixels = width*height;
	bufferSize = (unsigned int)(byteCount/25ULL);
	if (bufferSize<8192)
		bufferSize = 8192;
	else
		bufferSize = (bufferSize/8192)*8192;
}

bool ImageReader::readPixels(std::fstream &in, tiffhndl::byte* data) {
	if(isEmpty)return false;
	
	try {
		switch (fi->fileType) {
		case FileInfo::GRAY8:
		case FileInfo::COLOR8:
			bytesPerPixel = 1;
			skip(in);
			read8bitImage(in, data);
			break;
		case FileInfo::GRAY16_SIGNED:
		case FileInfo::GRAY16_UNSIGNED:
			bytesPerPixel = 2;
			skip(in);
			read16bitImage(in, (short *)data);
			break;
		case FileInfo::GRAY32_INT:
		case FileInfo::GRAY32_UNSIGNED:
		case FileInfo::GRAY32_FLOAT:
			bytesPerPixel = 4;
			skip(in);
			read32bitImage(in, (float *)data);
			break;
/*		case FileInfo.GRAY64_FLOAT:
			bytesPerPixel = 8;
			skip(in);
			pixels = (Object)read64bitImage(in);
			break;
		case FileInfo.RGB:
		case FileInfo.BGR:
		case FileInfo.ARGB:
		case FileInfo.ABGR:
		case FileInfo.BARG:
		case FileInfo.CMYK:
			bytesPerPixel = fi->getBytesPerPixel();
			skip(in);
			pixels = (Object)readChunkyRGB(in);
			break;
		case FileInfo.RGB_PLANAR:
			bytesPerPixel = 3;
			skip(in);
			pixels = (Object)readPlanarRGB(in);
			break;
		case FileInfo.BITMAP:
			bytesPerPixel = 1;
			skip(in);
			pixels = (Object)read1bitImage(in);
			break;
		case FileInfo.RGB48:
			bytesPerPixel = 6;
			skip(in);
			pixels = (Object)readRGB48(in);
			break;
		case FileInfo.RGB48_PLANAR:
			bytesPerPixel = 2;
			skip(in);
			pixels = (Object)readRGB48Planar(in);
			break;
		case FileInfo.GRAY12_UNSIGNED:
			skip(in);
			short[] data = read12bitImage(in);
			pixels = (Object)data;
			break;
		case FileInfo.GRAY24_UNSIGNED:
			skip(in);
			pixels = (Object)read24bitImage(in);
			break;
*/		default:
			data = NULL;
			return false;
		}
	}
	catch (std::ios_base::failure) {
		return false;
	}
	return true;
}

//   Skips the specified number of bytes, then reads an image and 
//   returns the pixel array (byte, short, int or float). Returns
//   null if there was an IO exception. Does not close the InputStream.
bool ImageReader::readPixels(std::fstream &in, tiffhndl::byte* data, unsigned long long skipCount) {
	if(isEmpty)return false;

	this->skipCount = skipCount;
	readPixels(in, data);
	if (eofErrorCount>0)
		return false;
	else
		return true;
}

/*
public static ColorModel createGrayscaleColorModel(boolean invert) {
	byte[] rLUT = new byte[256];
	byte[] gLUT = new byte[256];
	byte[] bLUT = new byte[256];
	if (invert)
		for(int i=0; i<256; i++) {
			rLUT[255-i]=(byte)i;
			gLUT[255-i]=(byte)i;
			bLUT[255-i]=(byte)i;
		}
	else {
		for(int i=0; i<256; i++) {
			rLUT[i]=(byte)i;
			gLUT[i]=(byte)i;
			bLUT[i]=(byte)i;
		}
	}
	return(new IndexColorModel(8, 256, rLUT, gLUT, bLUT));
}

public ColorModel createColorModel(FileInfo fi) {
	if (fi.fileType==FileInfo.COLOR8 && fi.lutSize>0)
		return new IndexColorModel(8, fi.lutSize, fi.reds, fi.greens, fi.blues);
	else
		return LookUpTable.createGrayscaleColorModel(fi.whiteIsZero);
}
*/
/*
//Attempts to determine the image file type by looking for
//'magic numbers' and the file name extension.
int Opener::getFileType(String path) {
	if (openUsingPlugins && !path.endsWith(".txt") &&  !path.endsWith(".java"))
		return UNKNOWN;
	File file = new File(path);
	String name = file.getName();
	InputStream is;
	byte[] buf = new byte[132];
	try {
		is = new FileInputStream(file);
		is.read(buf, 0, 132);
		is.close();
	} catch (IOException e) {
		return UNKNOWN;
	}

	int b0=buf[0]&255, b1=buf[1]&255, b2=buf[2]&255, b3=buf[3]&255;
	//IJ.log("getFileType: "+ name+" "+b0+" "+b1+" "+b2+" "+b3);

	// Combined TIFF and DICOM created by GE Senographe scanners
	if (buf[128]==68 && buf[129]==73 && buf[130]==67 && buf[131]==77
		&& ((b0==73 && b1==73)||(b0==77 && b1==77)))
		return TIFF_AND_DICOM;

	// Big-endian TIFF ("MM")
	if (name.endsWith(".lsm"))
		return UNKNOWN; // The LSM  Reader plugin opens these files
	if (b0==73 && b1==73 && b2==42 && b3==0 && !(bioformats&&name.endsWith(".flex")))
		return TIFF;

	// Little-endian TIFF ("II")
	if (b0==77 && b1==77 && b2==0 && b3==42)
		return TIFF;

	// JPEG
	if (b0==255 && b1==216 && b2==255)
		return JPEG;

	// GIF ("GIF8")
	if (b0==71 && b1==73 && b2==70 && b3==56)
		return GIF;

	name = name.toLowerCase(Locale.US);

	// DICOM ("DICM" at offset 128)
	if (buf[128]==68 && buf[129]==73 && buf[130]==67 && buf[131]==77 || name.endsWith(".dcm")) {
		return DICOM;
	}

	// ACR/NEMA with first tag = 00002,00xx or 00008,00xx
	if ((b0==8||b0==2) && b1==0 && b3==0 && !name.endsWith(".spe") && !name.equals("fid"))  
		return DICOM;

	// PGM ("P1", "P4", "P2", "P5", "P3" or "P6")
	if (b0==80&&(b1==49||b1==52||b1==50||b1==53||b1==51||b1==54)&&(b2==10||b2==13||b2==32||b2==9))
		return PGM;

	// Lookup table
	if (name.endsWith(".lut"))
		return LUT;

	// PNG
	if (b0==137 && b1==80 && b2==78 && b3==71)
		return PNG;

	// ZIP containing a TIFF
	if (name.endsWith(".zip"))
		return ZIP;

	// FITS ("SIMP")
	if ((b0==83 && b1==73 && b2==77 && b3==80) || name.endsWith(".fts.gz") || name.endsWith(".fits.gz"))
		return FITS;

	// Java source file, text file or macro
	if (name.endsWith(".java") || name.endsWith(".txt") || name.endsWith(".ijm") || name.endsWith(".js")
		|| name.endsWith(".bsh") || name.endsWith(".py") || name.endsWith(".html"))
		return JAVA_OR_TEXT;

	// ImageJ, NIH Image, Scion Image for Windows ROI
	if (b0==73 && b1==111) // "Iout"
		return ROI;

	// ObjectJ project
	if ((b0=='o' && b1=='j' && b2=='j' && b3==0) || name.endsWith(".ojj") )
		return OJJ;

	// Results table (tab-delimited or comma-separated tabular text)
	if (name.endsWith(".xls") || name.endsWith(".csv")) 
		return TABLE;

	// AVI
	if (name.endsWith(".avi"))
		return AVI;

	// Text file
	boolean isText = true;
	for (int i=0; i<10; i++) {
		int c = buf[i]&255;
		if ((c<32&&c!=9&&c!=10&&c!=13) || c>126) {
			isText = false;
			break;
		}
	}
	if (isText)
		return TEXT;

	// BMP ("BM")
	if ((b0==66 && b1==77)||name.endsWith(".dib"))
		return BMP;

	// RAW
	if (name.endsWith(".raw"))
		return RAW;

	return UNKNOWN;
}
*/

FileOpener::FileOpener(){
	initVal();
}

void FileOpener::initVal(){
	fi = NULL;
	pixels = NULL;
	nInfo = 0;
	isEmpty = true;
	debugMode = false;
}

void FileOpener::clear(){
	SAFE_DELETE_ARRAY(fi);
	SAFE_DELETE_ARRAY(pixels);
	nInfo = 0;
	isEmpty = true;
	debugMode = false;
}

//コンストラクタからの呼び出し前には必ずinitValを呼び出しておく必要がある(free(NULL)が発生)
bool FileOpener::init(const FileInfo* src_info, int Info_num){
	if(!isEmpty)clear();

	if(Info_num <= 0){
		return false;
	}
	nInfo = Info_num;

	fi = new FileInfo[nInfo];
	for(int i = 0; i < nInfo; i++){
		fi[i] = src_info[i];
	}

	isEmpty = false;

	fcal = getCalibration(fi[0]);
	if(!fcal.loaded)return false;

	cout << "\nCalibration" << endl;
	cout << "channels: " << fcal.channels << endl;
	cout << "slices: " << fcal.slices << endl;
	cout << "frames: " << fcal.frames << endl;
	cout << "hyperstack: " << (fcal.hyperstack ? "true" : "false") << endl;
	cout << endl;

	if(!loadTiffStack()){
		cout <<	"loadTiffStack: Error" << endl;
		clear();
		return false;
	}
	
	return true;
}

//コンストラクタからの呼び出し前には必ずinitValを呼び出しておく必要がある(free(NULL)が発生)
bool FileOpener::open(const char *directory, const char *name){
			
	TiffDecoder td(directory, name);
	td.enableDebugging();
	if(!td.loadTiffInfo()){
		cout << "loadTiffInfo Error" << endl;
		return false;
	}
	FileInfo *fis = new FileInfo[td.getFileInfoNum()];
	td.getFileInfo(fis);
	cout << fis[0].debugInfo << endl;

	char fi_info[1000];
	fis[0].to_c_string(fi_info);
	if(fis[0].description)cout << fi_info << "\nimage_description:\n" << fis[0].description << "//spacing = PixelDepth\n" << endl;
	else cout << fi_info << "\n" << endl;
	cout << "FileInfo_NUM: " << td.getFileInfoNum() << endl;

	if( !init(fis, td.getFileInfoNum()) ){
		delete [] fis;
		return false;
	}
	
	delete [] fis;

	return true;
}

FileOpener::FileOpener(const char *path){
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];
	_splitpath_s(path, drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);
	string directory, name;
	
	directory = drive;
	directory += dir;
	
	name = fname;
	name += ext;

	cout << directory << endl;
	cout << fname << endl;

	initVal();

	open(directory.c_str(), name.c_str());
}

FileOpener::FileOpener(const char *directory, const char *name){
	initVal();

	open(directory, name);
}

FileOpener::FileOpener(const FileInfo* src_info, int info_num){
	initVal();

	init(src_info, info_num);
}

FileOpener::~FileOpener(){
	clear();
}

bool FileOpener::empty(){
	return isEmpty;
}

void FileOpener::setDebugMode(bool val){
	debugMode = val;
}

std::string stringReplace(const std::string& str, const std::string& from, const std::string& to) {
	std::string dst(str);
	std::string::size_type pos = 0;
	while(pos = dst.find(from, pos), pos != std::string::npos) {
		dst.replace(pos, from.length(), to);
		pos += to.length();
	}
	return dst;
}

void FileOpener::decodeDescriptionString(FileInfo &fi) {
	if (isEmpty)return;
	if (fi.description==NULL)return;

	string id(fi.description);

	if(id.length()<7)return;

	if(debugMode){
		cout << "Image Description: " << stringReplace(id, "\n", " ") << endl;
	}
	if(id.find("ImageJ") != 0)return;
	
	string dsUnit = getPropertyString(fi.description, "unit");
	if(!dsUnit.empty()){
		if(string(fi.unit) == "cm" && dsUnit == "um"){
			fi.pixelWidth *= 10000;
			fi.pixelHeight *= 10000;
		}
		SAFE_DELETE_ARRAY(fi.unit);
		_stringNewAndCopy(fi.unit, dsUnit.c_str());
	}
		
	int n = getPropertyInt(fi.description, "cf", -1);
	if (n != -1) fi.calibrationFunction = n;
	double c[5];
	int count = 0;
	for (int i=0; i<5; i++) {
		stringstream ss;
		ss << 'c' << i;
		n = getPropertyDouble(fi.description, ss.str().c_str(), DBL_MAX);
		if (n==DBL_MAX) break;
		c[i] = n;
		count++;
	}
	if (count>=2) {
		SAFE_DELETE_ARRAY(fi.coefficients);
		fi.coefficients = new double[count];
		fi.nCoefficients = count;
		for (int i=0; i<count; i++)
			fi.coefficients[i] = c[i];
	}
	string vUnit = getPropertyString(fi.description, "vunit");
	if(!vUnit.empty()){
		SAFE_DELETE_ARRAY(fi.valueUnit);
		_stringNewAndCopy(fi.valueUnit, vUnit.c_str());
	}
	
	n = getPropertyInt(fi.description, "images", -1);
	if (n > 1)
		fi.nImages = n;
	double spacing = getPropertyDouble(fi.description, "spacing", 0.0);
	if (spacing != 0.0) {
		if (spacing < 0) spacing = -spacing;
		fi.pixelDepth = spacing;
	}

	string name = getPropertyString(fi.description, "name");
	if (!name.empty()){
		SAFE_DELETE_ARRAY(fi.fileName);
		_stringNewAndCopy(fi.fileName, name.c_str());
	}

	return;
}

Calibration FileOpener::getCalibration(FileInfo &fi){
	
//	if (fi.fileType==FileInfo.GRAY16_SIGNED) {
//		if (IJ.debugMode) IJ.log("16-bit signed");
//		imp.getLocalCalibration().setSigned16BitCalibration();
//	}
	Calibration cal;
	cal.calibrated = false;
	cal.loaded = false;

	if (isEmpty)return cal;
	
	decodeDescriptionString(fi);

	cal.width = fi.width;
	cal.height = fi.height;

	if (fi.pixelWidth>0.0 && fi.unit!=NULL) {
		cal.pixelWidth = fi.pixelWidth;
		cal.pixelHeight = fi.pixelHeight;
		cal.pixelDepth = fi.pixelDepth;
		//cal.setUnit(fi.unit);
		cal.calibrated = true;
	}
	else{
		cal.pixelWidth = 0.0;
		cal.pixelHeight = 0.0;
		cal.pixelDepth = 0.0;
	}
	
//	if (fi.valueUnit!=null) {
//		int f = fi.calibrationFunction;
//		if ((f>=Calibration.STRAIGHT_LINE && f<=Calibration.RODBARD2 &&
//			fi.coefficients!=null)
//			|| f==Calibration.UNCALIBRATED_OD) {
//				boolean zeroClip = props!=null && props.getProperty("zeroclip",
//					"false").equals("true");    
//				cal.setFunction(f, fi.coefficients, fi.valueUnit, zeroClip);
//				calibrated = true;
//		}
//	}
//	
//	if (calibrated)
//		checkForCalibrationConflict(imp, cal);

	if (fi.frameInterval!=0.0)
		cal.frameInterval = fi.frameInterval;

	cal.xOrigin = getPropertyDouble(fi.description,"xorigin");
	cal.yOrigin = getPropertyDouble(fi.description,"yorigin");
	cal.zOrigin = getPropertyDouble(fi.description,"zorigin");
	
	//cal.info = props.getProperty("info");
	
	cal.fps = getPropertyDouble(fi.description,"fps");
	cal.loop = getPropertyBool(fi.description, "loop");
	cal.frameInterval = getPropertyDouble(fi.description,"finterval");
	//cal.setTimeUnit(props.getProperty("tunit", "sec"));
	
	/*
	double displayMin = getDouble(props,"min");
	double displayMax = getDouble(props,"max");
	if (!(displayMin==0.0&&displayMax==0.0)) {
		int type = imp.getType();
		ImageProcessor ip = imp.getProcessor();
		if (type==ImagePlus.GRAY8 || type==ImagePlus.COLOR_256)
			ip.setMinAndMax(displayMin, displayMax);
		else if (type==ImagePlus.GRAY16 || type==ImagePlus.GRAY32) {
			if (ip.getMin()!=displayMin || ip.getMax()!=displayMax)
				ip.setMinAndMax(displayMin, displayMax);
		}
	}
	*/
	
	int stackSize = fi.nImages;
	if (stackSize>1) {
		cal.channels = getPropertyInt(fi.description,"channels");
		cal.slices	 = getPropertyInt(fi.description,"slices");
		cal.frames	 = getPropertyInt(fi.description,"frames");
		if (cal.channels==0) cal.channels = 1;
		if (cal.slices==0) cal.slices = 1;
		if (cal.frames==0) cal.frames = 1;
		//IJ.log("setCalibration: "+channels+"  "+slices+"  "+frames);
		if (cal.channels*cal.slices*cal.frames==stackSize) {
			cal.hyperstack = getPropertyBool(fi.description, "hyperstack");
		}
		else{
			cal.hyperstack = false;
			cal.slices = stackSize;
			cal.channels = 1;
			cal.frames = 1;
		}
	}
	else{
		cal.channels = 1;
		cal.slices = nInfo;//RGBの場合は不適 要修正
		cal.frames = 1;
	}

	cal.loaded = true;

	return cal;
}

Calibration FileOpener::getImageCalibration(){
	return fcal; 
}

// Opens a stack of images. 
bool FileOpener::openStack() {
	vector<unique_ptr<byte[]>> stack;
	FileInfo first_fi = fi[0];
	long skip = first_fi.getOffset();

	unsigned int imageSize = first_fi.getImageSize();
	if (first_fi.fileType==FileInfo::GRAY12_UNSIGNED)imageSize = (int)(first_fi.width*first_fi.height*2);//12bitの場合は16bit配列に格納
	cout << "ImageSize: " << imageSize << " byte" << endl;

	try {
		stringstream fpass;
		fpass << first_fi.directory << '\\' << first_fi.fileName;
		fstream in(fpass.str().c_str(), ios::binary|ios::in);
		if(in.fail()){
			//cout << "File Open Error" << endl;
			return false;
		}
		ImageReader reader(first_fi);
		for (int i = 0; i < first_fi.nImages; i++) {
			unique_ptr<byte[]> slice;
			slice = move(unique_ptr<byte[]>(new byte[imageSize]));
			reader.readPixels(in, slice.get(), skip);

			if(i < 10 && (first_fi.fileType == FileInfo::GRAY32_INT || first_fi.fileType == FileInfo::GRAY32_UNSIGNED || first_fi.fileType == FileInfo::GRAY32_FLOAT)){
				cout << i << ": ";
				int j = 0;
				for(int k = 0; k < imageSize/first_fi.getBytesPerPixel() && k < 5; k++){
					int tmp = (int)(((slice[j+3]&0xff)<<24) | ((slice[j+2]&0xff)<<16) | ((slice[j+1]&0xff)<<8) | (slice[j]&0xff));
					Float_Int_Converter conv;
					conv.itr = tmp;
					cout << conv.f << ' ';
					j += 4;
				}
				cout << endl;
			}

			if(i < 10 && (first_fi.fileType == FileInfo::GRAY16_SIGNED || first_fi.fileType == FileInfo::GRAY16_UNSIGNED)){
				cout << i << ": ";
				for (int k = 0, j = 0; k < imageSize/first_fi.getBytesPerPixel() && k < 5; k++, j+=2){
					cout << (short)(((slice[j+1]&0xff)<<8) | (slice[j]&0xff)) << ' ';
				}
				cout << endl;
			}

			if(i < 10 && (first_fi.fileType == FileInfo::GRAY8 || first_fi.fileType == FileInfo::COLOR8)){
				cout << i << ": ";
				for (int j = 0; j < imageSize && j < 5; j++){
					cout << (int)(slice[j]&0xff) << ' ';
				}
				cout << endl;
			}

			if (!slice) break;
			stack.push_back(move(slice));
			skip = first_fi.gapBetweenImages;
		}
		in.close();
	}
	catch (std::exception &ex) {
		cerr << ex.what() << endl;
	}

	if (stack.size() != first_fi.nImages)return false;

	SAFE_DELETE_ARRAY(pixels);
	pixels = new byte[stack.size()*imageSize];
	for(int i = 0; i < stack.size(); i++){
		memcpy((byte *)(pixels + imageSize*i), stack[i].get(), imageSize);
	}

	for(int i = 0; i < stack.size(); i++){

		if(i < 10 && (first_fi.fileType == FileInfo::GRAY32_INT || first_fi.fileType == FileInfo::GRAY32_UNSIGNED || first_fi.fileType == FileInfo::GRAY32_FLOAT)){
			cout << i << ": ";
			int j = 0;
			for(int k = 0; k < imageSize/first_fi.getBytesPerPixel() && k < 5; k++){
				int tmp = (int)(((pixels[i*imageSize + j+3]&0xff)<<24) | ((pixels[i*imageSize + j+2]&0xff)<<16) | ((pixels[i*imageSize + j+1]&0xff)<<8) | (pixels[i*imageSize + j]&0xff));
				Float_Int_Converter conv;
				conv.itr = tmp;
				cout << conv.f << ' ';
				j += 4;
			}
			cout << endl;
		}

		if(i < 10 && (first_fi.fileType == FileInfo::GRAY16_SIGNED || first_fi.fileType == FileInfo::GRAY16_UNSIGNED)){
			cout << i << ": ";
			for (int k = 0, j = 0; k < imageSize/first_fi.getBytesPerPixel() && k < 5; k++, j+=2){
				cout << (short)(((pixels[i*imageSize + j+1]&0xff)<<8) | (pixels[i*imageSize + j]&0xff)) << ' ';
			}
			cout << endl;
		}

		if(i < 10 && (first_fi.fileType == FileInfo::GRAY8 || first_fi.fileType == FileInfo::COLOR8)){
			cout << i << ": ";
			for (int j = 0; j < imageSize && j < 5; j++){
				cout << (int)(pixels[i*imageSize + j]&0xff) << ' ';
			}
			cout << endl;
		}
	}

	/*
	if (fi.sliceLabels!=null && fi.sliceLabels.length<=stack.getSize()) {
		for (int i=0; i<fi.sliceLabels.length; i++)
			stack.setSliceLabel(fi.sliceLabels[i], i+1);
	}
	ImagePlus imp = new ImagePlus(fi.fileName, stack);
	if (fi.info!=null)
		imp.setProperty("Info", fi.info);
	if (fi.roi!=null)
		imp.setRoi(RoiDecoder.openFromByteArray(fi.roi));
	if (fi.overlay!=null)
		setOverlay(imp, fi.overlay);
	if (show) imp.show();
	imp.setFileInfo(fi);
	setCalibration(imp);
	ImageProcessor ip = imp.getProcessor();
	if (ip.getMin()==ip.getMax())  // find stack min and max if first slice is blank
		setStackDisplayRange(imp);
	if (!silentMode) IJ.showProgress(1.0);
	//silentMode = false;
	*/
	return true;
}

bool FileOpener::allSameSizeAndType(FileInfo *info, size_t len) {
	if (isEmpty)return false;
	bool sameSizeAndType = true;
	bool contiguous = true;
	long startingOffset = info[0].getOffset();
	int size = info[0].width*info[0].height*info[0].getBytesPerPixel();
	for (int i=1; i<len; i++) {
		sameSizeAndType &= info[i].fileType==info[0].fileType
			&& info[i].width==info[0].width
			&& info[i].height==info[0].height;
		contiguous &= info[i].getOffset()==startingOffset+i*size;
	}
	if (contiguous &&  info[0].fileType!=FileInfo::RGB48)
		info[0].nImages = len;
	
	return sameSizeAndType;
}

bool FileOpener::loadTiffStack(){
	if (isEmpty)return false;

	vector<unique_ptr<byte[]>> stack;

	if (nInfo>1 && !allSameSizeAndType(fi, nInfo)){
		cout << "allSameSizeAndType: false" << endl;
		return false;
	}
	FileInfo first_fi = fi[0];
	if (first_fi.nImages>1)
		return openStack(); // open contiguous images as stack
	else {
		long long skip = first_fi.getOffset();
		cout << "skip: " << skip << endl;
		unsigned int imageSize = first_fi.getImageSize();
		if (first_fi.fileType==FileInfo::GRAY12_UNSIGNED)imageSize = (int)(first_fi.width*first_fi.height*2);//12bitの場合は16bit配列に格納
		cout << "ImageSize: " << imageSize << " byte" << endl;
		long loc = 0L;
		int nChannels = 1;
		try {
			stringstream fpass;
			fpass << first_fi.directory << '\\' << first_fi.fileName;
			fstream in(fpass.str().c_str(), ios::binary|ios::in);
			if(in.fail()){
				//cout << "File Open Error" << endl;
				return false;
			}
			for (int i=0; i<nInfo; i++) {
				nChannels = 1;
				vector<unique_ptr<byte[]>> channels;
				unique_ptr<byte[]> slice;
				ImageReader reader(fi[i]);
				if (fi[i].compression>=FileInfo::LZW) {
					first_fi.stripOffsets = fi[i].stripOffsets;
					first_fi.stripLengths = fi[i].stripLengths;
				}
				int bpp = fi[i].getBytesPerPixel();
				if (fi[i].samplesPerPixel>1 && !(bpp==3||bpp==4||bpp==6)) {
					nChannels = first_fi.samplesPerPixel;
					channels.resize(nChannels);
					for (int c=0; c<nChannels; c++) {
						unique_ptr<byte[]> chdata(new byte[imageSize]);
						reader.readPixels(in, chdata.get(), c==0?skip:0L);
						channels[c] = move(chdata);
					}
				} else{
					slice = move(unique_ptr<byte[]>(new byte[imageSize]));
					reader.readPixels(in, slice.get(), skip);
					/*
					if(i < 10 && (first_fi.fileType == FileInfo::GRAY32_INT || first_fi.fileType == FileInfo::GRAY32_UNSIGNED || first_fi.fileType == FileInfo::GRAY32_FLOAT)){
						cout << i << ": ";
						int j = 0;
						for(int k = 0; k < imageSize/first_fi.getBytesPerPixel() && k < 5; k++){
							int tmp = (int)(((slice[j+3]&0xff)<<24) | ((slice[j+2]&0xff)<<16) | ((slice[j+1]&0xff)<<8) | (slice[j]&0xff));
							Float_Int_Converter conv;
							conv.itr = tmp;
							cout << conv.f << ' ';
							j += 4;
						}
						cout << endl;
					}
					*/
				}
				if (!slice && channels.empty()) break;
				loc += imageSize*nChannels+skip;
				if (i<(nInfo-1)) {
					skip = fi[i+1].getOffset()-loc;
					if (fi[i+1].compression>=FileInfo::LZW) skip = 0;
					if (skip<0L) {
						cerr << "Opener: Unexpected image offset" << endl;
						break;
					}
				}
				/*if (first_fi.fileType==FileInfo::RGB48) {
					Object[] pixels2 = (Object[])pixels;
					stack.addSlice(null, pixels2[0]);                   
					stack.addSlice(null, pixels2[1]);                   
					stack.addSlice(null, pixels2[2]);
					isRGB48 = true;                 
				} else*/ if (nChannels>1) {
					for (int c=0; c<nChannels; c++) {
						stack.push_back(move(channels[c]));
					}
				} else
					stack.push_back(move(slice));
			}
			in.close();
		}
		catch (std::exception &ex) {
			cerr << ex.what() << endl;
		}
				
		if (stack.size() != nInfo)return false;//RGBの場合は不適 要修正

		SAFE_DELETE_ARRAY(pixels);
		pixels = new byte[stack.size()*imageSize];
		for(int i = 0; i < stack.size(); i++){
			memcpy((byte *)(pixels + imageSize*i), stack[i].get(), imageSize);
		}

		/*
		if (first_fi.fileType==FileInfo::GRAY16_UNSIGNED||first_fi.fileType==FileInfo::GRAY12_UNSIGNED
			||first_fi.fileType==FileInfo::GRAY32_FLOAT||first_fi.fileType==FileInfo::RGB48) {
				ImageProcessor ip = stack.getProcessor(1);
				ip.resetMinAndMax();
				stack.update(ip);
		}
		//if (first_fi.whiteIsZero)
		//  new StackProcessor(stack, stack.getProcessor(1)).invert();
		ImagePlus imp = new ImagePlus(first_fi.fileName, stack);
		new FileOpener(first_fi).setCalibration(imp);
		imp.setFileInfo(first_fi);
		if (first_fi.info!=null)
			imp.setProperty("Info", first_fi.info);
		if (first_fi.description!=null && first_fi.description.contains("order=zct"))
			new HyperStackConverter().shuffle(imp, HyperStackConverter.ZCT);
		int stackSize = stack.getSize();
		if (nChannels>1 && (stackSize%nChannels)==0) {
			imp.setDimensions(nChannels, stackSize/nChannels, 1);
			imp = new CompositeImage(imp, CompositeImage.COMPOSITE);
			imp.setOpenAsHyperStack(true);
		} else if (imp.getNChannels()>1)
			imp = makeComposite(imp, first_fi);
		IJ.showProgress(1.0);
		return imp;
		*/
	}

	return true;
}

bool FileOpener::getXYZStackRaw(byte *dst, int channel, int frame){
	if(pixels == NULL || fi == NULL || dst == NULL || channel < 0 || frame < 0)return false;
	if(fcal.channels <= channel || fcal.frames <= frame)return false;

	unsigned int imageSize = fi[0].getImageSize();
	for(int i = 0; i < fcal.slices; i++){
		//imageJのスタックはXYCZT
		memcpy((byte *)(dst + imageSize*i),
			   (byte *)(pixels + imageSize*channel + imageSize*fcal.channels*i + imageSize*fcal.channels*fcal.slices*frame),
			   imageSize);
	}

	return true;
}

bool FileOpener::getXYZStackFloat(float *dst, bool normalize, int channel, int frame){
	if(pixels == NULL || fi == NULL || dst == NULL || channel < 0 || frame < 0)return false;
	if(fcal.channels <= channel || fcal.frames <= frame)return false;

	unsigned int imageSize = fi[0].getImageSize();
	if (fi[0].fileType==FileInfo::GRAY12_UNSIGNED)imageSize = (int)(fi[0].width*fi[0].height*2);
	unsigned int pixel_num = fi[0].width*fi[0].height;

	float *temp = new float[pixel_num*fcal.slices];

	//imageJのスタックはXYCZT
	if(fi[0].fileType == FileInfo::GRAY32_INT || fi[0].fileType == FileInfo::GRAY32_UNSIGNED || fi[0].fileType == FileInfo::GRAY32_FLOAT){
		for(int i = 0; i < fcal.slices; i++){
			memcpy(temp + pixel_num*i,
				   (float *)pixels + pixel_num*channel + pixel_num*fcal.channels*i + pixel_num*fcal.channels*fcal.slices*frame,
				   imageSize);
		}
	}

	if(fi[0].fileType == FileInfo::GRAY16_SIGNED){
		for(int i = 0; i < fcal.slices; i++){
			for(int j = 0; j < pixel_num; j++)
			temp[j + pixel_num*i] =
				(float)((short *)pixels)[j + pixel_num*channel + pixel_num*fcal.channels*i + pixel_num*fcal.channels*fcal.slices*frame];
		}
	}

	if(fi[0].fileType == FileInfo::GRAY16_UNSIGNED){
		for(int i = 0; i < fcal.slices; i++){
			for(int j = 0; j < pixel_num; j++)
			temp[j + pixel_num*i] =
				(float)((unsigned short *)pixels)[j + pixel_num*channel + pixel_num*fcal.channels*i + pixel_num*fcal.channels*fcal.slices*frame];
		}
	}

	if(fi[0].fileType == FileInfo::GRAY8 || fi[0].fileType == FileInfo::COLOR8){
		for(int i = 0; i < fcal.slices; i++){
			for(int j = 0; j < pixel_num; j++)
			temp[j + pixel_num*i] =
				(float)pixels[j + pixel_num*channel + pixel_num*fcal.channels*i + pixel_num*fcal.channels*fcal.slices*frame];
		}
	}

	if(normalize){
		float fmax = FLT_MIN;
		float fmin = FLT_MAX;
		for(int i = 0; i < fcal.slices; i++){
			for(int j = 0; j < pixel_num; j++){
				fmax = (fmax < temp[j + pixel_num*i]) ? temp[j + pixel_num*i] : fmax;
				fmin = (fmin > temp[j + pixel_num*i]) ? temp[j + pixel_num*i] : fmax;
			}
		}
		float denom = (abs(fmax) > abs(fmin)) ? abs(fmax) : abs(fmin);
		for(int i = 0; i < fcal.slices; i++){
			for(int j = 0; j < pixel_num; j++)
				temp[j + pixel_num*i] /= denom;
		}
	}
	/*
	cout << '\n';
	for(int i = 0; i < 10; i++){
		cout << i << ": ";
		for(int j = 0; j < 5; j++){
			cout << temp[i*pixel_num + j] << ' ';
		}
		cout << endl;
	}
	*/
	memcpy(dst, temp, pixel_num*fcal.slices*sizeof(float));

	delete [] temp;
	
	return true;
}
/*
int _tmain(int argc, _TCHAR* argv[]){

#ifdef _DEBUG 
	_CrtDumpMemoryLeaks();
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	Double_ULL_Converter conv;
	conv.ull = 4596373779694328218ULL;
	printf("%f : %lld\n", conv.d, conv.ull);
	char *test = NULL;
	if(test == NULL)cout << "test == NULL" << endl;
	test = new char[100];
	cout << "allocated" << endl;
	if(test == NULL)cout << "test == NULL" << endl;
	delete [] test;
	cout << "deleted" << endl;
	if(test == NULL)cout << "test == NULL" << endl;

	stringstream ss;
	double d = 1.0;
	ss << "A";
	ss >> d;
	cout << d << "\n" << endl;

	test = new char(100);
	delete [] test;
	unsigned char upper_b = 0xff;
	unsigned char lower_b = 0xff;
	short val = (short)((((upper_b&0xff)<<8) | (lower_b&0xff))+0);
	cout << (short)(65535) << endl;

	//TiffDecoder td("C:\\Users\\T.Kawase\\Desktop\\step-tablet\\", "step-tablet_calibrated.tif");
	//TiffDecoder td("C:\\Users\\T.Kawase\\Desktop", "Multi_Color_Stack.tif");
	//TiffDecoder td("C:\\Users\\T.Kawase\\Desktop", "image2 (1).tif");
	TiffDecoder td("C:\\Users\\Nishimura\\Desktop", "multiColorStack.tif");
	//TiffDecoder td("C:\\Users\\Nishimura\\Desktop\\images", "C2-smooth.tif");
	td.enableDebugging();
	if(!td.loadTiffInfo()){
		cout << "loadTiffInfo Error" << endl;
		return -1;
	}
	FileInfo *fis = new FileInfo[td.getFileInfoNum()];
	td.getFileInfo(fis);
	cout << fis[0].debugInfo << endl;

	FileOpener fo(fis, td.getFileInfoNum());
	if(!fo.empty())fo.loadTiffStack();

	Calibration cal = fo.getImageCalibration();
	if(cal.loaded){
		float *dummy = new float[cal.width*cal.height*cal.slices];
		fo.getXYZStackFloat(dummy, false, 0);
		delete[] dummy;
	}
	else cout << "getImageCalibration: FAILED" << endl;

	char fi_info[1000];
	fis[0].to_c_string(fi_info);
	if(fis[0].description)cout << fi_info << "\nimage_description:\n" << fis[0].description << "//spacing = PixelDepth\n" << endl;
	else cout << fi_info << "\n" << endl;
	cout << "FileInfo_NUM: " << td.getFileInfoNum() << endl;

	delete[] fis;

	return 0;
}
*/
#pragma warning(pop)