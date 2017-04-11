/*
 * FileHandle.h
 *
 *  Created on: 2016-7-19
 *      Author: zys
 */

#ifndef FILEHANDLE_H_
#define FILEHANDLE_H_

#include "Resource.h"

class FileHandle {

public:

	FileHandle();
	virtual ~FileHandle();

	static FileHandle* Get();
	void ReadBinaryFile(const string fname, int size, double* data_h);

	void ReadTextFile(const string fname,int size,double* img);

	void ReadTextFile2(const string fname, int rows,int cols,double* out);
	void splitString(const string& s,vector<double>& v,const string & c);
};

#endif /* FILEHANDLE_H_ */
