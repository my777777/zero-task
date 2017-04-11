/*
 * FileHandle.cpp
 *
 *  Created on: 2016-7-19
 *      Author: zys
 */
#include "FileHandle.h"

FileHandle::FileHandle() {
	// TODO Auto-generated constructor stub

}

FileHandle::~FileHandle() {
	// TODO Auto-generated destructor stub
}

void FileHandle::ReadBinaryFile(const string fname, int size, double* data_h) {

	LOG(INFO)<<"FileHandle::ReadBinaryFile:"<<fname;
	std::ifstream dataFile(fname.c_str(), std::ios::in | std::ios::binary);
	std::stringstream error_s;
	if (!dataFile) {
		error_s << "Error opening file " << fname;
		FatalError(error_s.str());
	}
	float* data_tmp = new float[size];
	int size_b = size * sizeof(float);
	if (!dataFile.read((char*) data_tmp, size_b)) {
		error_s << "Error reading file " << fname;
		FatalError(error_s.str());
	}
	for(int i=0;i<size;i++) {
		data_h[i]=data_tmp[i];
	}
}

void FileHandle::ReadTextFile(const string fname, int size, double* img) {

	std::fstream file(fname.c_str(), std::ios_base::in);
	std::stringstream error_s;
	if (!file) {
		error_s << "Error opening file " << fname;
		FatalError(error_s.str());
	}
	char* data_tmp = new char[size];
	int size_b = size * sizeof(char);
	if (!file.read((char*) data_tmp, size_b)) {
		error_s << "Error reading file " << fname;
		FatalError(error_s.str());
	}
	for (int i = 0; i < size; i++) {
		img[i] = (data_tmp[i] - 48);

	}
}

void FileHandle::splitString(const string& s,vector<double>& v,const string & c){

	std::string::size_type p1,p2;
	p2=s.find(c);
	p1=0;
	while(std::string::npos!=p2){
		v.push_back(atof(s.substr(p1,p2-p1).c_str()));
		p1=p2+c.size();
		p2=s.find(c,p1);
	}
	if(p1!=s.length()){
		v.push_back(atof(s.substr(p1).c_str()));
	}

}
void FileHandle::ReadTextFile2(const string fname, int rows,int cols,double* out) {

	ifstream file(fname, std::ios_base::in);
	std::stringstream error_s;
	if (!file) {
		error_s << "Error opening file " << fname;
		FatalError(error_s.str());
	}
	int i=0;
	for(string line;getline(file,line);){
		vector<double> v;
		splitString(line,v, " ");
		for(int c=0;c<cols;c++){
			out[i*cols+c]=v[c];
		}
		i++;
	}
	assert((i)==rows);
}

FileHandle* FileHandle::Get() {
	static FileHandle *inst = new FileHandle();
	return inst;
}
