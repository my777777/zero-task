/*
 * ZeroLogger.cpp
 *
 *  Created on: Jan 13, 2017
 *      Author: zys
 */
#include "ZeroLogger.h"

ZeroLogger::ZeroLogger() {
	// TODO Auto-generated constructor stub

}

ZeroLogger::~ZeroLogger() {
	// TODO Auto-generated destructor s
}

void ZeroLogger::LogIfInfo(bool printFlag,double* deviceData,int rows,int cols,string infoHead) {

	if (printFlag) {
		ostringstream oss;
		long dataSize=rows*cols;
		double* hostData = new double[dataSize];
		checkCudaErrors(cudaMemcpy(hostData, deviceData, dataSize * sizeof(double), cudaMemcpyDeviceToHost));
		oss << infoHead << ":\n";
		for (long i = 0; i < dataSize; ++i) {
			oss << hostData[i] << ",";
			if ((i + 1) % cols == 0) {
				oss << "\n";
			}
		}
		LOG(INFO)<<oss.str();
		delete[] hostData;
	}
}

void ZeroLogger::LogIfInfoHost(bool printFlag,double *hostData, int rows, int cols, string infoHead) {

	if (printFlag) {
		ostringstream oss;
		long dataSize=rows*cols;
		oss << infoHead << ":\n";
		for (long i = 0; i < dataSize; ++i) {
			oss << hostData[i] << ",";
			if ((i + 1) % cols == 0) {
				oss << "\n";
			}
		}
		LOG(INFO)<<oss.str();
	}
}


