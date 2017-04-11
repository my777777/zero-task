/*
 * DataLayer.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */
#include "DataLayer.h"
#include "ZeroNet.h"
#include "ZeroMessage.h"
#include "ZeroLogger.h"
#include "LayerFactory.hpp"

DataLayer::DataLayer(const LayerParam& layerParam) :
		Layer(layerParam) {
	mp_trainPrefetcher = new TrainPrefetcher();
	mp_testPrefetcher=new TestPrefetcher();

}
DataLayer::~DataLayer() {
	// TODO Auto-generated destructor stub
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	checkCudaErrors(cudaSetDevice(getGpuIndex()));
	ExeMemoryRelease();
}

void DataLayer::ExeMemoryAlloc() {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	cudaSetDevice(getGpuIndex());
	long inDataSize = getBatchSize() * getInMapsOrNeurons() * getMapWidth() * getMapHeight();
	mp_hostInData = new double[inDataSize];
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutData, inDataSize * sizeof(double)));

	int outLabelSize = getBatchSize() * gi_mnistNumberOfClass;
	mp_hostInLabel = new double[outLabelSize];
	//checkCudaErrors(cudaMalloc((void** )&mp_deviceOutLabel, outLabelSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutDiff, outLabelSize * sizeof(double)));
	mp_trainPrefetcher->start();
	mp_testPrefetcher->start();
}

void DataLayer::ExeMemoryRelease() {

	if (mp_trainPrefetcher) {
		delete mp_trainPrefetcher;
		mp_trainPrefetcher = NULL;
	}
	if (mp_testPrefetcher) {
		delete mp_testPrefetcher;
		mp_testPrefetcher = NULL;
	}
	if (mp_hostInData != NULL) {
		delete[] mp_hostInData;
		mp_hostInData = NULL;
	}
	if (mp_hostInLabel != NULL) {
		delete[] mp_hostInLabel;
		mp_hostInLabel = NULL;
	}
}
void DataLayer::ExeForward(int fragment) {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment+1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	long outDataSize = getFragmentSize() * getOutMapsOrNeurons() * getMapWidth() * getMapHeight();
//	assert(outDataSize != 0);
//	checkCudaErrors(cudaMemcpy((void* )(mp_deviceOutData + fragment * outDataSize), (void* )(mp_hostInData + fragment * outDataSize), outDataSize * sizeof(double), cudaMemcpyHostToDevice));
//	int outLabelSize = getFragmentSize() * gi_mnistNumberOfClass;
//	checkCudaErrors(cudaMemcpy((void* )(mp_deviceOutLabel + fragment * outLabelSize), (void* )(mp_hostInLabel + fragment * outLabelSize), outLabelSize * sizeof(double), cudaMemcpyHostToDevice));
	ZeroLogger::LogIfInfo(gb_logDataFlag,mp_deviceOutData+fragment*outDataSize,getFragmentSize(),getOutMapsOrNeurons()* getMapWidth()*getMapHeight(),getLayerName()+ " outData:");
}

void DataLayer::ExeBackward(int fragment) {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment+1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();
}

void DataLayer::PrefetchNextBatch(int phase) {

//	int batchSize=getBatchSize() ;
//	int outDataSizeOfImage=getOutMapsOrNeurons() * getMapArea();
//	int outLabelSize = batchSize* gi_mnistNumberOfClass;
//	assert(outDataSizeOfImage != 0 && batchSize != 0);
//	for (int b = 0; b < getBatchSize(); ++b) {
//		for (int n = 0; n < outDataSizeOfImage; ++n) {
//			mp_hostInData[b * outDataSizeOfImage + n] = rand() % 3 + 1;
//		}
//		for (int l = 0; l < gi_mnistNumberOfClass; ++l) {
//			mp_hostInLabel[b * gi_mnistNumberOfClass + l] = 0;
//		}
//		mp_hostInLabel[b * gi_mnistNumberOfClass + rand() % gi_mnistNumberOfClass] = 1;
//	}
//	checkCudaErrors(cudaMemcpy((void* )(mp_deviceOutData), (void* )(mp_hostInData), batchSize * outDataSizeOfImage * sizeof(double), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy((void* )(mp_deviceOutLabel), (void* )(mp_hostInLabel), outLabelSize * sizeof(double), cudaMemcpyHostToDevice));
	struct timeval t1,t2,t3,t4;
	gettimeofday(&t1,NULL);
	DataBatch* dataBatch=phase?mp_testPrefetcher->getNextBatch():mp_trainPrefetcher->getNextBatch();
	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" ###################costTime1:"<<costTime;
	int outDataSize=getBatchSize() * getOutMapsOrNeurons() * getMapArea() ;
	checkCudaErrors(cudaMemcpy((void* )(mp_deviceOutData), (void* )(dataBatch->mp_deviceData), outDataSize* sizeof(double), cudaMemcpyDeviceToDevice));
	int outLabelSize = getBatchSize() * gi_mnistNumberOfClass;
	//checkCudaErrors(cudaMemcpy((void* )(mp_deviceOutLabel), (void* )(dataBatch->mp_deviceLabel), outLabelSize * sizeof(double), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy((void* )(mp_deviceOutDiff), (void* )(dataBatch->mp_deviceLabel), outLabelSize * sizeof(double), cudaMemcpyDeviceToDevice));

	gettimeofday(&t3, NULL);
	costTime = (t3.tv_sec - t2.tv_sec) * 1000 + (t3.tv_usec - t2.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" ###################costTime2:"<<costTime;
	delete dataBatch;
	gettimeofday(&t4, NULL);
	costTime = (t4.tv_sec - t3.tv_sec) * 1000 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" ###################costTime3:"<<costTime;
}

REGISTER_LAYER_CLASS(Data);
