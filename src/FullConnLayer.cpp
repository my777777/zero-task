/*
 * FullConnLayer.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */

#include "FullConnLayer.h"
#include "ZeroLogger.h"
#include "ZeroMessage.h"
#include "FileHandle.h"
#include "Filler.h"
#include "Shape.h"
#include "LayerFactory.hpp"

FullConnLayer::FullConnLayer(const LayerParam& layerParam) :
		Layer(layerParam) {

	LOG_IF(INFO,gb_logFlag) << this->getLayerName() << ":" << __func__;
	InitLayer(layerParam);

}

void FullConnLayer::InitLayer(const LayerParam& layerParam) {

//	checkCublasErrors(cublasCreate(&m_cublasHandle));
//	if (mi_streams != 0) {
//		checkCublasErrors(cublasSetStream(m_cublasHandle,mp_cudaStreams[0]));
//	}
}

FullConnLayer::~FullConnLayer() {
	LOG_IF(INFO,gb_logFlag) << this->getLayerName() << ":" << __func__;
	checkCudaErrors(cudaSetDevice(getGpuIndex()));
//	checkCublasErrors(cublasDestroy(m_cublasHandle));
	ExeMemoryRelease();
}

void FullConnLayer::ExeMemoryAlloc() {

	long outDataSize = gi_batchSize * getOutMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutData, outDataSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_deviceOutData, 0, outDataSize * sizeof(double)));

	long outDiffSize = gi_batchSize * getInMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutDiff, outDiffSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_deviceOutDiff, 0, outDiffSize * sizeof(double)));

	// if data malloc on different devices,need copy.
	if (getPrevLayer()->getGpuIndex() != getGpuIndex()) {
		LOG_IF(INFO,gb_logFlag) << getLayerName() << " forward different device";
		long prevLayerFragmentOutDataSize = getFragmentSize() * getInMapsOrNeurons();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInData, prevLayerFragmentOutDataSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInData, 0, prevLayerFragmentOutDataSize * sizeof(double)));
	}
	if (getNextLayer()->getGpuIndex() != getGpuIndex()) {
		LOG_IF(INFO,gb_logFlag) << getLayerName() << " backward different device";
		long nextLayerFragmentOutDiffSize = getFragmentSize() * getOutMapsOrNeurons();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInDiff, nextLayerFragmentOutDiffSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInDiff, 0, nextLayerFragmentOutDiffSize * sizeof(double)));
	}

	long weightSize = getOutMapsOrNeurons() * getInMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceWeight, weightSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_deviceWeightGradient, weightSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_historyWeightGradient, weightSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_historyWeightGradient, 0, weightSize * sizeof(double)));
	InitWeight(weightSize);

	long biasSize = getOutMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceBias, biasSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_deviceBiasGradient, biasSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_historyBiasGradient, biasSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_historyBiasGradient, 0, biasSize * sizeof(double)));
	InitBias(biasSize);

	long biasMultiplierSize = getFragmentSize();
	checkCudaErrors(cudaMalloc((void** )&mp_biasMultiplier, biasMultiplierSize * sizeof(double)));
	InitBiasMultiplier(biasMultiplierSize);
}

void FullConnLayer::InitWeight(long weightSize) {

	MSRAFiller filler;
	int v[] = { getOutMapsOrNeurons(), getInMapsOrNeurons() };
	Shape shape(2, v);
	filler.Fill(mp_deviceWeight, shape);
}

void FullConnLayer::InitBias(long biasSize) {

	ConstantFiller biasFiller;
	int biasV[] = { getOutMapsOrNeurons() };
	Shape biasShape(1, biasV);
	biasFiller.Fill(mp_deviceBias, biasShape);
}

void FullConnLayer::InitBiasMultiplier(long biasMultiplierSize) {

	double* biasMultiplier = new double[biasMultiplierSize];
	for (int i = 0; i < biasMultiplierSize; i++) {
		biasMultiplier[i] = 1;
	}
	checkCudaErrors(cudaMemcpy((void* )mp_biasMultiplier, (const void* )biasMultiplier, biasMultiplierSize * sizeof(double), cudaMemcpyHostToDevice));
	delete biasMultiplier;
}

void FullConnLayer::ExeMemoryRelease() {
}

void FullConnLayer::ExeForward(int fragment) {

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment + 1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	int outNeurons = getOutMapsOrNeurons();
	int inNeurons = getInMapsOrNeurons();
	int fragmentSize = getFragmentSize();

//	ZeroLogger::LogIfInfo(gb_logDatlaFlag, mp_fragmentOutData, fragmentSize, outNeurons, getLayerName() + " outData 1:");

	/*
	 * calculate bias: mp_deviceOutData=mp_biasMultiplier*mp_deviceBias
	 * mp_biasMultiplier:dim(fragmentSize*1)
	 * mp_deviceBias:dim(1*outNeurons)
	 * mp_deviceOutData:(fragmentSize*outNeurons)
	 */
	double alpha = 1.0;
	double beta = 0.0;
	checkCublasErrors(cublasDgemm (m_cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, //
			outNeurons,fragmentSize,1,//
			&alpha,mp_deviceBias,outNeurons,//
			mp_biasMultiplier,1,//
			&beta,mp_fragmentOutData,outNeurons//
			));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceBias, 1, outNeurons, getLayerName() + " bias:");
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outNeurons, getLayerName() + " outData 2:");
	/*
	 * calculate outData:mp_deviceOutData=inData*weight+mp_deviceOutData
	 * inData:dim(fragmentSize*inNeurons)
	 * weight:dim(outNeurons*inNeurons)
	 * mp_deviceOutData:dim(fragmentSize*outNeurons)
	 */
	beta = 1.0;
	checkCublasErrors(cublasDgemm(m_cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N, //
			outNeurons,fragmentSize,inNeurons,//
			&alpha,mp_deviceWeight,inNeurons,//
			mp_fragmentInData,inNeurons,//
			&beta,mp_fragmentOutData,outNeurons));
	//ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInData, fragmentSize, inNeurons, getLayerName() + " inData:");
	//ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeight, outNeurons, inNeurons, getLayerName() + " weightData:");
	//ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceBias, 1, outNeurons, getLayerName() + " bias:");
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outNeurons, getLayerName() + " outData 3:");


	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" -------------->costTime:"<<costTime;
}

void FullConnLayer::ExeBackward(int fragment) {

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment + 1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	int outNeurons = getOutMapsOrNeurons();
	int fragmentSize = getFragmentSize();
	int inNeurons = getInMapsOrNeurons();

	/*
	 * calculate outDiff: inDiff*weight
	 *  inDiff matrix,	dimA(fragmentSize*outNeurons)
	 *  weight matrix , dimB(outNeurons*inNeurons),
	 *  outDiff matrix , dimC(fragmentSize*inNeurons)
	 */
	double alpha = 1.0;
	double beta = 0.0;
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInDiff, fragmentSize, outNeurons, getLayerName() + " inDiff");
	checkCublasErrors(cublasDgemm (m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, //
			inNeurons,fragmentSize,outNeurons,//
			&alpha,mp_deviceWeight, inNeurons,//
			mp_fragmentInDiff,outNeurons,//
			&beta,mp_fragmentOutDiff, inNeurons));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutDiff, fragmentSize, inNeurons, getLayerName() + " outDiff");
	/*
	 * calculate mp_deviceWeightGradient: inDiff* deviceInData(mp_preLayer->mp_deviceOutData)
	 * inDiff:dim(fragmentSize*outNeurons)
	 * deviceInData :dim(fragmentSize*inNeurons)
	 * mp_deviceWeightGradient:dim(outNeurons*inNeurons)
	 *
	 * if fragment!=0 ,then beta=1.0(add current result to previous result).
	 */
	beta = (fragment) ? 1.0 : 0.0;
	checkCublasErrors(cublasDgemm (m_cublasHandle, CUBLAS_OP_N,CUBLAS_OP_T, //
			inNeurons,outNeurons,fragmentSize,//
			&alpha,mp_fragmentInData,inNeurons,//
			mp_fragmentInDiff,outNeurons,//
			&beta,mp_deviceWeightGradient,inNeurons));

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeightGradient, outNeurons, inNeurons, getLayerName() + " WeightGradient");
	/*
	 * Calculate mp_deviceBiasGradient: mp_biasMultiplier*inDiff
	 *	mp_biasMultiplier:dim(fragmentSize*1)
	 *	inDiff:dim(fragmentSize*outNeurons)
	 *	mp_deviceBiasGradient:dim(outNeurons*1)
	 *
	 *	if fragment!=0 ,then beta=1.0(add current result to previous result).
	 */
	beta = (fragment) ? 1.0 : 0.0;
	checkCublasErrors(cublasDgemm (m_cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, //
			outNeurons,1,fragmentSize,//
			&alpha,mp_fragmentInDiff,outNeurons,//
			mp_biasMultiplier,fragmentSize,//
			&beta,mp_deviceBiasGradient,outNeurons//
			));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceBiasGradient, 1, outNeurons, getLayerName() + " BiasGradient");

	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" <--------------costTime:"<<costTime;
}

void FullConnLayer::ExeUpdate(long executedBatchNumber) {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__ << "; executedBatchNumber " << executedBatchNumber;

	double local_lr1 = gd_lr_mult1 * gd_learnRate * pow(1 + gd_gamma * executedBatchNumber, gd_power);
	double local_lr2 = gd_lr_mult2 * gd_learnRate * pow(1 + gd_gamma * executedBatchNumber, gd_power);
	double alpha = -1;
	long biasSize = getOutMapsOrNeurons();
	long weightSize = getOutMapsOrNeurons() * getInMapsOrNeurons();

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeight, getOutMapsOrNeurons(), getInMapsOrNeurons(), getLayerName() + " mp_deviceWeight1");
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeightGradient, getOutMapsOrNeurons(), getInMapsOrNeurons(), getLayerName() + " mp_deviceWeightGradient1");
	//(1) cublasDaxpy: delta = decay*weight + delta
	//checkCublasErrors(cublasDaxpy( m_cublasHandle, weightSize, &gd_decay, mp_deviceWeight, 1, mp_deviceWeightGradient, 1 ));
	//checkCublasErrors(cublasDaxpy( m_cublasHandle, biasSize, &gd_decay, mp_deviceBias, 1, mp_deviceBiasGradient, 1 ));
	//(2) cublasDgeam: history = learn_rate*delta + momentum*history
	checkCublasErrors(cublasDgeam(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, weightSize, 1, &gd_moment, //
			mp_historyWeightGradient, weightSize, &local_lr1, mp_deviceWeightGradient, weightSize, mp_historyWeightGradient, weightSize));
	checkCublasErrors(cublasDgeam(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, biasSize, 1, &gd_moment, //
			mp_historyBiasGradient, biasSize, &local_lr2, mp_deviceBiasGradient, biasSize, mp_historyBiasGradient, biasSize));
	//(3) cublasDaxpy: weight =weight +(-1) *history
	checkCublasErrors(cublasDaxpy( m_cublasHandle, weightSize, &alpha, mp_historyWeightGradient, 1,mp_deviceWeight, 1 ));
	//(3) cublasDaxpy: bias =bias +(-1) *history
	checkCublasErrors(cublasDaxpy( m_cublasHandle, biasSize, &alpha, mp_historyBiasGradient, 1,mp_deviceBias, 1 ));

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeight, getOutMapsOrNeurons(), getInMapsOrNeurons(), getLayerName() + " mp_deviceWeight3");
}
REGISTER_LAYER_CLASS(FullConn);
