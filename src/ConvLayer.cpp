/*
 * ConvLayer.cpp
 *
 *  Created on: 2016-6-14
 *      Author: zys
 */
#include "ConvLayer.h"
#include "ZeroLogger.h"
#include "ZeroMessage.h"
#include "Filler.h"
#include "Shape.h"
#include "LayerFactory.hpp"

ConvLayer::ConvLayer(const LayerParam& layerParam) :
		Layer(layerParam) {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	InitLayer(layerParam);
}

void ConvLayer::InitLayer(const LayerParam& layerParam) {

	//TODO
	m_dataType = CUDNN_DATA_DOUBLE;
	m_tensorFormat = CUDNN_TENSOR_NCHW;
	m_convFwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	m_convBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	m_convBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

//	checkCUDNN(cudnnCreate(&m_cudnnHandle));
//	checkCublasErrors(cublasCreate(&m_cublasHandle));
//	if (mi_streams != 0) {
//		checkCUDNN(cudnnSetStream(m_cudnnHandle, mp_cudaStreams[0]));
//		checkCublasErrors(cublasSetStream(m_cublasHandle,mp_cudaStreams[0]));
//	}
}
ConvLayer::~ConvLayer() {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	checkCudaErrors(cudaSetDevice(getGpuIndex()));
//	checkCUDNN(cudnnDestroy(m_cudnnHandle));
//	checkCublasErrors(cublasDestroy(m_cublasHandle));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(m_convDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(m_filterDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_biasTensorDesc));

	ExeMemoryRelease();

}

void ConvLayer::ExeMemoryAlloc() {

	// m_srcTensorDesc
	int n = getFragmentSize();
	int c = getInMapsOrNeurons();
	int h = getPrevLayer()->getMapHeight();
	int w = getPrevLayer()->getMapWidth();
	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc));
	SetTensorDesc(m_srcTensorDesc, n, c, h, w);

	// m_dstTensorDesc
	c = getOutMapsOrNeurons();
	h = getMapHeight();
	w = getMapWidth();
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc));
	SetTensorDesc(m_dstTensorDesc, n, c, h, w);

	//m_filterDesc
	const int tensorDims = 4;
	const int filterDimA[tensorDims] = { getOutMapsOrNeurons(), getInMapsOrNeurons(), getKernelHeight(), getKernelWidth() };
	checkCUDNN(cudnnCreateFilterDescriptor(&m_filterDesc));
	checkCUDNN(cudnnSetFilterNdDescriptor(m_filterDesc, m_dataType, CUDNN_TENSOR_NCHW, tensorDims, filterDimA));

	//m_convDesc
	const int convDims = 2;
	int padA[convDims] = { 0, 0 };
	int filterStrideA[convDims] = { 1, 1 };
	int upscaleA[convDims] = { 1, 1 };
	checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDesc));
	checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, m_dataType));

	// m_biasTensorDesc
	checkCUDNN(cudnnCreateTensorDescriptor(&m_biasTensorDesc));
	SetTensorDesc(m_biasTensorDesc, 1, c, 1, 1);

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnnHandle, m_srcTensorDesc, m_filterDesc, m_convDesc, m_dstTensorDesc, m_convFwdAlgo, &m_fwdDataSizeInBytes));
	if (m_fwdDataSizeInBytes != 0) {
		checkCudaErrors(cudaMalloc(&m_fwdDataWorkSpace, m_fwdDataSizeInBytes));
	}
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(m_cudnnHandle, m_filterDesc, m_dstTensorDesc, m_convDesc, m_srcTensorDesc, m_convBwdDataAlgo, &m_bwdDataSizeInBytes));
	if (m_bwdDataSizeInBytes != 0) {
		checkCudaErrors(cudaMalloc(&m_bwdDataWorkSpace, m_bwdDataSizeInBytes));
	}
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_cudnnHandle, m_srcTensorDesc, m_dstTensorDesc, m_convDesc, m_filterDesc, m_convBwdFilterAlgo, &m_bwdFilterSizeInBytes));
	if (m_bwdFilterSizeInBytes != 0) {
		checkCudaErrors(cudaMalloc(&m_bwdFilterWorkSpace, m_bwdFilterSizeInBytes));
	}

	long weightSize = getKernelArea() * getOutMapsOrNeurons() * getInMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceWeight, weightSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_deviceWeightGradient, weightSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_historyWeightGradient, weightSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_historyWeightGradient, 0, weightSize * sizeof(double)));
	LOG_IF(INFO,gb_logFlag) << " weight size :" << weightSize;
	InitWeight(weightSize);

	long biasSize = getOutMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceBias, biasSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_deviceBiasGradient, biasSize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** )&mp_historyBiasGradient, biasSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_historyBiasGradient, 0, biasSize * sizeof(double)));
	LOG_IF(INFO,gb_logFlag) << " bias size :" << biasSize;
	InitBias(biasSize);

	//mp_deviceOutData
	long outDataSize = gi_batchSize * getOutMapsOrNeurons() * getMapArea();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutData, outDataSize * sizeof(double)));

	//mp_deviceOutDiff
	long outDiffSize = gi_batchSize * getInMapsOrNeurons() * getPrevLayer()->getMapArea();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutDiff, outDiffSize * sizeof(double)));

	// if data malloc on different devices,need copy.
	if (getPrevLayer()->getGpuIndex() != getGpuIndex()) {
		LOG_IF(INFO,gb_logFlag) << getLayerName() << "forward different device";
		long prevLayerFragmentOutDataSize = getFragmentSize() * getInMapsOrNeurons() * getPrevLayer()->getMapArea();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInData, prevLayerFragmentOutDataSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInData, 0, prevLayerFragmentOutDataSize * sizeof(double)));
	}
	if (getNextLayer()->getGpuIndex() != getGpuIndex()) {
		LOG_IF(INFO,gb_logFlag) << getLayerName() << "backward different device";
		long nextLayerFragmentOutDiffSize = getFragmentSize() * getOutMapsOrNeurons() * getMapArea();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInDiff, nextLayerFragmentOutDiffSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInDiff, 0, nextLayerFragmentOutDiffSize * sizeof(double)));
	}

}

void ConvLayer::ExeMemoryRelease() {

	if (m_fwdDataWorkSpace != NULL) {
		checkCudaErrors(cudaFree(m_fwdDataWorkSpace));
		m_fwdDataWorkSpace = NULL;
	}
	if (m_bwdDataWorkSpace != NULL) {
		checkCudaErrors(cudaFree(m_bwdDataWorkSpace));
		m_bwdDataWorkSpace = NULL;
	}
	if (m_bwdFilterWorkSpace != NULL) {
		checkCudaErrors(cudaFree(m_bwdFilterWorkSpace));
		m_bwdFilterWorkSpace = NULL;
	}
}

void ConvLayer::InitWeight(long weightSize) {

	MSRAFiller filler;
	int v[] = { getOutMapsOrNeurons(), getInMapsOrNeurons(), getKernelHeight(), getKernelWidth() };
	Shape shape(4, v);
	filler.Fill(mp_deviceWeight, shape);
}

void ConvLayer::InitBias(long biasSize) {

	ConstantFiller filler;
	int v[] = { getOutMapsOrNeurons() };
	Shape shape(1, v);
	filler.Fill(mp_deviceBias, shape);
}

void ConvLayer::SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w) {
	const int nDims = 4;
	int dimA[nDims] = { n, c, h, w };
	int strideA[nDims] = { c * h * w, h * w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, m_dataType, 4, dimA, strideA));
}

void ConvLayer::ExeForward(int fragment) {

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment + 1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();
	int outMaps = getOutMapsOrNeurons();
	int inMaps = getInMapsOrNeurons();
	int fragmentSize = getFragmentSize();
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInData, fragmentSize, inMaps * getPrevLayer()->getMapArea(), getLayerName() + " inData");
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeight, outMaps, inMaps * getKernelArea(), getLayerName() + " Weight");
	double alpha = 1.0;
	double beta = 0.0;
	checkCUDNN(cudnnConvolutionForward(m_cudnnHandle, //
			&alpha, //
			m_srcTensorDesc, //
			mp_fragmentInData, //the input data of current layer ,also is the output data of preLayer.
			m_filterDesc, //
			mp_deviceWeight, //
			m_convDesc, //
			m_convFwdAlgo, //
			m_fwdDataWorkSpace, //
			m_fwdDataSizeInBytes, //
			&beta, //
			m_dstTensorDesc, //
			mp_fragmentOutData));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outMaps * getMapArea(), getLayerName() + " outData");
	// add bias.
	beta = 1.0;
	checkCUDNN(cudnnAddTensor(m_cudnnHandle, &alpha, m_biasTensorDesc, mp_deviceBias, &beta, m_dstTensorDesc, mp_fragmentOutData));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceBias, 1, outMaps, getLayerName() + " bias");
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outMaps * getMapArea(), getLayerName() + " outData");

	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" -------------->costTime:"<<costTime;

}

void ConvLayer::ExeBackward(int fragment) {

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment + 1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	static int outMaps = getOutMapsOrNeurons();
	static int fragmentSize = getFragmentSize();
	static int inMaps = getInMapsOrNeurons();

	//Calculate outDiff
	double alpha = 1.0;
	double beta = 0.0;
	size_t dataSize = 0;
	void* dataWorkspace = NULL;
	if (getPrevLayer()->getLayerType() != LAYER_TYPE_INPUT) {
		ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInDiff, fragmentSize, outMaps * getMapArea(), getLayerName() + " inDiff");
		checkCUDNN(cudnnConvolutionBackwardData(m_cudnnHandle, //
				&alpha, m_filterDesc, //
				mp_deviceWeight, //
				m_dstTensorDesc, //
				mp_fragmentInDiff, //
				m_convDesc, //
				m_convBwdDataAlgo, //
				dataWorkspace, //
				dataSize, //
				&beta, //
				m_srcTensorDesc, //
				mp_fragmentOutDiff));

		ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutDiff, fragmentSize, inMaps * getPrevLayer()->getMapArea(), getLayerName() + " outDiff");
	}
	//Calculate mp_deviceWeightGradient
	checkCUDNN(cudnnConvolutionBackwardFilter(m_cudnnHandle, //
			&alpha, //
			m_srcTensorDesc, //
			mp_fragmentInData, //
			m_dstTensorDesc, //
			mp_fragmentInDiff, //
			m_convDesc, //
			m_convBwdFilterAlgo, //
			m_bwdFilterWorkSpace, //
			m_bwdFilterSizeInBytes, //
			&beta, //
			m_filterDesc, //
			mp_deviceWeightGradient));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeightGradient, outMaps, inMaps * getKernelArea(), getLayerName() + " WeightGradient");
	//Calculate mp_deviceBiasGradient
	checkCUDNN(cudnnConvolutionBackwardBias(m_cudnnHandle, //
			&alpha, //
			m_dstTensorDesc, //
			mp_fragmentInDiff, //
			&beta, //
			m_biasTensorDesc, //
			mp_deviceBiasGradient //
			));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceBiasGradient, 1, outMaps, getLayerName() + " BiasGradient");

	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" <--------------costTime:"<<costTime;


}

void ConvLayer::ExeUpdate(long executedBatchNumber) {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__ << "; executedBatchNumber " << executedBatchNumber;
	double local_lr1 = gd_lr_mult1 * gd_learnRate * pow(1 + gd_gamma * executedBatchNumber, gd_power);
	double local_lr2 = gd_lr_mult2 * gd_learnRate * pow(1 + gd_gamma * executedBatchNumber, gd_power);
	double alpha = -1;
	long biasSize = getOutMapsOrNeurons();
	long weightSize = getKernelArea() * getOutMapsOrNeurons() * getInMapsOrNeurons();
	//(1) cublasDaxpy: delta = decay*weight + delta
	//checkCublasErrors(cublasDaxpy( m_cublasHandle, weightSize, &gd_decay, mp_deviceWeight, 1, mp_deviceWeightGradient, 1 ));
	//checkCublasErrors(cublasDaxpy( m_cublasHandle, biasSize, &gd_decay, mp_deviceBias, 1, mp_deviceBiasGradient, 1 ));

//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeight, 1, weightSize, getLayerName() + " mp_deviceWeight1");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_historyWeightGradient, 1, weightSize, getLayerName() + " mp_historyWeightGradient1");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeightGradient, 1, weightSize, getLayerName() + " mp_deviceWeightGradient1");

	//	ZeroLogger::LogIfInfo(true, mp_deviceBias, 1, biasSize, getLayerName() + " mp_deviceBias1");
	//	ZeroLogger::LogIfInfo(true, mp_historyBiasGradient, 1, biasSize, getLayerName() + " mp_historyBiasGradient1");
	//	ZeroLogger::LogIfInfo(true, mp_deviceBiasGradient, 1, biasSize, getLayerName() + " mp_deviceBiasGradient1");

	//(2) cublasDgeam: history = learn_rate*delta + momentum*history
	checkCublasErrors(cublasDgeam(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, weightSize, 1, &gd_moment, //
			mp_historyWeightGradient, weightSize, &local_lr1, mp_deviceWeightGradient, weightSize, mp_historyWeightGradient, weightSize));

	checkCublasErrors(cublasDgeam(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, biasSize, 1, &gd_moment, //
			mp_historyBiasGradient, biasSize, &local_lr2, mp_deviceBiasGradient, biasSize, mp_historyBiasGradient, biasSize));

//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_historyWeightGradient, 1, weightSize, getLayerName() + " mp_historyWeightGradient2");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeightGradient, 1, weightSize, getLayerName() + " mp_deviceWeightGradient2");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_historyBiasGradient, 1, biasSize, getLayerName() + " mp_historyBiasGradient2");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceBiasGradient, 1, biasSize, getLayerName() + " mp_deviceBiasGradient2");

	//(3) cublasDaxpy: weight =weight +(-1) *history
	checkCublasErrors(cublasDaxpy( m_cublasHandle, weightSize, &alpha, mp_historyWeightGradient, 1,mp_deviceWeight, 1 ));
	//(3) cublasDaxpy: bias =bias +(-1) *history
	checkCublasErrors(cublasDaxpy( m_cublasHandle, biasSize, &alpha, mp_historyBiasGradient, 1,mp_deviceBias, 1 ));

//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceWeight, 1, weightSize, getLayerName() + " mp_deviceWeight3");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_historyWeightGradient, 1, biasSize, getLayerName() + " mp_historyWeightGradient3");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_deviceBias, 1, biasSize, getLayerName() + " mp_deviceBias3");
//	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_historyBiasGradient, 1, biasSize, getLayerName() + " mp_historyBiasGradient3");

}

REGISTER_LAYER_CLASS(Conv);
