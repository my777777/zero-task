/*
 * PoolLayer.cpp
 *
 *  Created on: 2016-6-14
 *      Author: zys
 */
#include "PoolLayer.h"
#include "ZeroLogger.h"
#include "ZeroMessage.h"
#include "Filler.h"
#include "Shape.h"
#include "LayerFactory.hpp"

PoolLayer::PoolLayer(const LayerParam& layerParam) :
		Layer(layerParam) {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	InitLayer(layerParam);
}

void PoolLayer::InitLayer(const LayerParam& layerParam) {

	//TODO
	m_dataType = CUDNN_DATA_DOUBLE;
	m_tensorFormat = CUDNN_TENSOR_NCHW;
//	checkCUDNN(cudnnCreate(&m_cudnnHandle));
//	if (mi_streams != 0) {
//		checkCUDNN(cudnnSetStream(m_cudnnHandle, mp_cudaStreams[0]));
//	}
}
PoolLayer::~PoolLayer() {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	checkCudaErrors(cudaSetDevice(getGpuIndex()));
//	checkCUDNN(cudnnDestroy(m_cudnnHandle));
	checkCUDNN(cudnnDestroyPoolingDescriptor(m_poolingDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc));
	ExeMemoryRelease();
}

void PoolLayer::ExeMemoryAlloc() {

	//m_srcTensorDesc
	int n = getFragmentSize();
	int c = getInMapsOrNeurons();
	int h = getPrevLayer()->getMapHeight();
	int w = getPrevLayer()->getMapWidth();
	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc));
	SetTensorDesc(m_srcTensorDesc, n, c, h, w);

	//m_dstTensorDesc
	c = getOutMapsOrNeurons();
	h = getMapHeight();
	w = getMapWidth();
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc));
	SetTensorDesc(m_dstTensorDesc, n, c, h, w);

	//m_poolingDesc
	const int poolDims = 2;
	int windowDimA[poolDims] = { 2, 2 };
	int paddingA[poolDims] = { 0, 0 };
	int strideA[poolDims] = { 2, 2 };
	checkCUDNN(cudnnCreatePoolingDescriptor(&m_poolingDesc));
	checkCUDNN(cudnnSetPoolingNdDescriptor(m_poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, poolDims, windowDimA, paddingA, strideA));

	//mp_deviceOutData
	long outDataSize = gi_batchSize * getOutMapsOrNeurons() * getMapArea();
	checkCudaErrors(cudaMalloc((void**)&mp_deviceOutData, outDataSize * sizeof(double)));

	//mp_deviceOutDiff
	long outDiffSize = gi_batchSize * getInMapsOrNeurons() * getPrevLayer()->getMapArea();
	checkCudaErrors(cudaMalloc((void**)&mp_deviceOutDiff, outDiffSize * sizeof(double)));

	// if data malloc on different devices,need copy.
	if (getPrevLayer()->getGpuIndex() != getGpuIndex()) {
		LOG_IF(INFO,gb_logFlag) << getLayerName() << " forward different device";
		long prevLayerFragmentOutDataSize = getFragmentSize() * getInMapsOrNeurons()*getPrevLayer()->getMapArea();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInData, prevLayerFragmentOutDataSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInData, 0, prevLayerFragmentOutDataSize * sizeof(double)));
	}
	if (getNextLayer()->getGpuIndex() != getGpuIndex()) {
		LOG_IF(INFO,gb_logFlag) << getLayerName() << " backward different device";
		long nextLayerFragmentOutDiffSize = getFragmentSize() * getOutMapsOrNeurons()*getMapArea();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInDiff, nextLayerFragmentOutDiffSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInDiff, 0, nextLayerFragmentOutDiffSize * sizeof(double)));
	}
}

void PoolLayer::ExeMemoryRelease() {
}

void PoolLayer::SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w) {
	const int nDims = 4;
	int dimA[nDims] = { n, c, h, w };
	int strideA[nDims] = { c * h * w, h * w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, m_dataType, 4, dimA, strideA));
}

void PoolLayer::ExeForward(int fragment) {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment+1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	int fragmentSize = getFragmentSize();
	int outMaps = getOutMapsOrNeurons();
	int inMaps = getInMapsOrNeurons();

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInData, fragmentSize,inMaps* getPrevLayer()->getMapArea(), getLayerName() + " inData");

	double alpha = 1.0;
	double beta = 0.0;
	checkCUDNN(cudnnPoolingForward(m_cudnnHandle, //
			m_poolingDesc, //
			&alpha, //
			m_srcTensorDesc, //
			mp_fragmentInData, //
			&beta, //
			m_dstTensorDesc, //
			mp_fragmentOutData//
			));

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outMaps * getMapArea(), getLayerName() + " outData");
}

void PoolLayer::ExeBackward(int fragment) {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment+1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	int outMaps = getOutMapsOrNeurons();
	int fragmentSize = getFragmentSize();
	int inMaps = getInMapsOrNeurons();

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInDiff, fragmentSize, outMaps * getMapArea(), getLayerName() + " inDiff");
	double alpha = 1.0;
	double beta = 0.0;
	checkCUDNN(cudnnPoolingBackward(m_cudnnHandle, m_poolingDesc, //
			&alpha, m_dstTensorDesc, mp_fragmentOutData, //
			m_dstTensorDesc, mp_fragmentInDiff, //
			m_srcTensorDesc, mp_fragmentInData, //
			&beta, m_srcTensorDesc, mp_fragmentOutDiff //
			));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutDiff, fragmentSize, inMaps * getPrevLayer()->getMapArea(), getLayerName() + " outDiff");
}
REGISTER_LAYER_CLASS(Pool);
