#include "ReLULayer.h"
#include "ZeroLogger.h"
#include "ZeroMessage.h"
#include "FileHandle.h"
#include "Filler.h"
#include "Shape.h"
#include "LayerFactory.hpp"

ReLULayer::ReLULayer(const LayerParam& layerParam) :
		Layer(layerParam) {
	LOG_IF(INFO, gb_logFlag) << getLayerName() << ":" << __func__;
	InitLayer(layerParam);
}

void ReLULayer::InitLayer(const LayerParam& layerParam) {

	m_dataType = CUDNN_DATA_DOUBLE;
//	checkCUDNN(cudnnCreate(&m_cudnnHandle));
//	if (mi_streams != 0) {
//		checkCUDNN(cudnnSetStream(m_cudnnHandle, mp_cudaStreams[0]));
//	}

	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc));
	//v5
	checkCUDNN(cudnnCreateActivationDescriptor(&activationtype));
	checkCUDNN(cudnnSetActivationDescriptor(activationtype, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
}
ReLULayer::~ReLULayer() {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc));
	checkCUDNN(cudnnDestroyActivationDescriptor(activationtype));
//	checkCUDNN(cudnnDestroy(m_cudnnHandle));

	ExeMemoryRelease();
}

void ReLULayer::ExeMemoryAlloc() {

	assert(getInMapsOrNeurons() && getOutMapsOrNeurons() && getMapHeight() && getMapWidth());

	int n = getFragmentSize();
	int c = getInMapsOrNeurons();
	int h = getMapHeight();
	int w = getMapWidth();
	SetTensorDesc(m_srcTensorDesc, n, c, h, w);
	c = getOutMapsOrNeurons();
	SetTensorDesc(m_dstTensorDesc, n, c, h, w);

	long outDataSize = gi_batchSize * getOutMapsOrNeurons();
	checkCudaErrors(cudaMalloc(&mp_deviceOutData, outDataSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_deviceOutData, 0, outDataSize * sizeof(double)));

	long outDiffSize = gi_batchSize * getInMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutDiff, outDiffSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_deviceOutDiff, 0, outDiffSize * sizeof(double)));

// if data malloc on different devices,need copy.
	if (getPrevLayer()->getGpuIndex() != getGpuIndex()) {
		long prevLayerFragmentOutDataSize = getFragmentSize() * getInMapsOrNeurons();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInData, prevLayerFragmentOutDataSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInData, 0, prevLayerFragmentOutDataSize * sizeof(double)));
	}
	if (getNextLayer()->getGpuIndex() != getGpuIndex()) {
		long nextLayerFragmentOutDiffSize = getFragmentSize() * getOutMapsOrNeurons();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInDiff, nextLayerFragmentOutDiffSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInDiff, 0, nextLayerFragmentOutDiffSize * sizeof(double)));
	}

}

void ReLULayer::ExeMemoryRelease() {

}

void ReLULayer::SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w) {
	const int nDims = 4;
	int dimA[nDims] = { n, c, h, w };
	int strideA[nDims] = { c * h * w, h * w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, m_dataType, 4, dimA, strideA));
}

void ReLULayer::ExeForward(int fragment) {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment + 1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	int outNeurons = getOutMapsOrNeurons();
	int fragmentSize = getFragmentSize();

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outNeurons, getLayerName() + " outData:");
	//V5
	double alpha = 1.0;
	double beta = 0.0;
	checkCUDNN(cudnnActivationForward(m_cudnnHandle, activationtype, //
			&alpha, m_srcTensorDesc, mp_fragmentInData, //
			&beta, m_dstTensorDesc, mp_fragmentOutData));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outNeurons, getLayerName() + " outData:");
}

void ReLULayer::ExeBackward(int fragment) {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment + 1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();
	int outNeurons = getOutMapsOrNeurons();
	int fragmentSize = getFragmentSize();
	int inNeurons = getInMapsOrNeurons();
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInDiff, fragmentSize, outNeurons, getLayerName() + " inDiff");
	double alpha = 1.0;
	double beta = 0.0;
	checkCUDNN(cudnnActivationBackward(m_cudnnHandle, activationtype, //
			&alpha, m_dstTensorDesc, mp_fragmentOutDiff, //
			m_dstTensorDesc, mp_fragmentInDiff, //
			m_srcTensorDesc, mp_fragmentInData, //
			&beta, m_srcTensorDesc, mp_fragmentOutDiff));

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutDiff, fragmentSize, inNeurons, getLayerName() + " outDiff");
}
REGISTER_LAYER_CLASS(ReLU);
