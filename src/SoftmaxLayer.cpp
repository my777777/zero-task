#include "SoftmaxLayer.h"
#include "ZeroLogger.h"
#include "ZeroMessage.h"
#include "FileHandle.h"
#include "Filler.h"
#include "Shape.h"
#include "DataLayer.h"
#include "LayerFactory.hpp"

SoftmaxLayer::SoftmaxLayer(const LayerParam& layerParam) :
		Layer(layerParam) {
	LOG_IF(INFO,gb_logFlag) << this->getLayerName() << ":" << __func__;
	InitLayer(layerParam);
}

void SoftmaxLayer::InitLayer(const LayerParam& layerParam) {

	m_dataType = CUDNN_DATA_DOUBLE;
//	checkCUDNN(cudnnCreate(&m_cudnnHandle));
//	checkCublasErrors(cublasCreate(&m_cublasHandle));
//	if (mi_streams != 0) {
//		checkCUDNN(cudnnSetStream(m_cudnnHandle, mp_cudaStreams[0]));
//		checkCublasErrors(cublasSetStream(m_cublasHandle, mp_cudaStreams[0]));
//	}
	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc));
}
SoftmaxLayer::~SoftmaxLayer() {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__;
	checkCudaErrors(cudaSetDevice(getGpuIndex()));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc));
//	checkCUDNN(cudnnDestroy(m_cudnnHandle));
//	cublasDestroy(m_cublasHandle);
	ExeMemoryRelease();
}

void SoftmaxLayer::ExeMemoryAlloc() {

	assert(getInMapsOrNeurons() && getOutMapsOrNeurons() && getMapHeight() && getMapWidth());

	int n = getFragmentSize();
	int c = getInMapsOrNeurons();
	int h = getMapHeight();
	int w = getMapWidth();
	SetTensorDesc(m_srcTensorDesc, n, c, h, w);
	c = getOutMapsOrNeurons();
	SetTensorDesc(m_dstTensorDesc, n, c, h, w);

	long outDataSize = gi_batchSize * getOutMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutData, outDataSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_deviceOutData, 0, outDataSize * sizeof(double)));

	long outDiffSize = gi_batchSize * getInMapsOrNeurons();
	checkCudaErrors(cudaMalloc((void** )&mp_deviceOutDiff, outDiffSize * sizeof(double)));
	checkCudaErrors(cudaMemset(mp_deviceOutDiff, 0, outDiffSize * sizeof(double)));

	// if data malloc on different devices,need copy.
	if (getPrevLayer() && getPrevLayer()->getGpuIndex() != getGpuIndex()) {
		long prevLayerFragmentOutDataSize = getFragmentSize() * getInMapsOrNeurons();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInData, prevLayerFragmentOutDataSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInData, 0, prevLayerFragmentOutDataSize * sizeof(double)));
	}
	if (getNextLayer() && getNextLayer()->getGpuIndex() != getGpuIndex()) {
		long nextLayerFragmentInDiffSize = getFragmentSize() * getOutMapsOrNeurons();
		checkCudaErrors(cudaMalloc((void** )&mp_fragmentInDiff, nextLayerFragmentInDiffSize * sizeof(double)));
		checkCudaErrors(cudaMemset(mp_fragmentInDiff, 0, nextLayerFragmentInDiffSize * sizeof(double)));
	}

}

void SoftmaxLayer::ExeMemoryRelease() {

}

void SoftmaxLayer::SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w) {
	const int nDims = 4;
	int dimA[nDims] = { n, c, h, w };
	int strideA[nDims] = { c * h * w, h * w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, m_dataType, 4, dimA, strideA));
}

void SoftmaxLayer::ExeForward(int fragment) {

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment+1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();

	int outNeurons = getOutMapsOrNeurons();
	int fragmentSize = getFragmentSize();
	int inNeurons = getInMapsOrNeurons();

	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInData, fragmentSize, inNeurons, getLayerName() + " inData:");
	double alpha = 1.0;
	double beta = 0.0;
	checkCUDNN(cudnnSoftmaxForward(m_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, //
			&alpha, m_srcTensorDesc, mp_fragmentInData, //
			&beta, m_dstTensorDesc, mp_fragmentOutData));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutData, fragmentSize, outNeurons, getLayerName() + " outData:");

	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" -------------->costTime:"<<costTime;
}


void SoftmaxLayer::ExeBackward(int fragment) {

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	LOG_IF(INFO,gb_logFlag) << getLayerName() << "_" << getGpuIndex() << ":" << __func__ << "; cur_fragment " << fragment+1 << ";total_fragmentNumber:" << getFragmentNumber() << ";fragmentDataSize:" << getFragmentSize();
	int outNeurons = getOutMapsOrNeurons();
	int fragmentSize = getFragmentSize();
	int inNeurons = getInMapsOrNeurons();
	double alpha = -1.0;
	double beta = -1.0 / fragmentSize;
	checkCublasErrors(cublasDaxpy(m_cublasHandle, fragmentSize * gi_mnistNumberOfClass, &alpha, mp_fragmentOutData, 1, mp_fragmentInDiff, 1));
	cublasDscal(m_cublasHandle, fragmentSize * gi_mnistNumberOfClass, &beta, mp_fragmentInDiff, 1);
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentInDiff, fragmentSize, outNeurons, getLayerName() + " inDiff");
	checkCudaErrors(cudaMemcpy(mp_fragmentOutDiff,mp_fragmentInDiff, fragmentSize * inNeurons * sizeof(double), cudaMemcpyDeviceToDevice));
	ZeroLogger::LogIfInfo(gb_logDataFlag, mp_fragmentOutDiff, fragmentSize, inNeurons, getLayerName() + " outDiff");

	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" <--------------costTime:"<<costTime;

}
REGISTER_LAYER_CLASS(Softmax);
