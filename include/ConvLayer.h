/*
 * ConvLayer.h
 *
 *  Created on: 2016-6-14
 *      Author: zys
 */

#ifndef NNCONVLAYER_H_
#define NNCONVLAYER_H_

#include "Resource.h"
#include "Layer.h"


class ConvLayer: public Layer {

private:


	cudnnTensorDescriptor_t m_srcTensorDesc=NULL, m_dstTensorDesc=NULL, m_biasTensorDesc=NULL;
	cudnnFilterDescriptor_t m_filterDesc=NULL;
	cudnnConvolutionDescriptor_t m_convDesc=NULL;

	cudnnDataType_t m_dataType;
	cudnnTensorFormat_t m_tensorFormat;
	cudnnConvolutionFwdAlgo_t m_convFwdAlgo;
	cudnnConvolutionBwdDataAlgo_t m_convBwdDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t m_convBwdFilterAlgo;

	size_t m_fwdDataSizeInBytes=0;
	void* m_fwdDataWorkSpace = NULL;

	size_t m_bwdDataSizeInBytes=0;
	void* m_bwdDataWorkSpace = NULL;

	size_t m_bwdFilterSizeInBytes=0;
	void* m_bwdFilterWorkSpace = NULL;


	//cudnnHandle_t m_cudnnHandle;
	////for update gradient
	//cublasHandle_t m_cublasHandle;

private:
	void ExeMemoryAlloc();
	void ExeMemoryRelease();
	void InitLayer(const LayerParam& layerParam);
	void InitWeight(long weightSize);
	void InitBias(long biasSize);

protected:
	void ExeForward(int ) override;
	void ExeBackward(int ) override;
	void ExeUpdate(long ) override;
	void SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc,int n,int c, int h, int w);

public:
	ConvLayer(const LayerParam& layerParam);
	~ConvLayer();
};

#endif /* NNCONVLAYER_H_ */
