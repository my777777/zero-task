/*
 * ReLULayer.h
 *
 *  Created on: 2016-6-14
 *      Author: zys
 */

#ifndef ReLULayer_H_
#define ReLULayer_H_

#include"Resource.h"
#include "Layer.h"

class ReLULayer: public Layer {

private:
	//cudnnHandle_t m_cudnnHandle;
	cudnnTensorDescriptor_t m_srcTensorDesc, m_dstTensorDesc;
	cudnnDataType_t m_dataType;
	cudnnActivationDescriptor_t activationtype;

private:
	void ExeMemoryAlloc();
	void ExeMemoryRelease();
	void InitLayer(const LayerParam& layerParam);

protected:
	void ExeForward(int) override;
	void ExeBackward(int) override;
	void SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w);

public:

	ReLULayer(const LayerParam& layerParam);
	~ReLULayer();
};

#endif /* ReLULayer_H_ */
