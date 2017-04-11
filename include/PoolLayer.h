/*
 * PoolLayer.h
 *
 *  Created on: 2016-6-14
 *      Author: zys
 */

#ifndef PoolLayer_H_
#define PoolLayer_H_

#include "Resource.h"
#include "Layer.h"

class PoolLayer: public Layer {

private:
	cudnnDataType_t m_dataType;
	cudnnTensorFormat_t m_tensorFormat;
//	cudnnHandle_t m_cudnnHandle;
	cudnnTensorDescriptor_t m_srcTensorDesc, m_dstTensorDesc;
	cudnnPoolingDescriptor_t m_poolingDesc;

private:
	void ExeMemoryAlloc();
	void ExeMemoryRelease();
	void InitLayer(const LayerParam& layerParam);

protected:
	void ExeForward(int) override;
	void ExeBackward(int) override;
	void SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w);

public:
	PoolLayer(const LayerParam& layerParam);
	virtual ~PoolLayer();
};

#endif /* PoolLayer_H_ */
