#ifndef  _SOFTMAXLAYER_H_
#define _SOFTMAXLAYER_H_

#include "Resource.h"
#include "Layer.h"

class SoftmaxLayer: public Layer {
private:

	//cudnnHandle_t m_cudnnHandle;
	//cublasHandle_t m_cublasHandle;


	cudnnTensorDescriptor_t m_srcTensorDesc, m_dstTensorDesc;
	cudnnDataType_t m_dataType;



private:
	void ExeMemoryAlloc();
	void ExeMemoryRelease();
	void SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, int n, int c, int h, int w);
	void InitLayer(const LayerParam& layerParam);
protected:
	void ExeForward(int );
	void ExeBackward(int );
public:
	SoftmaxLayer(const LayerParam& layerParam);
	~SoftmaxLayer();
};
#endif
