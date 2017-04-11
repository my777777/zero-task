/*
 * ZeroTest.h
 *
 *  Created on: Dec 26, 2016
 *      Author: zys
 */

#ifndef ZEROTEST_H_
#define ZEROTEST_H_

#include "Resource.h"

class ZeroTest {

private:
/*	int m_convAlgorithm;
	cudnnDataType_t m_dataType = CUDNN_DATA_DOUBLE;
	cudnnTensorFormat_t m_tensorFormat;
	cudnnHandle_t m_cudnnHandle;
	cudnnTensorDescriptor_t m_srcTensorDesc, m_dstTensorDesc, m_biasTensorDesc;
	cudnnFilterDescriptor_t m_filterDesc;
	cudnnConvolutionDescriptor_t m_convDesc;*/

	cublasHandle_t m_cublasHandle;

public:
	ZeroTest();

	virtual ~ZeroTest();

	//__global__ void fillKernel(int *a, int n, int offset);

//	void testSoftmaxParallel();
//
//	void testFullConnParallel();
//
//	void testFB();
};

__global__ void fillKernel(int *a,long n, long offset);

__global__ void fillKernel2(int *a,long n, long offset);

#endif /* ZEROTEST_H_ */
