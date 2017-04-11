/*
 * ZeroTest.cpp
 *
 *  Created on: Dec 26, 2016
 *      Author: zys
 */

#include "ZeroTest.cuh"

ZeroTest::ZeroTest() {
	// TODO Auto-generated constructor stub
	checkCublasErrors(cublasCreate(&m_cublasHandle));

}

ZeroTest::~ZeroTest() {
	// TODO Auto-generated destructor stub
	checkCublasErrors(cublasDestroy(m_cublasHandle));
}



