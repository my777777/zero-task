/*
 * MultiCard.hpp
 *
 *  Created on: Sep 7, 2016
 *      Author: zys
 */

#ifndef MULTICARD_HPP_
#define MULTICARD_HPP_

#include "Resource.h"

namespace mcard {

class MultiCard {

public:

	virtual ~MultiCard();
	void Train();

	static MultiCard& Instance();
	static cublasHandle_t& CublasHandle();

private:
	MultiCard();
	cublasHandle_t m_cublasHandle;

};

} /* namespace mcard */

#endif /* MULTICARD_HPP_ */
