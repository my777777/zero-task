/*
 * Filler.h
 *
 *  Created on: Aug 1, 2016
 *      Author: zys
 */

#ifndef FILLER_H_
#define FILLER_H_

#include "Resource.h"
#include "Shape.h"

class Filler {
public:
	Filler();
	virtual ~Filler();
	virtual void Fill(double* data, Shape& shape) = 0;
};

class ConstantFiller: public Filler {
public:
	ConstantFiller();
	~ConstantFiller();
	void Fill(double* data, Shape& shape);
};

/*
 class UniformFiller : public Filler {
 public:
 UniformFiller();
 ~UniformFiller();
 void Fill(double* data,Shape& shape);
 };

 class XavierFiller : public Filler {
 public:
 XavierFiller();
 ~XavierFiller();
 void Fill(double* data,Shape& shape);
 };
 */
class MSRAFiller: public Filler {

public:
	MSRAFiller();
	~MSRAFiller();
	void Fill(double* data, Shape& shape);
};

#endif /* FILLER_H_ */
