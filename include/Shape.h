/*
 * Shape.h
 *
 *  Created on: Aug 1, 2016
 *      Author: zys
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include "Resource.h"

class Shape {
private:
	vector<int> mv_shape;
	int count;
public:
	Shape(vector<int> v);
	Shape(int dims,int* s);
	int Count();
	int Dim(unsigned int d);
	virtual ~Shape();
};

#endif /* SHAPE_H_ */
