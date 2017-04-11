/*
 * Shape.cpp
 *
 *  Created on: Aug 1, 2016
 *      Author: zys
 */

#include "Shape.h"

Shape::Shape(vector<int> v) {
	// TODO Auto-generated constructor stub
	mv_shape=v;
	count=1;
	int size=mv_shape.size();
	for(int i=0;i<size;i++){
		count*=mv_shape[i];
	}
}

Shape::Shape(int dims,int* s){

	count=1;
	for(int i=0;i<dims;i++){
		mv_shape.push_back(s[i]);
		count*=mv_shape[i];
	}
}
Shape::~Shape() {
	// TODO Auto-generated destructor stub
}

int Shape::Count(){
	return count;
}


int Shape::Dim(unsigned int d){

	assert(d <mv_shape.size());
	return mv_shape[d];
}
