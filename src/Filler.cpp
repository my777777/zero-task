/*
 * Filler.cpp
 *
 *  Created on: Aug 1, 2016
 *      Author: zys
 */

#include "Filler.h"
#include <curand.h>

Filler::Filler() {
	// TODO Auto-generated constructor stub

}

Filler::~Filler() {
	// TODO Auto-generated destructor stub
}

//ConstantFiller
ConstantFiller::ConstantFiller() :
		Filler() {

}

ConstantFiller::~ConstantFiller() {

}

void ConstantFiller::Fill(double* data, Shape& shape) {
	int count = shape.Count();
	checkCudaErrors(cudaMemset(data, 0, count * sizeof(double)));
}
/*

 UniformFiller::UniformFiller():Filler(){

 }

 UniformFiller::~UniformFiller(){

 }

 void UniformFiller::Fill(double* data,Shape& shape){

 }

 XavierFiller::XavierFiller():Filler(){

 }

 XavierFiller::~XavierFiller(){

 }

 void XavierFiller::Fill(double* data,Shape& shape){
 int count=shape.Count();
 double fan_in=(1.0*count)/shape.Dim(0);
 double scale = sqrt(3.0/ fan_in);

 //TODO
 //Unfinished
 }
 */
MSRAFiller::MSRAFiller() :
		Filler() {

}

MSRAFiller::~MSRAFiller() {

}

void MSRAFiller::Fill(double* data, Shape& shape) {

	int count = shape.Count();
	double fan_in = (1.0 * count) / shape.Dim(0);
	double std = sqrt(2.0 / fan_in);

	curandGenerator_t curand_generator;

	checkCuRandErrors(curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT));

	int random = rand() % RAND_MAX;
	checkCuRandErrors(curandSetPseudoRandomGeneratorSeed(curand_generator, random));

	LOG_IF(INFO,gb_logFlag)<<"count:"<<count;
	checkCuRandErrors(curandGenerateNormalDouble(curand_generator, data, count, 0, std));

}
