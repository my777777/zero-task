/*
 * FullConnLayer.h
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */

#ifndef FULLCONNLAYER_H_
#define FULLCONNLAYER_H_

#include "Resource.h"
#include "Layer.h"

class FullConnLayer: public Layer {

private:
	//cublasHandle_t m_cublasHandle;
	// unit vector,dims(mp_biasMultiplier)=gi_batchSize
	double* mp_biasMultiplier=NULL;

private:
	void ExeMemoryAlloc();
	void ExeMemoryRelease();
	void InitLayer(const LayerParam& layerParam);
	void InitWeight(long weightSize);
	void InitBias(long biasSize);
	void InitBiasMultiplier(long biasMultiplierSize);

protected:
	void ExeForward(int );
	void ExeBackward(int );
	void ExeUpdate(long );

public:
	FullConnLayer(const LayerParam& layerParam);
	virtual ~FullConnLayer();

};

#endif /* FULLCONNLAYER_H_ */
