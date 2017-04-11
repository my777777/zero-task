/*
 * Layer.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */

#include "Layer.h"
#include "ZeroNet.h"
#include "ZeroMessage.h"
#include "DataLayer.h"

void Layer::Forward(int fragment) {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__ << ",fragmentNumber:" << getFragmentNumber();
	//TODO-----if commented, will appear:
	//Error: an illegal memory access was encountered
	//src/SoftmaxLayer.cpp:151
	struct timeval t1,t2;
	gettimeofday(&t1,NULL);
	checkCudaErrors(cudaSetDevice(getGpuIndex()));

	FowardDataPreparation(fragment);

	ExeForward(fragment);

	CudaStreamSync();

	FowardSendMsg(fragment);

	gettimeofday(&t2, NULL);
	double costTime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" ---->costTime:"<<costTime;
}

void Layer::Backward(int fragment) {
	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__ << ",fragmentNumber:" << getFragmentNumber();
	struct timeval t1,t2,t3,t4,t5;
	gettimeofday(&t1,NULL);
	checkCudaErrors(cudaSetDevice(getGpuIndex()));
	gettimeofday(&t2,NULL);
	double costTime=(t2.tv_sec - t1.tv_sec)*1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" <----costTime1:"<<costTime;
	BackwardDataPreparation(fragment);

	gettimeofday(&t3,NULL);
	costTime=(t3.tv_sec - t2.tv_sec)*1000 + (t3.tv_usec - t2.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" <----costTime2:"<<costTime;
	ExeBackward(fragment);

	gettimeofday(&t4,NULL);
	costTime=(t4.tv_sec - t3.tv_sec)*1000 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" <----costTime3:"<<costTime;

	//CudaStreamSync();

	BackwardSendMsg(fragment);
	gettimeofday(&t5,NULL);
	costTime=(t5.tv_sec - t4.tv_sec)*1000 + (t5.tv_usec - t4.tv_usec) / 1000.0;
	LOG_IF(INFO,gb_logTime)<<__func__<< " "<<getLayerName()<<" <----costTime4:"<<costTime;
}

void Layer::Update(long executedBatchNumber) {

	ExeUpdate(executedBatchNumber);
}

void Layer::FowardDataPreparation(int fragment) {

	if (getLayerType() != LAYER_TYPE_INPUT) {

		Layer* prevLayer = getPrevLayer();
		assert(prevLayer!=NULL);

		int outNeurons = getOutMapsOrNeurons();
		int inNeurons = getInMapsOrNeurons();
		int fragmentSize = getFragmentSize();

		if (getLayerType() == LAYER_TYPE_CONV || getLayerType() == LAYER_TYPE_POOL) {
			outNeurons *= getMapArea();
			inNeurons *= prevLayer->getMapArea();
		}
		int inDataOffset = fragment * fragmentSize * inNeurons;
		int outDataOffset = fragment * fragmentSize * outNeurons;

		//output data of previous layer,also is input data of current layer
		if (isNeedCopyInData()) {
			assert(mp_fragmentInData!=NULL);
			checkCudaErrors(cudaMemcpy(mp_fragmentInData, prevLayer->getDeviceOutData() + inDataOffset, fragmentSize * inNeurons * sizeof(double), cudaMemcpyDeviceToDevice));

		} else {
			mp_fragmentInData = prevLayer->getDeviceOutData() + inDataOffset;
		}
		//out data of current layer
		mp_fragmentOutData = mp_deviceOutData + outDataOffset;

	}

}

void Layer::FowardSendMsg(int fragment) {

	//send msg to next layer
	LayerType layerType = getLayerType();
	if (layerType != LAYER_TYPE_SOFTMAX && getNextLayer()) {
		getNextLayer()->getBindThread()->getMsgQueue().enqueue(new FpropMessage(getLayerId() + 1, fragment));
	}
	if (layerType == LAYER_TYPE_INPUT && (fragment + 1) != getFragmentNumber()) {
		getBindThread()->getMsgQueue().enqueue(new FpropMessage(getLayerId(), fragment + 1));
	}

//	if(gi_phase==0 && layerType== LAYER_TYPE_SOFTMAX && fragment==0){
//		ZeroNet::getInstance().getNetMsgQue().enqueue(new ZeroMessage(NET_CAN_BPROP));
//	}else if(gi_phase==1 && layerType== LAYER_TYPE_SOFTMAX && (fragment + 1) == getFragmentNumber()){//test phase
//		LOG_IF(ERROR,gb_logFlag) << "Forward End,Calculate Accuracy-------------------------";
//		ZeroNet::getInstance().getNetMsgQue().enqueue(new ZeroMessage(NET_CAN_CALACC));
//	}

	//train phase
	if (gi_phase == 0 && layerType == LAYER_TYPE_SOFTMAX && (fragment + 1) == getFragmentNumber()) {
		LOG_IF(ERROR,gb_logFlag) << "Forward End,Begin Backward-------------------------";
		ZeroNet::getInstance().getNetMsgQue().enqueue(new ZeroMessage(NET_CAN_BPROP));
	} else if (gi_phase == 1 && layerType == LAYER_TYPE_SOFTMAX && (fragment + 1) == getFragmentNumber()) {	//test phase
		LOG_IF(ERROR,gb_logFlag) << "Forward End,Calculate Accuracy-------------------------";
		ZeroNet::getInstance().getNetMsgQue().enqueue(new ZeroMessage(NET_CAN_CALACC));
	}
}

void Layer::BackwardDataPreparation(int fragment) {

	if (getLayerType() != LAYER_TYPE_INPUT) {

		Layer* nextLayer = getNextLayer();
		Layer* prevLayer = getPrevLayer();
		assert(prevLayer!= NULL && nextLayer!=NULL);
		int outNeurons = getOutMapsOrNeurons();
		int fragmentSize = getFragmentSize();
		int inNeurons = getInMapsOrNeurons();
		if (getLayerType() == LAYER_TYPE_CONV || getLayerType() == LAYER_TYPE_POOL) {
			outNeurons *= getMapArea();
			inNeurons *= prevLayer->getMapArea();
		}
		int inDataOffset = fragment * fragmentSize * inNeurons;
		int outDataOffset = fragment * fragmentSize * outNeurons;
		int inDiffOffset = outDataOffset;
		int outDiffOffset = inDataOffset;
		//inDiff
		if (isNeedCopyInDiff()) {
			assert(mp_fragmentInDiff!=NULL);
			checkCudaErrors(cudaMemcpy(mp_fragmentInDiff, nextLayer->getDeviceOutDiff() + inDiffOffset, fragmentSize * outNeurons * sizeof(double), cudaMemcpyDeviceToDevice));

		} else {
			mp_fragmentInDiff = nextLayer->getDeviceOutDiff() + inDiffOffset;
		}
		//outDiff
		mp_fragmentOutDiff = mp_deviceOutDiff + outDiffOffset;
		//outData
		mp_fragmentOutData = mp_deviceOutData + outDataOffset;
	}
}

void Layer::BackwardSendMsg(int fragment) {

	if (getPrevLayer()) {
		getPrevLayer()->getBindThread()->getMsgQueue().enqueue(new BpropMessage(getLayerId() - 1, fragment));
	}
	if (getLayerType() == LAYER_TYPE_SOFTMAX && (fragment + 1) != getFragmentNumber()) {
		getBindThread()->getMsgQueue().enqueue(new BpropMessage(getLayerId(), fragment + 1));
	} else if (getLayerType() == LAYER_TYPE_INPUT && (fragment + 1) == getFragmentNumber()) {
		LOG_IF(INFO,gb_logFlag) << "Backward End,Begin Update-------------------------";
		ZeroNet::getInstance().getNetMsgQue().enqueue(new ZeroMessage(NET_CAN_UPDATE));
	}
}

void Layer::ExeUpdate(long executedBatchNumber) {

	LOG_IF(INFO,gb_logFlag) << getLayerName() << ":" << __func__ << ",executedBatchNumber:" << executedBatchNumber;
}

Layer::Layer(const LayerParam& layerParam) {
	InitLayer(layerParam);
}

void Layer::InitLayer(const LayerParam& layerParam) {

	mi_gpuIndex = layerParam.getGpuIndex();
	mi_layerId = layerParam.getLayerId();
	ms_layerName = layerParam.getLayerName();
	me_layerType = layerParam.getLayerType();
	mi_outMapsOrNeurons = layerParam.getOutMapsOrNeurons();
	mi_kernelWidth = layerParam.getKWidth();
	mi_kernelHeight = layerParam.getKHeight();

	checkCudaErrors(cudaSetDevice(mi_gpuIndex));
	if (mi_streams != 0) {
		mp_cudaStreams = new cudaStream_t[mi_streams];
		for (int i = 0; i < mi_streams; ++i) {
			checkCudaErrors(cudaStreamCreate(&(mp_cudaStreams[i])));
		}
	}
//	m_cudnnHandle=ZeroNet::getCudnnHandle();
//	m_cublasHandle=ZeroNet::getCublasHandle();
	checkCUDNN(cudnnCreate(&m_cudnnHandle));
	checkCublasErrors(cublasCreate(&m_cublasHandle));
	if (mi_streams != 0) {
		//if(getLayerType()==LAYER_TYPE_SOFTMAX ){
			checkCUDNN(cudnnSetStream(m_cudnnHandle, mp_cudaStreams[0]));
		//}
		checkCublasErrors(cublasSetStream(m_cublasHandle, mp_cudaStreams[0]));
	}
}

void Layer::CalcMapDims() {

	switch (me_layerType) {
		case LAYER_TYPE_INPUT:
			mi_mapWidth = gi_imageWidth;
			mi_mapHeight = gi_imageHeight;
			break;
		case LAYER_TYPE_CONV: {
			assert(mp_prevLayer!=NULL);
			mi_mapWidth = (mp_prevLayer->mi_mapWidth) + 1 - mi_kernelWidth;
			mi_mapHeight = (mp_prevLayer->mi_mapHeight) + 1 - mi_kernelHeight;
			break;
		}
		case LAYER_TYPE_POOL:
			mi_mapWidth = mp_prevLayer->mi_mapWidth / mi_kernelWidth;
			mi_mapHeight = mp_prevLayer->mi_mapHeight / mi_kernelHeight;
			break;
		default: //LAYER_TYPE_FULL LAYER_TYPE_RELU and so on.
			mi_mapWidth = 1;
			mi_mapHeight = 1;
			break;
	}

	mb_isNeedCopyInDiff = getNextLayer() && (getNextLayer()->getGpuIndex() != getGpuIndex());
	mb_isNeddCopyInData = getPrevLayer() && (getPrevLayer()->getGpuIndex() != getGpuIndex());

}

void Layer::CalcInMapsOrNeurons() {

	switch (me_layerType) {
		case LAYER_TYPE_INPUT:
			mi_inMapsOrNeurons = mi_outMapsOrNeurons;
			break;
		case LAYER_TYPE_CONV:
			assert(mp_prevLayer!=NULL);
			mi_inMapsOrNeurons = mp_prevLayer->mi_outMapsOrNeurons;
			break;
		case LAYER_TYPE_POOL:
			assert(mp_prevLayer!=NULL);
			mi_inMapsOrNeurons = mp_prevLayer->mi_outMapsOrNeurons;
			break;
		case LAYER_TYPE_FULL:
			assert(mp_prevLayer!=NULL);
			CalcFullInNeurons(mp_prevLayer);
			break;
		case LAYER_TYPE_RELU:
			mi_inMapsOrNeurons = mi_outMapsOrNeurons;
			break;
		case LAYER_TYPE_SOFTMAX:
			mi_inMapsOrNeurons = mi_outMapsOrNeurons;
			break;
		default:
			break;
	}

}

void Layer::AllocMemory() {

	CalcMapDims();
	CalcInMapsOrNeurons();
	LOG(INFO)<< getLayerName() << "IDims\tN:" << getBatchSize() << "(" << getFragmentSize() << ")" << "\tC:" << getInMapsOrNeurons() << "\tH:" << getMapHeight()<< "\tW:" << getMapWidth();
	LOG(INFO)<< getLayerName() << "ODims\tN:" << getBatchSize() << "(" << getFragmentSize() << ")" << "\tC:" << getOutMapsOrNeurons() << "\tH:" << getMapHeight()<< "\tW:" << getMapWidth();

	checkCudaErrors(cudaSetDevice(mi_gpuIndex));
	ExeMemoryAlloc();
}

Layer::~Layer() {

	LOG_IF(INFO,gb_logFlag) << this->getLayerName() << ":" << __func__;

	if (mp_deviceOutData != NULL) {
		checkCudaErrors(cudaFree(mp_deviceOutData));
		mp_deviceOutData = NULL;
	}
	if (mp_deviceWeight != NULL) {
		checkCudaErrors(cudaFree(mp_deviceWeight));
		mp_deviceWeight = NULL;
	}

	if (mp_deviceOutDiff != NULL) {
		checkCudaErrors(cudaFree(mp_deviceOutDiff));
		mp_deviceOutDiff = NULL;
	}
	if (mp_deviceWeightGradient != NULL) {
		checkCudaErrors(cudaFree(mp_deviceWeightGradient));
		mp_deviceWeightGradient = NULL;
	}

	if (mp_historyWeightGradient != NULL) {
		checkCudaErrors(cudaFree(mp_historyWeightGradient));
		mp_historyWeightGradient = NULL;
	}
	if (mp_deviceBias != NULL) {
		checkCudaErrors(cudaFree(mp_deviceBias));
		mp_deviceBias = NULL;
	}

	if (mp_deviceBiasGradient != NULL) {
		checkCudaErrors(cudaFree(mp_deviceBiasGradient));
		mp_deviceBiasGradient = NULL;
	}

	if (mp_historyBiasGradient != NULL) {
		checkCudaErrors(cudaFree(mp_historyBiasGradient));
		mp_historyBiasGradient = NULL;
	}

	if (mp_fragmentInDiff != NULL && isNeedCopyInDiff()) {
		checkCudaErrors(cudaFree(mp_fragmentInDiff));
		mp_fragmentInDiff = NULL;
	}

	if (mp_fragmentInData != NULL && isNeedCopyInData()) {
		checkCudaErrors(cudaFree(mp_fragmentInData));
		mp_fragmentInData = NULL;
	}

	checkCudaErrors(cudaSetDevice(mi_gpuIndex));
	if (mi_streams != 0) {
		for (int i = 0; i < mi_streams; ++i) {
			checkCudaErrors(cudaStreamDestroy(mp_cudaStreams[i]));
		}
		delete[] mp_cudaStreams;
	}

	checkCUDNN(cudnnDestroy(m_cudnnHandle));
	checkCublasErrors(cublasDestroy(m_cublasHandle));
}

void Layer::CudaStreamSync(){
	if(mi_streams){
		checkCudaErrors(cudaStreamSynchronize(mp_cudaStreams[0]));
	}else{
		//checkCudaErrors(cudaStreamSynchronize(cudaStreamDefualt));
	}
}
