/*
 * ZeroNet.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */
#include "ZeroNet.h"
#include "Layer.h"
#include "ZeroMessage.h"
#include "DataLayer.h"
#include "FullConnLayer.h"
#include "SoftmaxLayer.h"
#include "ReLULayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "LayerFactory.hpp"
#include "ZeroLogger.h"

ZeroNet::ZeroNet() :
		Thread(true, "ZeroNet") {

	LOG_IF(INFO,gb_logFlag) << __func__;

	CreateLayers();
	ConnectAdjacentLayers();
	AllocLayerMemory();

	mp_netParam = new NetParam(gi_batchSize, gi_fragmentNumber, gi_totalIterNumber, mv_layerVec.size());
	mp_sync = new ThreadSynchronizer(mv_zeroThreadVec.size() + 1);
	//first batch
	mi_executedIterNumber = 0;
	mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_FPROP));


	gettimeofday(&mt_startTime, NULL);
	mv_accuracyVec.push_back(0);
}
ZeroNet::~ZeroNet() {

	LOG_IF(INFO,gb_logFlag) << __func__;
	for (vector<Layer*>::iterator layerIt = mv_layerVec.begin(); layerIt != mv_layerVec.end(); ++layerIt) {
		if (*layerIt != NULL) {
			delete *layerIt;
			*layerIt = NULL;
		}
	}

	for (vector<LayerParam*>::iterator layerParamIt = mv_layerParamVec.begin(); layerParamIt != mv_layerParamVec.end(); ++layerParamIt) {
		if (*layerParamIt != NULL) {
			delete *layerParamIt;
			*layerParamIt = NULL;
		}
	}

	for (vector<ZeroThread*>::iterator zeroThreadIt = mv_zeroThreadVec.begin(); zeroThreadIt != mv_zeroThreadVec.end(); ++zeroThreadIt) {
		(*zeroThreadIt)->getMsgQueue().enqueue(new ZeroMessage(LAYER_CAN_EXIT));
		(*zeroThreadIt)->join();
		delete *zeroThreadIt;
		*zeroThreadIt = NULL;
	}
	delete mp_sync;
	delete mp_netParam;

	LOG_IF(INFO,gb_logFlag) << "~ZeroNet()------------------END";

}

void ZeroNet::CreateLayers() {

	int gpu0 = 0;
	int gpu1 = gb_multiGpus ? 1 : 0;
	int layerId = 0;
	LayerParam* dataLayerParam = new LayerParam(layerId++, "data", LAYER_TYPE_INPUT, gi_datachannel_out, gpu0);
	mv_layerParamVec.push_back(dataLayerParam);

	LayerParam* conv1LayerParam = new LayerParam(layerId++, "conv1", LAYER_TYPE_CONV, gi_conv1_out, gpu0);
	mv_layerParamVec.push_back(conv1LayerParam);

	LayerParam* pool1LayerParam = new LayerParam(layerId++, "pool1", LAYER_TYPE_POOL, gi_conv1_out, gpu0);
	mv_layerParamVec.push_back(pool1LayerParam);

	LayerParam* conv2LayerParam = new LayerParam(layerId++, "conv2", LAYER_TYPE_CONV, gi_conv2_out, gpu1);
	mv_layerParamVec.push_back(conv2LayerParam);

	LayerParam* pool2LayerParam = new LayerParam(layerId++, "pool2", LAYER_TYPE_POOL, gi_conv2_out, gpu1);
	mv_layerParamVec.push_back(pool2LayerParam);

	LayerParam* fc1LayerParam = new LayerParam(layerId++, "fc1", LAYER_TYPE_FULL, gi_fc1_out, gpu1);
	mv_layerParamVec.push_back(fc1LayerParam);

	LayerParam* reLU1LayerParam = new LayerParam(layerId++, "relu1", LAYER_TYPE_RELU, gi_fc1_out, gpu1);
	mv_layerParamVec.push_back(reLU1LayerParam);

	LayerParam* fc2LayerParam = new LayerParam(layerId++, "fc2", LAYER_TYPE_FULL, gi_fc2_out, gpu1);
	mv_layerParamVec.push_back(fc2LayerParam);

	LayerParam* softmaxLayerParam = new LayerParam(layerId++, "softmax", LAYER_TYPE_SOFTMAX, gi_fc2_out, gpu1);
	mv_layerParamVec.push_back(softmaxLayerParam);

//	LayerParam* fc1LayerParam = new LayerParam(layerId++, "fc1", LAYER_TYPE_FULL, gi_fc1_out, gpu0);
//	mv_layerParamVec.push_back(fc1LayerParam);
//
//	LayerParam* fc2LayerParam = new LayerParam(layerId++, "fc2", LAYER_TYPE_FULL, gi_fc1_out, gpu0);
//	mv_layerParamVec.push_back(fc2LayerParam);
//
//	LayerParam* fc3LayerParam = new LayerParam(layerId++, "fc3", LAYER_TYPE_FULL, gi_fc1_out, gpu1);
//	mv_layerParamVec.push_back(fc3LayerParam);
//
//	LayerParam* fc4LayerParam = new LayerParam(layerId++, "fc4", LAYER_TYPE_FULL, gi_fc1_out, gpu1);
//	mv_layerParamVec.push_back(fc4LayerParam);
//
//	LayerParam* fc5LayerParam = new LayerParam(layerId++, "fc5", LAYER_TYPE_FULL, gi_mnistNumberOfClass, gpu1);
//	mv_layerParamVec.push_back(fc5LayerParam);
//
//	LayerParam* softmaxLayerParam = new LayerParam(layerId++, "softmax", LAYER_TYPE_SOFTMAX, gi_mnistNumberOfClass, gpu1);
//	mv_layerParamVec.push_back(softmaxLayerParam);

	//create thread
	int threadNumber = mv_layerParamVec.size();
	//int threadNumber=2;
	for (int i = 0; i < threadNumber; i++) {
		ZeroThread* zeroThread = new ZeroThread();
		mv_zeroThreadVec.push_back(zeroThread);
	}
	//create layer
	int layerNumber = mv_layerParamVec.size();
	for (int i = 0; i < layerNumber; ++i) {
		Layer* layer = LayerFactory::CreateLayer(*(mv_layerParamVec[i]));
		layer->setBindThread(mv_zeroThreadVec[i]);
		mv_layerVec.push_back(layer);
	}
	mp_dataLayer = dynamic_cast<DataLayer*>(mv_layerVec[0]);
	mp_softmaxLayer = dynamic_cast<SoftmaxLayer*>(mv_layerVec[layerNumber - 1]);
	assert(mp_dataLayer!=NULL && mp_softmaxLayer!=NULL);
}

void ZeroNet::ConnectAdjacentLayers() {

	for (unsigned int index = 0; index < mv_layerVec.size(); ++index) {
		if (index == 0) {
			mv_layerVec[index]->ConnectPrevLayer(NULL);
			mv_layerVec[index]->ConnectNextLayer(mv_layerVec[index + 1]);
			continue;
		} else if (index == mv_layerVec.size() - 1) {
			mv_layerVec[index]->ConnectPrevLayer(mv_layerVec[index - 1]);
			//the next layer of the last layer is : dataLayer
			mv_layerVec[index]->ConnectNextLayer(mv_layerVec[0]);
			continue;
		}
		mv_layerVec[index]->ConnectPrevLayer(mv_layerVec[index - 1]);
		mv_layerVec[index]->ConnectNextLayer(mv_layerVec[index + 1]);
	}
}

void ZeroNet::AllocLayerMemory() {

	for (unsigned int index = 0; index < mv_layerVec.size(); ++index) {
		mv_layerVec[index]->AllocMemory();
	}
}

void* ZeroNet::run() {

	LOG_IF(INFO,gb_logFlag) << __func__;

	for (vector<ZeroThread*>::iterator it = mv_zeroThreadVec.begin(); it != mv_zeroThreadVec.end(); ++it) {
		(*it)->start();
	}
	bool exit = false;


	while (!exit) {
		ZeroMessage* m = mq_netMsgQue.dequeue();
		MESSAGES type = m->getType();
		if (type == NET_CAN_FPROP) {
			NetForward();
		} else if (type == NET_CAN_BPROP) {
			NetBackward();
		} else if (type == NET_CAN_UPDATE) {
			NetParallelUpdate();
		} else if (type == NET_CAN_SYNC) {
			NetSyncWithChildThreads();
		} else if (type == NET_CAN_TEST) {
			NetBeginTest();
			//NetEndTest();
		} else if (type == NET_END_TEST) {
			NetEndTest();
		} else if (type == NET_CAN_CALACC) {
			NetCalTestAccuracy();
		} else if (type == NET_CAN_EXIT) {
			LOG_IF(ERROR,gb_logFlag) << "EXIT_BEGIN--------------------";
			exit = true;
		}
		delete m;
	}
	return NULL;
}

void ZeroNet::NetForward() {


	gettimeofday(&mt_iterStartTime, NULL);
	gettimeofday(&mt_beginForward, NULL);

	LOG_IF(ERROR,gb_logFlag) << "FPROP_BEGIN--------------------";
	//Why Can Not :mp_dataLayer->Forward()??
	mp_dataLayer->PrefetchNextBatch(gi_phase);
	mp_dataLayer->getBindThread()->getMsgQueue().enqueue(new FpropMessage(mp_dataLayer->getLayerId()));

}

void ZeroNet::NetBackward() {

	gettimeofday(&mt_beginBackward, NULL);
	double costTime=(mt_beginBackward.tv_sec - mt_beginForward.tv_sec)*1000 + (mt_beginBackward.tv_usec - mt_beginForward.tv_usec) / 1000.0;
	if(!gi_phase) {
		LOG_IF(INFO,gb_logTime)<<"Iter #"<<mi_executedIterNumber<<" End.>>>Forward_CostTiime "<<costTime << "(ms)";
		mv_totalTime[0]+=costTime;
	}

	LOG_IF(ERROR,gb_logFlag) << "BPROP_BEGIN--------------------";
	static int len = mv_layerVec.size();
	Layer* layer = mv_layerVec[len - 1];
	//Why Can Not :layer->Backward()??
	layer->getBindThread()->getMsgQueue().enqueue(new BpropMessage(layer->getLayerId()));
}

void ZeroNet::NetParallelUpdate() {

	gettimeofday(&mt_beginUpdate, NULL);
	double costTime=(mt_beginUpdate.tv_sec - mt_beginBackward.tv_sec)*1000 + (mt_beginUpdate.tv_usec - mt_beginBackward.tv_usec) / 1000.0;
	if(!gi_phase) {
		LOG_IF(INFO,gb_logTime)<<"Iter #"<<mi_executedIterNumber<<" End.<<<Backward_CostTiime "<<costTime << "(ms)";
		mv_totalTime[1]+=costTime;
	}

	LOG_IF(ERROR,gb_logFlag) << "UPDATE_BEGIN_PARALLEL--------------------";
	for (unsigned int index = 0; index < mv_layerVec.size(); ++index) {
		Layer* layer = mv_layerVec[index];
		layer->getBindThread()->getMsgQueue().enqueue(new UpdateMessage(layer->getLayerId()));
	}
	mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_SYNC));
}

void ZeroNet::NetSyncWithChildThreads() {
	LOG_IF(ERROR,gb_logFlag) << "SYNC_BEGIN--------------------";
	for (vector<ZeroThread*>::iterator it = mv_zeroThreadVec.begin(); it != mv_zeroThreadVec.end(); ++it) {
		(*it)->getMsgQueue().enqueue(new ZeroMessage(LAYER_CAN_SYNC));
	}
	mp_sync->sync();
	gettimeofday(&mt_iterEndTime,NULL);
	double costTime2=(mt_iterEndTime.tv_sec - mt_beginUpdate.tv_sec)*1000 + (mt_iterEndTime.tv_usec - mt_beginUpdate.tv_usec) / 1000.0;
	if(!gi_phase){
		LOG_IF(INFO,gb_logTime)<<"Iter #"<<mi_executedIterNumber<<" End.^^^Update_CostTiime "<<costTime2 << "(ms)";
		mv_totalTime[2]+=costTime2;
	}

	double costTime=(mt_iterEndTime.tv_sec - mt_iterStartTime.tv_sec)*1000 + (mt_iterEndTime.tv_usec - mt_iterStartTime.tv_usec) / 1000.0;
	if(!gi_phase) {
		LOG_IF(INFO,gb_logTime)<<"Iter #"<<mi_executedIterNumber<<" End.*****CostTiime "<<costTime << "(ms)";
		mv_totalTime[3]+=costTime;
		LOG_IF(INFO,gb_logTime)<<"";
	}
	++mi_executedIterNumber;
	static int totalIterNumber = getNetParam()->getTotalIterNumber();
	//LOG(ERROR)<< "SYNC_END--------------------totalIterNumber:" << totalIterNumber << ";ExecutedBatchNumber:" << mi_executedIterNumber;
	if ((mi_executedIterNumber % gi_testInterval) == 0) {//begin to test
		mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_TEST));
	} else if (mi_executedIterNumber < totalIterNumber) {// continue to fprop
		mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_FPROP));
	} else { // exit the net
		mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_EXIT));
	}
}

void ZeroNet::NetBeginTest() {
	LOG_IF(ERROR,gb_logFlag) << "NET_TESTING-------------------";
	gi_phase = 1; //TEST
	mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_FPROP));
}
void ZeroNet::NetEndTest() {
	gi_phase = 0; //TRAIN
	if (mi_executedIterNumber < getNetParam()->getTotalIterNumber()) {
		mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_FPROP));
	} else {
		mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_EXIT));
	}
}
void ZeroNet::NetCalTestAccuracy() {
	LOG_IF(ERROR,gb_logFlag) << "NetCalTestAccuracy-------------------";
	static long totalRightCounter = 0;
	static long testedIterNumber = 0;
	static long testedEpoch = 1;
	double *pTargetOut = new double[gi_batchSize * gi_mnistNumberOfClass];
	double *pActualOut = new double[gi_batchSize * gi_mnistNumberOfClass];

	checkCudaErrors(cudaMemcpy(pTargetOut, mp_dataLayer->getDeviceOutDiff(), gi_batchSize * gi_mnistNumberOfClass * sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pActualOut, mp_softmaxLayer->getDeviceOutData(), gi_batchSize * gi_mnistNumberOfClass * sizeof(double), cudaMemcpyDeviceToHost));
//	ZeroLogger::LogIfInfoHost(gb_logDataFlag, pTargetOut, gi_batchSize, gi_mnistNumberOfClass, "target");
//	ZeroLogger::LogIfInfoHost(gb_logDataFlag, pActualOut, gi_batchSize, gi_mnistNumberOfClass, "actual");
//	ZeroLogger::LogIfInfoHost(true, pTargetOut, gi_batchSize, gi_mnistNumberOfClass, "target");
//	ZeroLogger::LogIfInfoHost(true, pActualOut, gi_batchSize, gi_mnistNumberOfClass, "actual");
	for (int i = 0; i < gi_batchSize; i++) {
		int maxIndex = 0;
		double maxValue = -1;
		for (int j = 0; j < gi_mnistNumberOfClass; j++) {
			if (pActualOut[i * gi_mnistNumberOfClass + j] > maxValue) {
				maxIndex = j;
				maxValue = pActualOut[i * gi_mnistNumberOfClass + j];
			}
		}
		if (pTargetOut[i * gi_mnistNumberOfClass + maxIndex] != 0) {
			(totalRightCounter)++;
		}
	}
	testedIterNumber += 1;
	if (testedIterNumber == gi_totalTestIterNumber) {
		gettimeofday(&mt_endTime, NULL);
		double costTime=mt_endTime.tv_sec - mt_startTime.tv_sec + (mt_endTime.tv_usec - mt_startTime.tv_usec) / 1000000.0;
		float accuracyRate=(totalRightCounter*1.0)/gi_totalTestImages;
		mv_accuracyVec.push_back(accuracyRate);
		LOG(INFO)<<"Test #"<<testedEpoch<<" End. totalRightCounter "<<totalRightCounter <<" TestRate "<<accuracyRate <<" CostTiime "<<costTime;
		testedIterNumber=0;testedEpoch++;totalRightCounter=0;
		gettimeofday(&mt_endTime, NULL);
		mq_netMsgQue.enqueue(new ZeroMessage(NET_END_TEST));
	} else {
		mq_netMsgQue.enqueue(new ZeroMessage(NET_CAN_FPROP));
	}
	delete[] pTargetOut;
	pTargetOut = NULL;
	delete[] pActualOut;
	pActualOut = NULL;
}
/*
 * ZeroThread.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: zys
 */
void* ZeroThread::run() {

	bool exit = false;
	while (!exit) {
		ZeroMessage* m = mq_msgQue.dequeue();
		MESSAGES type = m->getType();
		if (type == LAYER_CAN_FPROP) {
			FpropMessage* msg = (dynamic_cast<FpropMessage*>(m));
			assert(msg!=NULL);
			int layerId = msg->getToLayerId();
			int fragment = msg->getFragment();
			ZeroNet::getInstance().getLayerById(layerId)->Forward(fragment);
		} else if (type == LAYER_CAN_BPROP) {
			BpropMessage* msg = (dynamic_cast<BpropMessage*>(m));
			assert(msg!=NULL);
			int layerId = msg->getToLayerId();
			int fragment = msg->getFragment();
			ZeroNet::getInstance().getLayerById(layerId)->Backward(fragment);
		} else if (type == LAYER_CAN_SYNC) {
			ZeroNet::getInstance().getSync()->sync();
		} else if (type == LAYER_CAN_UPDATE) {
			int layerId = (dynamic_cast<UpdateMessage*>(m))->getToLayerId();
			ZeroNet::getInstance().getLayerById(layerId)->Update(ZeroNet::getInstance().getExecutedIterNumber());
		} else if (type == LAYER_CAN_EXIT) {
			exit = true;
		}
		delete m;
	}
	return NULL;
}
