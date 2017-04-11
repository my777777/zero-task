/*
 * ZeroNet.h
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */
#ifndef ZERONET_H_
#define ZERONET_H_

#include "Resource.h"
#include "thread.h"
#include "queue.h"
#include "sync.h"

class Layer;
class ZeroMessage;
class LayerParam;
class ZeroThread;
class DataLayer;
class FullConnLayer;
class NetParam;
class DataLayer;
class SoftmaxLayer;

class ZeroNet: public Thread {

private:

	/*
	 * ZeroNet param.
	 */
	NetParam* mp_netParam;

	/*
	 * A vector which store LayerParam*
	 */
	vector<LayerParam*> mv_layerParamVec;

	/*
	 * A vector which store each layer,
	 *
	 * layer is created by LayerFactory::CreateLayer(LayerParam*);
	 */
	vector<Layer*> mv_layerVec;

	/*
	 * A vector which store ZeroThread* .
	 *
	 * ZeroThread is designed to scheduled the operations of different layers.
	 */
	vector<ZeroThread*> mv_zeroThreadVec;

	/*
	 * A pointer which point to ThreadSynchronizer, to synchronize threads.
	 */
	ThreadSynchronizer* mp_sync;

	/*
	 * A queue which store messages from different objects.
	 *
	 * ZeroNet will executed the corresponding operation according to the type of messages.
	 */
	Queue<ZeroMessage*> mq_netMsgQue;

	/*
	 * the executed 'iteration_number'
	 */
	int mi_executedIterNumber;

	//17.02.23
	DataLayer* mp_dataLayer;
	SoftmaxLayer* mp_softmaxLayer;

	/*
	 * A vector which store accuracy rate calculated by 'NetCalTestAccuracy()'
	 */
	vector<float> mv_accuracyVec;
	struct timeval mt_startTime, mt_endTime;
	struct timeval mt_iterStartTime,mt_iterEndTime;
	struct timeval mt_beginForward,mt_beginBackward,mt_beginUpdate;
	/*
	 * mv_totalTime[0]:total forward time
	 * mv_totalTime[1]:total backward time
	 * mv_totalTime[2]:total update time
	 * mv_totalTime[3]:total time
	 */
	vector<double> mv_totalTime={0,0,0,0};


private:

	/*
	 * ZeroNet is a singleton class,
	 * So we need to set the following functions to private.
	 */
	ZeroNet();
	ZeroNet(const ZeroNet &);
	ZeroNet& operator =(const ZeroNet &);

	/*
	 * Implement the virtual function 'run()' of Thread.
	 */
	void* run();

	/*
	 * forward
	 */
	void NetForward();

	/*
	 *  backward
	 */
	void NetBackward();

	/*
	 * parallel update
	 */
	void NetParallelUpdate();

	/*
	 * synchronize
	 */
	void NetSyncWithChildThreads();

	/*
	 * create Layers
	 */
	void CreateLayers();

	/*
	 * connect adjacent layers.
	 */
	void ConnectAdjacentLayers();

	/*
	 * allocate memory for layers
	 */
	void AllocLayerMemory();

	/*
	 * begin to test net
	 */
	void NetBeginTest();
	/*
	 * end test net
	 */
	void NetEndTest();
	/*
	 * calculate the accuracy rate of net
	 */
	void NetCalTestAccuracy();

public:

	virtual ~ZeroNet();

	static ZeroNet& getInstance() {
		static ZeroNet instance;
		return instance;
	}

public:


	vector<ZeroThread*>& getZeroThreadVec() {
		return mv_zeroThreadVec;
	}
	ThreadSynchronizer* getSync() {
		return mp_sync;
	}

	Queue<ZeroMessage*>& getNetMsgQue() {
		return mq_netMsgQue;
	}
	vector<Layer*>& getLayerVec() {
		return mv_layerVec;
	}

	NetParam* getNetParam() {
		return mp_netParam;
	}
	int getExecutedIterNumber() {
		return mi_executedIterNumber;
	}

	inline Layer* getLayerById(int index) {

		static int size = mv_layerVec.size();
		if (index <= -1 || index >= size) {
			return NULL;
		}
		return mv_layerVec[index];
	}

	DataLayer* getDataLayer() {
		return mp_dataLayer;
	}

	vector<float>& getAccuracyRate(){
		return mv_accuracyVec;
	}

	vector<double>& getTotalTime(){
		return mv_totalTime;
	}
};


class ZeroThread: public Thread {
	Queue<ZeroMessage*> mq_msgQue;
public:
	ZeroThread():Thread(true){};
	virtual ~ZeroThread(){};
	Queue<ZeroMessage*>& getMsgQueue() {
		return mq_msgQue;
	}
private:
	void* run();
};

class NetParam {

private:
	int mi_batchSize;
	int mi_totalIterNumber;
	int mi_fragmentNumber;
	int mi_totalLayerNumber;

public:
	NetParam(int batchSize, int fragmentNum, int totalIterNum, int totalLayerNum) {
		assert((batchSize % fragmentNum) == 0);
		mi_batchSize = batchSize;
		mi_totalIterNumber = totalIterNum;
		mi_fragmentNumber = fragmentNum;
		mi_totalLayerNumber = totalLayerNum;
	}

	NetParam(string filePath) {
		mi_batchSize = 300;
		mi_totalIterNumber = 300;
		mi_fragmentNumber = 3;
		mi_totalLayerNumber = 0;
	}

	int getBatchSize() {
		return mi_batchSize;
	}

	int getTotalIterNumber() {
		return mi_totalIterNumber;
	}

	int getFragmentNumber() {
		return mi_fragmentNumber;
	}

	int getTotalLayerNumber() {
		return mi_totalLayerNumber;
	}

};
#endif
