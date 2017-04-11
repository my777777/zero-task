/*
 * DataLayer.h
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */

#ifndef DATALAYER_H_
#define DATALAYER_H_

#include "Resource.h"
#include "Layer.h"
#include "MnistIterator.h"

class DataPrefetcher;

class DataBatch {

public:
	double* mp_deviceData = NULL;
	double* mp_deviceLabel = NULL;

public:
	DataBatch(double* data, double* label) {
		int dataSize = gi_imageSize * gi_batchSize;
		checkCudaErrors(cudaMalloc(&mp_deviceData, dataSize * sizeof(double)));
		checkCudaErrors(cudaMemcpy((void* )mp_deviceData, (const void* )data, dataSize * sizeof(double), cudaMemcpyHostToDevice));

		int labelSize = gi_mnistNumberOfClass * gi_batchSize;
		checkCudaErrors(cudaMalloc(&mp_deviceLabel, dataSize * sizeof(double)));
		checkCudaErrors(cudaMemcpy((void* )mp_deviceLabel, (const void* )label, labelSize * sizeof(double), cudaMemcpyHostToDevice));
	}

	virtual ~DataBatch() {
		if (mp_deviceLabel != NULL) {
			checkCudaErrors(cudaFree(mp_deviceLabel));
			mp_deviceLabel = NULL;
		}
		if (mp_deviceData != NULL) {
			checkCudaErrors(cudaFree(mp_deviceData));
			mp_deviceData = NULL;
		}
	}
};

class DataLayer: public Layer {

private:
	double* mp_hostInData = NULL;
	double* mp_hostInLabel = NULL;
//	double* mp_deviceOutLabel = NULL;
	DataPrefetcher* mp_trainPrefetcher;
	DataPrefetcher* mp_testPrefetcher;

public:
	DataLayer(const LayerParam& layerParam);
	virtual ~DataLayer();

	void PrefetchNextBatch(int);
//	double* getDeviceOutLabel() {
//		return mp_deviceOutLabel;
//	}

protected:
	void ExeForward(int);
	void ExeBackward(int);
	void ExeMemoryAlloc();
	void ExeMemoryRelease();

};

class DataPrefetcher: public Thread {

protected:
	Queue<DataBatch*> mp_dataBatchQue;
	boost::condition_variable batch_cond;
	boost::mutex batch_mutex;
public:
	virtual DataBatch* getNextBatch() {
		return NULL;
	}
	DataPrefetcher() :
			Thread(true) {
	}
	virtual ~DataPrefetcher() {
	}
private:
	virtual void* run() {
		return NULL;
	}

	virtual void produce(DataBatch* dataBatch) {
	}
};

class TrainPrefetcher: public DataPrefetcher {

public:
	TrainPrefetcher() :
			DataPrefetcher() {
	}
	virtual ~TrainPrefetcher() {
	}

	DataBatch* getNextBatch() {
		boost::unique_lock<boost::mutex> lk(batch_mutex);
		batch_cond.wait(lk, [=] {return mp_dataBatchQue.getNumElements()!=0;});
		static int nums = 0;
		DataBatch* dataBatch = mp_dataBatchQue.dequeue();
		nums = nums + 1;
		LOG_IF(ERROR,gb_logFlag) << __func__ << ";total:" << gi_totalIterNumber << "----------------Got:" << nums<<";TrainPrefetcher";
		int n = mp_dataBatchQue.getNumElements();
		lk.unlock();
		if (n <= 1)
			batch_cond.notify_one();
		return dataBatch;
	}


private:
	void* run() {
		DataBatch* dataBatch;
		MnistIterator* iterator = new MnistIterator();
		double* data = new double[gi_imageSize * gi_batchSize];
		double* label = new double[gi_mnistNumberOfClass * gi_batchSize];
		long iterNumber = 0;
		while (iterNumber < gi_totalIterNumber) {
			iterator->nextTrainBatch(iterNumber % gi_totalTrainBatchNumber, data);
			iterator->nextTrainLabelBatch(iterNumber % gi_totalTrainBatchNumber, label);
			dataBatch = new DataBatch(data, label);
			produce(dataBatch);
			iterNumber++;
		}
		delete data;
		delete label;
		delete (iterator);
		return NULL;
	}

	void produce(DataBatch* dataBatch) {
		boost::unique_lock<boost::mutex> lk(batch_mutex);
		batch_cond.wait(lk, [&] {return (mp_dataBatchQue.getNumElements()<gi_prefetchBatchNumber);});
		static int nums = 0;
		static int total = gi_totalIterNumber;
		mp_dataBatchQue.enqueue(dataBatch);
		nums = nums + 1;
		LOG_IF(ERROR,gb_logFlag) << __func__ << ";total:" << total << "************************Added:" << nums <<";TrainPrefetcher";
		lk.unlock();
		batch_cond.notify_one();
	}
};

class TestPrefetcher: public DataPrefetcher {
public:

	TestPrefetcher() :
			DataPrefetcher() {
	}
	virtual ~TestPrefetcher() {
	}

	DataBatch* getNextBatch() {
		boost::unique_lock<boost::mutex> lk(batch_mutex);
		batch_cond.wait(lk, [=] {return mp_dataBatchQue.getNumElements()!=0;});
		static int nums = 0;
		DataBatch* dataBatch = mp_dataBatchQue.dequeue();
		nums = nums + 1;
		LOG_IF(ERROR,gb_logFlag) << __func__ << ";total:" << gi_totalTestIterNumber*gi_testEpoch<< "----------------Got:" << nums<<";TestPrefetcher";
		int n = mp_dataBatchQue.getNumElements();
		lk.unlock();
		if (n <= 1)
			batch_cond.notify_one();
		return dataBatch;
	}

private:
	void* run() {

		DataBatch* dataBatch;
		MnistIterator* iterator = new MnistIterator();
		double* data = new double[gi_imageSize * gi_batchSize];
		double* label = new double[gi_mnistNumberOfClass * gi_batchSize];
		long iterNumber = 0;
		while (iterNumber <gi_totalTestIterNumber*gi_testEpoch) {
			iterator->nextTestBatch(iterNumber % gi_totalTestBatchNumber, data);
			iterator->nextTestLabelBatch(iterNumber % gi_totalTestBatchNumber, label);
			dataBatch = new DataBatch(data, label);
			produce(dataBatch);
			iterNumber++;
		}
		delete data;
		delete label;
		delete (iterator);
		return NULL;
	}

	void produce(DataBatch* dataBatch) {
		boost::unique_lock<boost::mutex> lk(batch_mutex);
		batch_cond.wait(lk, [&] {return (mp_dataBatchQue.getNumElements()<gi_prefetchBatchNumber);});
		static int nums = 0;
		mp_dataBatchQue.enqueue(dataBatch);
		nums = nums + 1;
		LOG_IF(ERROR,gb_logFlag) << __func__ << ";total:" << gi_totalTestIterNumber << "************************Added:" << nums <<";TestPrefetcher";
		lk.unlock();
		batch_cond.notify_one();
	}

};

#endif /* DATALAYER_H_ */
