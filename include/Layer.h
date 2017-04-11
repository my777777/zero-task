/*
 * Layer.h
 *
 *  Created on: Dec 8, 2016
 *      Author: zys
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "Resource.h"
#include "ZeroNet.h"
class ZeroNet;
class ZeroThread;
class LayerParam;

class Layer {

private:

	int mi_gpuIndex;
	int mi_layerId;
	string ms_layerName;
	LayerType me_layerType;

	/*
	 *the number of output maps on convLayer/poolLayer,
	 *the number of output neurons on other layers
	*/
	int mi_outMapsOrNeurons;
	int mi_inMapsOrNeurons = 0;

	int mi_mapWidth = 1;
	int mi_mapHeight = 1;
	int mi_kernelWidth=1;
	int mi_kernelHeight=1;

	/*
	 * the binded thread of current layer
	 * the thread will schedule the operations(forward/backward/update/synchronize)
	 * of the current layer depends on messages in its messages_queue(mq_msgQue)
	 */
	ZeroThread* mp_bindThread=NULL;

	/*
	 * if the current layer and the next layer is on different devices,
	 * then it needs to copy the input_diff during the backward,
	 * so (mb_isNeedCopyInDiff=true),else (mb_isNeedCopyInDiff=false)
	 */
	bool mb_isNeedCopyInDiff=false;

	/*
	 * if the current layer and the previous layer is on different devices,
	 * then it needs to copy the input_data during the forward,
	 * so (mb_isNeddCopyInData=true),else (mb_isNeddCopyInData=false)
	 */
	bool mb_isNeddCopyInData=false;


protected:

	cudaStream_t* mp_cudaStreams;
	const int mi_streams = gi_streams;
	cublasHandle_t m_cublasHandle;
	cudnnHandle_t m_cudnnHandle;

	/*
	 * A pointer which point to the previous layers.
	 */
	Layer* mp_prevLayer = NULL;
	Layer* mp_nextLayer = NULL;

	/*
	 * A pointer of layers' weight ,
	 * which will be allocated device memory via cudaMollc(...) in some layers(need it)
	 */
	double* mp_deviceWeight = NULL;

	/*
	 * A pointer of layers' bias ,
	 * which will be allocated device memory via cudaMollc(...) in some layers
	 */
	double* mp_deviceBias = NULL;

	/*
	 * A pointer which point to output_data,
	 * which is also the input_data of the next layer.
	 */
	double* mp_deviceOutData = NULL;

	/*
	 * A pointer which point to the weight gradient,
	 * which is used for updating weight of layers.
	 */
	double* mp_deviceWeightGradient = NULL;

	/*
	 * A pointer which point to the bias gradient,
	 * which is used for updating bias of layers.
	 */
	double* mp_deviceBiasGradient = NULL;

	/*
	 * A pointer which point to the history weight gradient,
	 * which is used for updating weight of layers.
	 */
	double* mp_historyWeightGradient = NULL;

	/*
	 * A pointer which point to the history bias gradient,
	 * which is used for updating bias of layers.
	 */
	double* mp_historyBiasGradient = NULL;


	/*
	 * A pointer which point to output_diff,
	 * which is also the input_diff of the previous layer.
	 */
	double* mp_deviceOutDiff = NULL;

	/*
	 * A pointer which point to one fragmentation of the current layer's input_data,
	 * if isNeedCopyInData=true,
	 * 		then we need to allocate device memory for mp_fragmentInData,
	 * 		and according to 'fragment number', we copy the corresponding fragmentation of the previous layer's output_data to mp_fragmentInData.
	 * 		Do not forget to release the memory at last!!
	 * else
	 * 		mp_fragmentInData just point to the corresponding fragmentation of the previous layer's output_data.
	 * 		eg.: mp_fragmentInData = prevLayer->getDeviceOutData() + inDataOffset;
	 */
	double* mp_fragmentInData = NULL;
	/*
	 * A pointer which point to one fragmentation of the current layer's out_data,
	 * mp_fragmentOutData = mp_deviceOutData + outDataOffset;
	 * outDataOffset is the offset of the current layer's output_data,which is calculated via the current fragment number.
	 */
	double* mp_fragmentOutData = NULL;
	/*
	 * A pointer which point to one fragmentation of the current layer's input_diff,
	 * if isNeedCopyInDiff=true,
	 * 		then we need to allocate device memory for mp_fragmentInDiff,
	 * 		and according to 'fragment number', we copy the corresponding fragmentation of the next layer's output_diff to mp_fragmentInDiff.
	 * 		Do not forget to release the memory at last!!
	 * else
	 * 		mp_fragmentInDiff just point to the corresponding fragmentation of the next layer's output_diff.
	 * 		eg.: mp_fragmentInDiff = prevLayer->getDeviceOutData() + inDataOffset;
	 */
	double* mp_fragmentInDiff = NULL;
	/*
	 * A pointer which point to one fragmentation of the current layer's out_diff,
	 * mp_fragmentOutDiff = mp_deviceOutDiff + outDiffOffset;
	 * outDiffOffset is the offset of the current layer's output_diff,which is calculated via the current fragment number.
	 */
	double* mp_fragmentOutDiff = NULL;


	/*
	 * To allocate memory
	 */
	virtual void ExeMemoryAlloc()=0;

	/*
	 * To release memory
	 */
	virtual void ExeMemoryRelease()=0;

	/*
	 * To execute 'forward operation',
	 * which is implemented by derived classes of 'Layer' to execute really 'forward operation'.
	 *
	 * fragment: the current fragment number
	 */
	virtual void ExeForward(int fragment)=0;

	/*
	 * To execute 'backward operation',
	 * which is implemented by derived classes of 'Layer' to execute really 'backward operation'.
	 *
	 * fragment: the current fragment number
	 */
	virtual void ExeBackward(int fragment)=0;

	/*
	 * To execute 'update operation',
	 * which is implemented by derived classes of 'Layer' to execute really 'update operation'.
	 *
	 * executedIterNumber: the executed 'iteration_number'
	 */
	virtual void ExeUpdate(long executedIterNumber);


private:
	/*
	 * To initialize member variables of 'Layer'.
	 * it is shadowed by the derived classes of 'Layer'
	 */
	void InitLayer(const LayerParam&);

	/*
	 * To calculate the width/the height of map for each layer.
	 */
	void CalcMapDims();

	/*
	 * To calculate the number of (input maps) or (input neurons) of each layer.
	 *
	 * input maps: convLayer/poolLayer
	 * input neurons: fullLayer,softmaxLayer and so on.
	 */
	void CalcInMapsOrNeurons();

	/*
	 * To set the pointers: mp_fragmentInData and mp_fragmentOutData.
	 *
	 * fragment: the current fragment number
	 */
	void FowardDataPreparation(int fragment);

	/*
	 * To set the pointers: mp_fragmentInDiff and mp_fragmentOutDiff.
	 *
	 * fragment: the current fragment number
	 */
	void BackwardDataPreparation(int fragment);

	/*
	 * To send messages to some layer.
	 */
	void FowardSendMsg(int fragment);

	/*
	 * To send messages to some layer.
	 */
	void BackwardSendMsg(int fragment);
	/*
	 * cuda stream sync
	 */
	void CudaStreamSync();

public:
	//17.03.01
	Layer(const LayerParam&);
	virtual ~Layer();

	/*
	 * Forward operation ,will call:
	 *  (1) FowardDataPreparation
	 *  (2) ExeForward
	 *  (3) FowardSendMsg
	 */
	void Forward(int);

	/*
	 * Backward operation ,will call:
	 *  (1) BackwardDataPreparation
	 *  (2) ExeBackward
	 *  (3) BackwardSendMsg
	 */
	void Backward(int);

	/*
	 * Update operation,will call
	 * (1) ExeUpdate
	 */
	void Update(long);

	/*
	 * To allocate memory
	 */
	void AllocMemory();

	/*
	 * To release memory
	 */
	void ReleaseMemory();


	void ConnectPrevLayer(Layer* prevLayer) {
		mp_prevLayer = prevLayer;
	}

	void ConnectNextLayer(Layer* nextLayer) {
		mp_nextLayer = nextLayer;
	}

	int getLayerId() const {
		return mi_layerId;
	}
	string getLayerName() const {
		return ms_layerName;
	}
	LayerType getLayerType() const {
		return me_layerType;
	}

	int getOutMapsOrNeurons() const  {
		return mi_outMapsOrNeurons;
	}
	int getInMapsOrNeurons() const {
		return mi_inMapsOrNeurons;
	}

	ZeroThread* getBindThread() const {
		return mp_bindThread;
	}

	int getMapWidth() const {
		return mi_mapWidth;
	}
	int getMapHeight() const {
		return mi_mapHeight;
	}

	int getMapArea() const {
		return getMapWidth() * getMapHeight();
	}
	int getKernelWidth() const {
		return mi_kernelWidth;
	}
	int getKernelHeight() const {
		return mi_kernelHeight;
	}

	int getKernelArea() const {
		return getKernelWidth() * getKernelHeight();
	}

	double* getDeviceOutData() {
		return mp_deviceOutData;
	}

	double* getDeviceOutDiff() {
		return mp_deviceOutDiff;
	}

	inline void CalcFullInNeurons(Layer* prevLayer) {
		LayerType prevType = prevLayer->getLayerType();
		if (prevType == LAYER_TYPE_FULL || prevType == LAYER_TYPE_RELU) {
			mi_inMapsOrNeurons = prevLayer->getOutMapsOrNeurons();
		} else if (prevType == LAYER_TYPE_CONV || prevType == LAYER_TYPE_POOL || prevType == LAYER_TYPE_INPUT) {
			mi_inMapsOrNeurons = (prevLayer->getMapArea()) * (prevLayer->getOutMapsOrNeurons());
		}
	}
	inline int getFragmentSize() const {
		assert((gi_batchSize % gi_fragmentNumber) == 0);
		return gi_batchSize / gi_fragmentNumber;
	}

	inline int getBatchSize() const {
		return gi_batchSize;
	}

	inline int getFragmentNumber() const {
		return gi_fragmentNumber;
	}

	Layer* getPrevLayer(){
		return mp_prevLayer;
	}

	Layer* getNextLayer() {
		return mp_nextLayer;
	}

	int getGpuIndex() const {
		return mi_gpuIndex;
	}

	void setBindThread(ZeroThread* bindThread){
		mp_bindThread=bindThread;
	}
	//if the next layer is on the different device ,need allocate memory to copy input_diff
	bool isNeedCopyInDiff(){
		return mb_isNeedCopyInDiff;
	}
	//if the previous layer is on the different device ,need allocate memory to copy input_data
	bool isNeedCopyInData(){
		return mb_isNeddCopyInData;
	}
};

class LayerParam {

private:
	int mi_layerId;
	string ms_layerName;
	LayerType me_layerType;
	vector<int> mv_nextIds;
	vector<int> mv_prevIds;
	int mi_prevId;
	int mi_nextId;
	int mi_outMapsOrNeurons;
//	int mi_inMapsOrNeurons;
	int mi_kernelWidth;
	int mi_kernelHeight;
	int mi_gpuIndex;

public:

	LayerParam(int layerId, string layerName, LayerType layerType, int outNumber, int gpuIndex = 0) {
		mi_layerId = layerId;
		ms_layerName = layerName;
		me_layerType = layerType;

		mi_nextId = mi_layerId + 1;
		mi_prevId = mi_layerId - 1;

		mi_outMapsOrNeurons = outNumber;
		if(layerType==LAYER_TYPE_CONV){
			mi_kernelWidth = 5;
			mi_kernelHeight = 5;
		}else if(layerType==LAYER_TYPE_POOL){
			mi_kernelWidth = 2;
			mi_kernelHeight = 2;
		}

		mi_gpuIndex = gpuIndex;

	}
	virtual ~LayerParam() {
	}

	int getLayerId() const {
		return mi_layerId;
	}
	string getLayerName() const {
		return ms_layerName;
	}
	LayerType getLayerType() const {
		return me_layerType;
	}
	int getPrevId() const {
		return mi_prevId;
	}
	int getNextId() const {
		return mi_nextId;
	}
	int getOutMapsOrNeurons() const {
		return mi_outMapsOrNeurons;
	}
//	int getInMapsOrNeurons() const {
//		return mi_inMapsOrNeurons;
//	}
	int getGpuIndex() const {
		return mi_gpuIndex;
	}
	int getKWidth() const {
		return mi_kernelWidth;
	}
	int getKHeight() const {
		return mi_kernelHeight;
	}

	string getType() const{
		LayerType type=me_layerType;
		if(type==LAYER_TYPE_INPUT){
			return "Data";
		}else if(type==LAYER_TYPE_CONV){
			return "Conv";
		}else if(type==LAYER_TYPE_POOL){
			return "Pool";
		}else if(type==LAYER_TYPE_FULL){
			return "FullConn";
		}else if(type==LAYER_TYPE_RELU){
			return "ReLU";
		}else if(type==LAYER_TYPE_SOFTMAX){
			return "Softmax";
		}
		return "NullType";
	}
};

#endif /* LAYER_H_ */
