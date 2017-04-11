#ifndef _COMMON_H_
#define _COMMON_H_

#include <math.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <malloc.h>
#include <stdint.h>
#include <alloca.h>
#include <cstring>
#include <string.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include "error_util.h"
#include <glog/logging.h>
#include <sys/time.h>
#include <map>
#include <vector>
#include <boost/thread.hpp>
#include <queue>
using namespace std;


typedef enum {
	LAYER_TYPE_INPUT, LAYER_TYPE_CONV, LAYER_TYPE_POOL, LAYER_TYPE_FULL, LAYER_TYPE_RELU, LAYER_TYPE_SOFTMAX
} LayerType;

#define SIGMOID(x) (1.7159*tanh(0.66666667*x))
#define DSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))  // derivative of the sigmoid as a function of the sigmoid's output
#define FALSE 0

typedef unsigned int UINT;
typedef bool BOOL;

const string gs_dataDir = "data/";
const string gs_testLabelPath = gs_dataDir + "t10k-labels.idx1-ubyte";
const string gs_testImagePath = gs_dataDir + "t10k-images.idx3-ubyte";
const string gs_trainImagePath = gs_dataDir + "train-images.idx3-ubyte";
const string gs_trainLabelPath = gs_dataDir + "train-labels.idx1-ubyte";

//const string gs_conv1_bin = "data/conv1.bin";
//const string gs_conv1_bias_bin = "data/conv1.bias.bin";
//const string gs_conv2_bin = "data/conv2.bin";
//const string gs_conv2_bias_bin = "data/conv2.bias.bin";
//const string gs_ip1_bin = "data/ip1.bin";
//const string gs_ip1_bias_bin = "data/ip1.bias.bin";
//const string gs_ip2_bin = "data/ip2.bin";
//const string gs_ip2_bias_bin = "data/ip2.bias.bin";

extern const int gi_conv1_out;
extern const int gi_conv2_out;

extern const int gi_fc1_out;
extern const int gi_fc2_out;
extern const int gi_fc3_out;
extern const int gi_fc4_out;


extern const int gi_batchSize;
extern const int gi_fragmentNumber;
extern const int gi_debugSize;
extern const int gi_imageWidth;
extern const int gi_imageHeight;
extern const int gi_datachannel_out;
extern const int gi_mnistNumberOfClass;
extern const int gi_conv1KernelWidth;
extern const int gi_conv1KernelHeight;
extern const int gi_pool1KernelWidth;
extern const int gi_pool1KernelHeight;
//*/
extern const int gi_totalTestImages;
extern const int gi_totalTrainImages;

extern const double gd_learnRate;
extern const double gd_moment;
extern const double gd_gamma;
extern const double gd_power;
extern const double gd_decay;
extern const double gd_lr_mult1;
extern const double gd_lr_mult2;

extern const int gi_imageSize;
extern const int gi_gpuNumbers;
extern const int gi_trainEpoch;

extern bool gb_logFlag;
extern bool gb_logDataFlag;
extern int gi_testInterval;
extern int gi_totalTrainBatchNumber;
extern int gi_totalTestBatchNumber;
extern int gi_totalIterNumber;
extern int gi_totalTestIterNumber;
extern int gi_testEpoch;
extern bool gb_parallelMode;
extern bool gb_realData;

extern const int gi_prefetchBatchNumber;
extern const int gi_streams;
extern const bool gb_multiGpus;
extern int gi_phase;

extern bool gb_logTime;
#endif
