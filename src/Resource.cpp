/*
 *Resource.cpp
 *
 *  Created on: 2016-6-1
 *      Author: zys
 */
#include "Resource.h"

bool gb_parallelMode = false;
//log debug information
bool gb_logFlag = false;
//log output/input data
bool gb_logDataFlag = false;
//log time information
bool gb_logTime = false;
#define REAL_DATA 1
#define TEST_REAL_DATA 1

#if REAL_DATA==1

const int gi_totalTrainImages = 50000;
const int gi_totalTestImages = 10000;
int gi_testInterval = 100;

const int gi_trainEpoch = 1;
const int gi_batchSize = 300;
const int gi_fragmentNumber = 3;

const int gi_mnistNumberOfClass = 10;
const int gi_datachannel_out=1;
const int gi_imageWidth = 28;
const int gi_imageHeight = 28;

const int gi_conv1_out = 20;
const int gi_conv2_out = 50;

const int gi_fc1_out = 500;
const int gi_fc2_out = gi_mnistNumberOfClass;

const int gi_conv1KernelWidth = 5;
const int gi_conv1KernelHeight = 5;
const int gi_pool1KernelWidth = 2;
const int gi_pool1KernelHeight = 2;


#elif TEST_REAL_DATA==1

const int gi_totalTrainImages =8;
const int gi_totalTestImages = 12;
int gi_testInterval = 100;

const int gi_trainEpoch = 1;
const int gi_batchSize = 4;
const int gi_fragmentNumber = 1;

const int gi_mnistNumberOfClass = 10;
const int gi_datachannel_out=1;
const int gi_imageWidth = 28;
const int gi_imageHeight = 28;

const int gi_conv1_out = 10;
const int gi_conv2_out = 10;

const int gi_fc1_out=10;
const int gi_fc2_out=gi_mnistNumberOfClass;

const int gi_conv1KernelWidth = 5;
const int gi_conv1KernelHeight = 5;
const int gi_pool1KernelWidth = 2;
const int gi_pool1KernelHeight = 2;

#else

const int gi_trainEpoch = 1;
const int gi_totalTrainImages =12;
const int gi_batchSize = 6;
const int gi_fragmentNumber = 3;

const int gi_totalTestImages = 4;
int gi_testInterval = 20;

const int gi_mnistNumberOfClass = 4;
const int gi_datachannel_out=1;
const int gi_imageWidth = 2;
const int gi_imageHeight = 4;
const int gi_fc1_out=4;
const int gi_fc2_out=4;
const int gi_fc3_out=4;
const int gi_fc4_out=gi_mnistNumberOfClass;

const int gi_conv1_out = 2;
const int gi_conv2_out = 5;

const int gi_conv1KernelWidth = 5;
const int gi_conv1KernelHeight = 5;
const int gi_pool1KernelWidth = 2;
const int gi_pool1KernelHeight = 2;

#endif

const int gi_imageSize = gi_imageHeight * gi_imageWidth;
const int gi_debugSize = 2;
const double gd_learnRate = 0.01;
const double gd_moment = 0.9;
const double gd_gamma = 0.0001;
const double gd_power = -0.75;
const double gd_decay = 0.0005;
const double gd_lr_mult1 = 1.0;
const double gd_lr_mult2 = 2.0;
const int gi_gpuNumbers = 2;

int gi_totalTrainBatchNumber = gi_totalTrainImages / gi_batchSize;
int gi_totalTestBatchNumber = gi_totalTestImages / gi_batchSize;
int gi_totalIterNumber=gi_totalTrainBatchNumber*gi_trainEpoch;
int gi_totalTestIterNumber=gi_totalTestBatchNumber;
int gi_testEpoch=gi_totalIterNumber/gi_testInterval;

const int gi_prefetchBatchNumber=50;
const int gi_streams=1;
const bool gb_multiGpus=true;
//0:train; 1:test
int gi_phase=0;
