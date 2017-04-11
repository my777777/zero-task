//============================================================================
// Name        : Zero.cpp
// Author      : onion
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "Resource.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

#include "ZeroNet.h"
#include "ZeroMessage.h"

int main(int argc, char* argv[]) {

	google::InitGoogleLogging(argv[0]);
	FLAGS_stderrthreshold = google::INFO;
	FLAGS_colorlogtostderr = true;
	ostringstream oss;
	oss<<"epoch "<<gi_trainEpoch<<" bSize "<<gi_batchSize<<" testEpoch "<<gi_testEpoch<<" fNum "<<gi_fragmentNumber<<" stream "<<gi_streams<<" gpus "<<gb_multiGpus;
	LOG(INFO)<<oss.str();
	LOG(ERROR) << "Begin---------";
	struct timeval st, et;
	double totalTimeUsed = 0;
	gettimeofday(&st, NULL);
	//get ZeroNet single instance.
	ZeroNet &model = ZeroNet::getInstance();
	//start the thread ,will call 'run()'.
	model.start();
	//the main thread will wait for 'ZeroNet' to end.
	model.join();

	google::ShutdownGoogleLogging();
	LOG(ERROR)<<"End-----------";
	gettimeofday(&et, NULL);
	totalTimeUsed=et.tv_sec - st.tv_sec + (et.tv_usec - st.tv_usec) / 1000000.0;
	LOG(INFO)<<oss.str();
	LOG(INFO)<<" time " <<totalTimeUsed<<" averTime(s) "<<totalTimeUsed/gi_trainEpoch<<" accRate "<<model.getAccuracyRate().back();
	LOG(INFO)<<" \n fTime(ms) "<<model.getTotalTime()[0]/gi_totalIterNumber
			 <<" \n bTime(ms) "<<model.getTotalTime()[1]/gi_totalIterNumber
			 <<" \n uTime(ms) "<<model.getTotalTime()[2]/gi_totalIterNumber
			 <<" \n tTime(ms) "<<model.getTotalTime()[3]/gi_totalIterNumber;
	return 0;
}
