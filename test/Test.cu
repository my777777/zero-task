/*
 * Test.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: zys
 */
#include "Resource.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
using namespace std;
#include <omp.h>

#include "ZeroTest.cuh"

__global__ void myKernel(int *d_o)
{
    long delay=100000000;
    while(delay>0){
    	delay--;
    }

    for(int i=0;i<1024*1024;++i){
    	d_o[i]=i;
    }
}

void testDgemmWithVec(){

	ostringstream oss;
	int fragmentNumber=3;
	double alpha = 1.0;
	double beta = 0.0;
	cublasHandle_t handles[fragmentNumber];
	cudaStream_t streams[fragmentNumber];
	for(int i=0;i<fragmentNumber;++i){
		checkCublasErrors(cublasCreate(&handles[i]));
		cudaStreamCreate(&streams[i]);
	}


	int M = 150;  //batch_size
	int N = 256*36; //in
	int Q = 4096; //out

	long size_A = (M * N); //input_data 3*2
	long size_B = (N * Q); //weight 2*1
	long size_C = (M * Q); //output_data 3*1

//	double *A,*B,*C;
//	A=(double*)malloc(size_A*sizeof(double));
//	oss.str("");
//	for(int i=0;i<size_A;++i){
//		A[i]=rand()%3;
//		oss<<A[i]<<" ";
//	}
//	LOG(INFO)<<"A:"<<oss.str();
//	sleep(1);
//	B=(double*)malloc(size_B*sizeof(double));
//	oss.str("");
//	for(int i=0;i<size_B;i++){
//		B[i]=rand()%2;
//		oss<<B[i]<<" ";
//	}
//	LOG(INFO)<<"B:"<<oss.str();
//
//	C=(double*)malloc(size_C*sizeof(double));
//	double *d_A,*d_B,*d_C;
//	checkCudaErrors(cudaMalloc(&d_A, size_A * sizeof(double)));
//	checkCudaErrors(cudaMalloc(&d_B, size_B * sizeof(double)));
//	checkCudaErrors(cudaMemcpy((void*)d_A,(void*)A,size_A*sizeof(double),cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy((void*)d_B,(void*)B,size_B*sizeof(double),cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMalloc(&d_C, size_C * sizeof(double)));
//	checkCublasErrors(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
//
//	checkCudaErrors(cudaMemcpy((void*)C,(void*)d_C,size_C*sizeof(double),cudaMemcpyDeviceToHost));
//	oss.str("");
//	for(int i=0;i<size_C;++i){
//		oss<<C[i]<<" ";
//	}
//	LOG(INFO)<<oss.str();



	vector<double*> d_InDataVec;
	vector<double*> d_OutDataVec;

	double* d_Weight=NULL;
	checkCudaErrors(cudaMalloc((void**)&d_Weight,size_B* sizeof(double)));
	double* weight=NULL;
	weight=(double*)malloc(size_B*sizeof(double));
//	srand(time(NULL));
//	oss.str("");
//	for(int i=0;i<size_B;++i){
//		weight[i]=rand()%3;
//		oss<<weight[i]<<" ";
//	}
//	LOG(INFO)<<"Weight:"<<oss.str();
	checkCudaErrors(cudaMemcpy((void*)d_Weight,(void*)weight,size_B*sizeof(double),cudaMemcpyHostToDevice));

	double* in=NULL;
	in=(double*)malloc(size_A/fragmentNumber*sizeof(double));
	for(int i=0;i<fragmentNumber;++i){
		double* d_in=NULL;
		double* d_out=NULL;
//		sleep(1);
//		srand(time(NULL));
//		oss.str("");
//		for(int j=0;j<size_A/fragmentNumber;++j){
//			in[j]=rand()%3;
//			oss<<in[j]<<" ";
//		}
//		LOG(INFO)<<"InData:"<<oss.str();

		checkCudaErrors(cudaMalloc((void**)&d_in, size_A/fragmentNumber * sizeof(double)));
		checkCudaErrors(cudaMemcpy((void*)d_in,(void*)in,size_A/fragmentNumber*sizeof(double),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_out, size_C/fragmentNumber * sizeof(double)));
		d_InDataVec.push_back(d_in);
		d_OutDataVec.push_back(d_out);
	}

	for(int i=0;i<fragmentNumber;++i){
		cublasSetStream(handles[i],streams[i]);
		checkCublasErrors(cublasDgemm(handles[i], CUBLAS_OP_N, CUBLAS_OP_N, Q, M/fragmentNumber, N, &alpha, d_Weight, Q, d_InDataVec[i], N, &beta, d_OutDataVec[i], Q));

	}

//	double* out=NULL;
//	out=(double*)malloc(size_C/fragmentNumber*sizeof(double));
//	checkCudaErrors(cudaMemcpy((void*)out,(void*)d_OutDataVec[0],size_C/fragmentNumber*sizeof(double),cudaMemcpyDeviceToHost));
//	oss.str("");
//	for(int i=0;i<size_C/fragmentNumber;++i){
//		oss<<out[i]<<" ";
//	}
//	LOG(INFO)<<"outData:"<<oss.str();
//
//	checkCudaErrors(cudaMemcpy((void*)out,(void*)d_OutDataVec[1],size_C/fragmentNumber*sizeof(double),cudaMemcpyDeviceToHost));
//	oss.str("");
//	for(int i=0;i<size_C/fragmentNumber;++i){
//		oss<<out[i]<<" ";
//	}
//	LOG(INFO)<<"outData:"<<oss.str();
//	free(out);
//

	free(in);
	free(weight);

	cudaFree(d_Weight);
	for(int i=0;i<fragmentNumber;++i){
		cudaFree(d_InDataVec[i]);
		cudaFree(d_OutDataVec[i]);
		cudaStreamDestroy(streams[i]);
		checkCublasErrors(cublasDestroy(handles[i]));

	}

	cudaDeviceReset();

//	free(A);
//	free(B);
//	free(C);
//	cudaFree(d_B);
//	cudaFree(d_A);
//	cudaFree(d_C);
}

void testDgemm(){

	ostringstream oss;
	int fragmentNumber=3;
	double alpha = 1.0;
	double beta = 0.0;
	cublasHandle_t handles[fragmentNumber];
	cudaStream_t streams[fragmentNumber];
	for(int i=0;i<fragmentNumber;++i){
		checkCublasErrors(cublasCreate(&handles[i]));
		cudaStreamCreate(&streams[i]);
	}


	int M = 150;  //batch_size
	int N = 256*36; //in
	int Q = 4096; //out

	long size_A = (M * N); //input_data 3*2
	long size_B = (N * Q); //weight 2*1
	long size_C = (M * Q); //output_data 3*1

//	double *A,*B,*C;
//	A=(double*)malloc(size_A*sizeof(double));
//	oss.str("");
//	for(int i=0;i<size_A;++i){
//		A[i]=rand()%3;
//		oss<<A[i]<<" ";
//	}
//	LOG(INFO)<<"A:"<<oss.str();
//	sleep(1);
//	B=(double*)malloc(size_B*sizeof(double));
//	oss.str("");
//	for(int i=0;i<size_B;i++){
//		B[i]=rand()%2;
//		oss<<B[i]<<" ";
//	}
//	LOG(INFO)<<"B:"<<oss.str();
//
//	C=(double*)malloc(size_C*sizeof(double));
//	double *d_A,*d_B,*d_C;
//	checkCudaErrors(cudaMalloc(&d_A, size_A * sizeof(double)));
//	checkCudaErrors(cudaMalloc(&d_B, size_B * sizeof(double)));
//	checkCudaErrors(cudaMemcpy((void*)d_A,(void*)A,size_A*sizeof(double),cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy((void*)d_B,(void*)B,size_B*sizeof(double),cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMalloc(&d_C, size_C * sizeof(double)));
//	checkCublasErrors(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
//
//	checkCudaErrors(cudaMemcpy((void*)C,(void*)d_C,size_C*sizeof(double),cudaMemcpyDeviceToHost));
//	oss.str("");
//	for(int i=0;i<size_C;++i){
//		oss<<C[i]<<" ";
//	}
//	LOG(INFO)<<oss.str();



	double* d_InData;
	double* d_OutData;
	double* inData=NULL;
	inData=(double*)malloc(size_A*sizeof(double));
	srand(time(NULL));
	oss.str("");
	for(int i=0;i<size_A;++i){
		inData[i]=rand()%3;
//		oss<<inData[i]<<" ";
//
//		if(((i+1)%N)==0){
//			oss<<"\n";
//		}
	}
	//LOG(INFO)<<"InData:"<<oss.str();
	checkCudaErrors(cudaMalloc((void**)&d_InData, size_A* sizeof(double)));
	checkCudaErrors(cudaMemcpy((void*)d_InData,(void*)inData,size_A*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_OutData, size_C * sizeof(double)));
	double* d_Weight=NULL;
	checkCudaErrors(cudaMalloc((void**)&d_Weight,size_B* sizeof(double)));
	double* weight=NULL;
	weight=(double*)malloc(size_B*sizeof(double));
	srand(time(NULL));
	oss.str("");
	for(int i=0;i<size_B;++i){
		weight[i]=rand()%3;
		oss<<weight[i]<<" ";
	}
	//LOG(INFO)<<"Weight:"<<oss.str();
	checkCudaErrors(cudaMemcpy((void*)d_Weight,(void*)weight,size_B*sizeof(double),cudaMemcpyHostToDevice));

	double* out=NULL;
	out=(double*)malloc(size_C/fragmentNumber*sizeof(double));
	for(int i=0;i<fragmentNumber;++i){
		cublasSetStream(handles[i],streams[i]);
	}
	for(int i=0;i<fragmentNumber;++i){
		checkCublasErrors(cublasDgemm(handles[i], CUBLAS_OP_N, CUBLAS_OP_N, Q, M/fragmentNumber, N, &alpha, d_Weight, Q, (d_InData+i*size_A/fragmentNumber), N, &beta, (d_OutData+i*size_C/fragmentNumber), Q));
//		checkCudaErrors(cudaMemcpy((void*)out,(void*)(d_OutData+i*size_C/fragmentNumber),size_C/fragmentNumber*sizeof(double),cudaMemcpyDeviceToHost));
//		oss.str("");
//		for(int j=0;j<size_C/fragmentNumber;++j){
//			oss<<out[j]<<" ";
//		}
//		LOG(INFO)<<"outData:"<<oss.str();
	}

//
//	checkCudaErrors(cudaMemcpy((void*)out,(void*)(d_OutData+0*size_C/fragmentNumber),size_C/fragmentNumber*sizeof(double),cudaMemcpyDeviceToHost));
//	oss.str("");
//	for(int i=0;i<size_C/fragmentNumber;++i){
//		oss<<out[i]<<" ";
//	}
//	LOG(INFO)<<"outData:"<<oss.str();
//
//	checkCudaErrors(cudaMemcpy((void*)out,(void*)(d_OutData+1*size_C/fragmentNumber),size_C/fragmentNumber*sizeof(double),cudaMemcpyDeviceToHost));
//	oss.str("");
//	for(int i=0;i<size_C/fragmentNumber;++i){
//		oss<<out[i]<<" ";
//	}
//	LOG(INFO)<<"outData:"<<oss.str();
	free(out);
	free(weight);
	cudaFree(d_InData);
	cudaFree(d_OutData);
	cudaFree(d_Weight);
	for(int i=0;i<fragmentNumber;++i){
		cudaStreamDestroy(streams[i]);
		checkCublasErrors(cublasDestroy(handles[i]));
	}
	cudaDeviceReset();

//	free(A);
//	free(B);
//	free(C);
//	cudaFree(d_B);
//	cudaFree(d_A);
//	cudaFree(d_C);
}
void testKernelWithStreams() {

	int nkernels = 32;
	int nStreams = nkernels;
    int nbytes=nkernels*1024*1024*sizeof(int)*10;
    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));

	cudaStream_t streams[nStreams];
	for (int i = 0; i < nStreams; ++i) {
		cudaStreamCreate(&streams[i]);
	}

	double startTime = omp_get_wtime();
	for (int i=0; i<nkernels; ++i){
		dim3 gridSize(64,1);
		dim3 blockSize(2,1);
		myKernel<<<gridSize,blockSize,0,streams[i]>>>(&d_a[i]);
	}
	cudaDeviceSynchronize();
	double endTime = omp_get_wtime();
	cout << "Runtime:" << endTime - startTime << endl;
    cudaFree(d_a);
	for (int i = 0; i < nStreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}
}

void testGemmsOnSameDevice(){
    int nkernels = 2;               // number of concurrent kernels
    int nstreams = nkernels;    // use one more stream than concurrent kernel
    int cuda_device = 0;

    float elapsed_time;   // timing variables
 //   printf("[%s] - Starting...\n", argv[0]);

    //fc6
//    int M = 40;  //batch_size
//    int N = 256*6*6; //in
//    int Q = 4096; //out

    //fc7
    int M = 40;  //batch_size
	int N = 4096; //in
	int Q = 4096; //out


//  int M = 128;  //batch_size
//	int N = 128; //in
//	int Q = 128; //out


	long size_A = (M * N); //input_data
	long size_B = (N * Q); //weight
	long size_C = (M * Q); //output_data

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    if ((deviceProp.concurrentKernels == 0))
    {
        printf("> GPU does not support concurrent kernel execution\n");
        printf("  CUDA kernel runs will be serialized\n");
    }

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
	cublasHandle_t *cublasHandle=(cublasHandle_t *)malloc(nstreams * sizeof(cublasHandle_t));

    for (int i = 0; i < nstreams; i++)
    {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
        checkCublasErrors(cublasCreate(&cublasHandle[i]));
    }

	double *d_A,*d_B,*d_C;
	checkCudaErrors(cudaMalloc(&d_A, size_A * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_B, size_B * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_C, size_C * sizeof(double)));
	double *d_A1,*d_B1,*d_C1;
	checkCudaErrors(cudaMalloc(&d_A1, size_A * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_B1, size_B * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_C1, size_C * sizeof(double)));

	double alpha = 1.0;
	double beta = 0.0;

	double alpha1 = 1.0;
	double beta1 = 0.0;
    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));
    checkCudaErrors(cudaEventRecord(start_event, 0));

    checkCublasErrors(cublasSetStream(cublasHandle[0],streams[0]));
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));


    checkCublasErrors(cublasSetStream(cublasHandle[1],streams[1]));
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    cout<<"Time:"<<elapsed_time<<endl;

    cudaDeviceSynchronize();

    // release resources
    for (int i = 0; i < nkernels; i++)
    {
        cudaStreamDestroy(streams[i]);
        checkCublasErrors(cublasDestroy(cublasHandle[i]));
    }

    free(streams);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);

//    cudaFree(d_X);
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("Test passed\n");
}

void testGemmsOnDiffDevice(){
    int nkernels = 2;               // number of concurrent kernels
    int nstreams = nkernels;    // use one more stream than concurrent kernel
    int cuda_device = 0;
    int nbytes=nkernels*1024*1024*sizeof(int)*10;


    //fc6
    int M = 100;  //batch_size
    int N = 256*6*6; //in
    int Q = 4096; //out

    //fc7
//    int M = 100;  //batch_size
//	int N = 4096; //in
//	int Q = 4096; //out

	long size_A = (M * N); //input_data
	long size_B = (N * Q); //weight
	long size_C = (M * Q); //output_data

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    if ((deviceProp.concurrentKernels == 0))
    {
        printf("> GPU does not support concurrent kernel execution\n");
        printf("  CUDA kernel runs will be serialized\n");
    }

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // allocate and initialize an array of stream handles
//  cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
	cublasHandle_t *cublasHandle=(cublasHandle_t *)malloc(nstreams * sizeof(cublasHandle_t));

//    for (int i = 0; i < nstreams; i++)
//    {
//        checkCudaErrors(cudaStreamCreate(&(streams[i])));
//        checkCublasErrors(cublasCreate(&cublasHandle[i]));
//    }

	double *d_A,*d_B,*d_C;
	cudaSetDevice(0);
	checkCudaErrors(cudaMalloc(&d_A, size_A * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_B, size_B * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_C, size_C * sizeof(double)));
    checkCublasErrors(cublasCreate(&cublasHandle[0]));
	double *d_A1,*d_B1,*d_C1;
	cudaSetDevice(1);
	checkCudaErrors(cudaMalloc(&d_A1, size_A * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_B1, size_B * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_C1, size_C * sizeof(double)));
    checkCublasErrors(cublasCreate(&cublasHandle[1]));

    // allocate device memory
    int *d_a = 0;             // pointers to data and init value in the device memory
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    int *a=(int*)malloc(nbytes);


	double alpha = 1.0;
	double beta = 0.0;

	double alpha1 = 1.0;
	double beta1 = 0.0;
    // create CUDA event handles
//    cudaEvent_t start_event, stop_event;
//    checkCudaErrors(cudaEventCreate(&start_event));
//    checkCudaErrors(cudaEventCreate(&stop_event));
//    checkCudaErrors(cudaEventRecord(start_event, 0));
	double startTime = omp_get_wtime();
	cudaSetDevice(0);
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[0], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));


	cudaSetDevice(1);
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));
    checkCublasErrors(cublasDgemm(cublasHandle[1], CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha1, d_B1, Q, d_A1, N, &beta1, d_C1, Q));


    cudaDeviceSynchronize();
	double endTime = omp_get_wtime();
	cout << "Runtime:" << endTime - startTime << endl;
//	for (int i=0; i<nkernels; ++i)
//	{
//		dim3 gridSize(8,8);
//		dim3 blockSize(16,16);
//		myKernel<<<gridSize,blockSize,0,streams[i]>>>(&d_a[i]);
//	}
//    checkCudaErrors(cudaEventRecord(stop_event, 0));
//    checkCudaErrors(cudaEventSynchronize(stop_event));
//    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
//    cout<<"Time:"<<elapsed_time<<endl;
    // release resources
    for (int i = 0; i < nkernels; i++)
    {
        checkCublasErrors(cublasDestroy(cublasHandle[i]));
    }

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);
    cudaFree(d_a);
    free(a);
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("Test passed\n");
}

void SetTensorDesc(cudnnTensorDescriptor_t& tensorDesc, cudnnDataType_t dataType, int n, int c, int h, int w) {
	const int nDims = 4;
	int dimA[nDims] = { n, c, h, w };
	int strideA[nDims] = { c * h * w, h * w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, dataType, 4, dimA, strideA));
}

void testConnvOnSameDevice(){

	int m_convAlgorithm;
	cudnnDataType_t m_dataType;
	cudnnHandle_t m_cudnnHandle[2];
	cudnnTensorDescriptor_t m_srcTensorDesc[2], m_dstTensorDesc[2], m_biasTensorDesc[2];
	cudnnFilterDescriptor_t m_filterDesc[2];
	cudnnConvolutionDescriptor_t m_convDesc[2];

	m_dataType = CUDNN_DATA_DOUBLE;
	m_convAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

	checkCUDNN(cudnnCreate(&m_cudnnHandle[0]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc[0]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc[0]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_biasTensorDesc[0]));
	checkCUDNN(cudnnCreateFilterDescriptor(&m_filterDesc[0]));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDesc[0]));

	checkCUDNN(cudnnCreate(&m_cudnnHandle[1]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc[1]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc[1]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_biasTensorDesc[1]));
	checkCUDNN(cudnnCreateFilterDescriptor(&m_filterDesc[1]));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDesc[1]));

	int batchSize = 10;
	//conv2
	int kWidth = 5, kHeight = 5;
	int inMaps = 96, outMaps = 256;
	int inMapSize = 27, outMapSize = 27;

	//conv3
//	int kWidth = 3, kHeight = 3;
//	int inMaps = 256, outMaps = 384;
//	int inMapSize = 13, outMapSize = 13;

	int n, c, h, w;
	double* deviceInData;
	double* deviceOutData;
	double* deviceWeight;

	double* deviceInData1;
	double* deviceOutData1;
	double* deviceWeight1;

	n = batchSize;
	c = inMaps;
	h = inMapSize;
	w = inMapSize;
	SetTensorDesc(m_srcTensorDesc[0], m_dataType, n, c, h, w);
	SetTensorDesc(m_srcTensorDesc[1], m_dataType, n, c, h, w);
	cudaMalloc(&deviceInData, n * c * h * w * sizeof(double));
	cudaMalloc(&deviceInData1, n * c * h * w * sizeof(double));

	c = outMaps;
	h = outMapSize;
	w = outMapSize;
	SetTensorDesc(m_dstTensorDesc[0], m_dataType, n, c, h, w);
	SetTensorDesc(m_dstTensorDesc[1], m_dataType, n, c, h, w);
	cudaMalloc(&deviceOutData, n * c * h * w * sizeof(double));
	cudaMalloc(&deviceOutData1, n * c * h * w * sizeof(double));

	const int tensorDims = 4;
	const int filterDimA[tensorDims] = { outMaps, inMaps, kWidth, kHeight };
	checkCUDNN(cudnnSetFilterNdDescriptor(m_filterDesc[0], m_dataType, CUDNN_TENSOR_NCHW, tensorDims, filterDimA));
	checkCUDNN(cudnnSetFilterNdDescriptor(m_filterDesc[1], m_dataType, CUDNN_TENSOR_NCHW, tensorDims, filterDimA));

	const int convDims = 2;
	int padA[convDims] = { 0, 0 };
	int filterStrideA[convDims] = { 1, 1 };
	int upscaleA[convDims] = { 1, 1 };
	checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convDesc[0], convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, m_dataType));
	checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convDesc[1], convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, m_dataType));

	long weightSize = kWidth * kHeight * outMaps * inMaps;
	cudaMalloc(&deviceWeight, weightSize * sizeof(double));
	cudaMalloc(&deviceWeight1, weightSize * sizeof(double));

	size_t m_fwdDataSizeInBytes;
	void * m_fwdDataWorkSpace;
	void * m_fwdDataWorkSpace1;
	cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t) m_convAlgorithm;

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnnHandle[0], m_srcTensorDesc[0], m_filterDesc[0], m_convDesc[0], m_dstTensorDesc[0], algo, &m_fwdDataSizeInBytes));
	if (m_fwdDataSizeInBytes != 0) {
		checkCudaErrors(cudaMalloc(&m_fwdDataWorkSpace, m_fwdDataSizeInBytes));
		checkCudaErrors(cudaMalloc(&m_fwdDataWorkSpace1, m_fwdDataSizeInBytes));
	}
	cudaStream_t streams[2];
	for (int i = 0; i < 2; i++)
	{
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}


	double alpha = 1.0;
	double beta = 0.0;
	cudnnSetStream(m_cudnnHandle[0],streams[0]);

	for(int i=0;i<4;i++){
		checkCUDNN(cudnnConvolutionForward(m_cudnnHandle[0], //
					&alpha, //
					m_srcTensorDesc[0], //
					deviceInData, //the input data of current layer ,also is the output data of preLayer.
					m_filterDesc[0], //
					deviceWeight, //
					m_convDesc[0], //
					algo, //
					m_fwdDataWorkSpace, //
					m_fwdDataSizeInBytes, //
					&beta, //
					m_dstTensorDesc[0], //
					deviceOutData //
					));
	}
	cudnnSetStream(m_cudnnHandle[1],streams[1]);

	for(int i=0;i<4;i++){
		checkCUDNN(cudnnConvolutionForward(m_cudnnHandle[1], //
					&alpha, //
					m_srcTensorDesc[1], //
					deviceInData1, //the input data of current layer ,also is the output data of preLayer.
					m_filterDesc[1], //
					deviceWeight1, //
					m_convDesc[1], //
					algo, //
					m_fwdDataWorkSpace1, //
					m_fwdDataSizeInBytes, //
					&beta, //
					m_dstTensorDesc[1], //
					deviceOutData1 //
					));
	}

	cudaDeviceSynchronize();

	checkCudaErrors(cudaFree(deviceInData));
	checkCudaErrors(cudaFree(deviceOutData));
	checkCudaErrors(cudaFree(deviceWeight));
	checkCudaErrors(cudaFree(m_fwdDataWorkSpace));


	checkCudaErrors(cudaFree(deviceInData1));
	checkCudaErrors(cudaFree(deviceOutData1));
	checkCudaErrors(cudaFree(deviceWeight1));
	checkCudaErrors(cudaFree(m_fwdDataWorkSpace1));

	checkCUDNN(cudnnDestroy(m_cudnnHandle[0]));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(m_convDesc[0]));
	checkCUDNN(cudnnDestroyFilterDescriptor(m_filterDesc[0]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc[0]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc[0]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_biasTensorDesc[0]));
	checkCUDNN(cudnnDestroy(m_cudnnHandle[1]));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(m_convDesc[1]));
	checkCUDNN(cudnnDestroyFilterDescriptor(m_filterDesc[1]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc[1]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc[1]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_biasTensorDesc[1]));

	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

}

int m_convAlgorithm;
cudnnDataType_t m_dataType;
cudnnTensorFormat_t m_tensorFormat;
cudnnHandle_t m_cudnnHandle[2];
cudnnTensorDescriptor_t m_srcTensorDesc[2], m_dstTensorDesc[2], m_biasTensorDesc[2];
cudnnFilterDescriptor_t m_filterDesc[2];
cudnnConvolutionDescriptor_t m_convDesc[2];
double* deviceInData;
double* deviceOutData;
double* deviceWeight;

double* deviceInData1;
double* deviceOutData1;
double* deviceWeight1;
void * m_fwdDataWorkSpace;
void * m_fwdDataWorkSpace1;

void testConnvOnDiffDevice(){



	m_dataType = CUDNN_DATA_DOUBLE;
	m_convAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	m_tensorFormat = CUDNN_TENSOR_NCHW;

//	pthread_t t1,t2;
//	int ret;
//	ret = pthread_create(&t1, NULL, func3, NULL);
//	if (ret != 0) {
//		LOG(INFO)<<"Error";
//	}
//
//	ret = pthread_create(&t2, NULL, func4, NULL);
//	if (ret != 0) {
//		LOG(INFO)<<"Error";
//	}
//
//	pthread_join(t1,NULL);
//	pthread_join(t2,NULL);

	cudaSetDevice(0);
	checkCUDNN(cudnnCreate(&m_cudnnHandle[0]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc[0]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc[0]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_biasTensorDesc[0]));
	checkCUDNN(cudnnCreateFilterDescriptor(&m_filterDesc[0]));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDesc[0]));

	cudaSetDevice(1);
	checkCUDNN(cudnnCreate(&m_cudnnHandle[1]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_srcTensorDesc[1]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_dstTensorDesc[1]));
	checkCUDNN(cudnnCreateTensorDescriptor(&m_biasTensorDesc[1]));
	checkCUDNN(cudnnCreateFilterDescriptor(&m_filterDesc[1]));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&m_convDesc[1]));

	int batchSize = 100;
	int kWidth = 3, kHeight = 3;
	int inMaps = 256, outMaps = 384;
	int inMapSize = 13, outMapSize = 13;

	int n, c, h, w;


	n = batchSize;
	c = inMaps;
	h = inMapSize;
	w = inMapSize;
	cudaSetDevice(0);
	SetTensorDesc(m_srcTensorDesc[0], m_dataType, n, c, h, w);
	cudaMalloc(&deviceInData, n * c * h * w * sizeof(double));

	cudaSetDevice(1);
	SetTensorDesc(m_srcTensorDesc[1], m_dataType, n, c, h, w);
	cudaMalloc(&deviceInData1, n * c * h * w * sizeof(double));

	c = outMaps;
	h = outMapSize;
	w = outMapSize;

	const int tensorDims = 4;
	const int filterDimA[tensorDims] = { outMaps, inMaps, kWidth, kHeight };
	const int convDims = 2;
	int padA[convDims] = { 0, 0 };
	int filterStrideA[convDims] = { 1, 1 };
	int upscaleA[convDims] = { 1, 1 };
	long weightSize = kWidth * kHeight * outMaps * inMaps;

	size_t m_fwdDataSizeInBytes;

	cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t) m_convAlgorithm;

	cudaSetDevice(0);
	SetTensorDesc(m_dstTensorDesc[0], m_dataType, n, c, h, w);
	cudaMalloc(&deviceOutData, n * c * h * w * sizeof(double));
	checkCUDNN(cudnnSetFilterNdDescriptor(m_filterDesc[0], m_dataType, CUDNN_TENSOR_NCHW, tensorDims, filterDimA));
	checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convDesc[0], convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, m_dataType));
	cudaMalloc(&deviceWeight, weightSize * sizeof(double));

	cudaSetDevice(1);
	SetTensorDesc(m_dstTensorDesc[1], m_dataType, n, c, h, w);
	cudaMalloc(&deviceOutData1, n * c * h * w * sizeof(double));
	checkCUDNN(cudnnSetFilterNdDescriptor(m_filterDesc[1], m_dataType, CUDNN_TENSOR_NCHW, tensorDims, filterDimA));
	checkCUDNN(cudnnSetConvolutionNdDescriptor(m_convDesc[1], convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, m_dataType));
	cudaMalloc(&deviceWeight1, weightSize * sizeof(double));

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_cudnnHandle[0], m_srcTensorDesc[0], m_filterDesc[0], m_convDesc[0], m_dstTensorDesc[0], algo, &m_fwdDataSizeInBytes));
	if (m_fwdDataSizeInBytes != 0) {
		cudaSetDevice(0);
		checkCudaErrors(cudaMalloc(&m_fwdDataWorkSpace, m_fwdDataSizeInBytes));
		cudaSetDevice(1);
		checkCudaErrors(cudaMalloc(&m_fwdDataWorkSpace1, m_fwdDataSizeInBytes));
	}
//	cudaStream_t streams[2];
//	for (int i = 0; i < 2; i++)
//	{
//		checkCudaErrors(cudaStreamCreate(&(streams[i])));
//	}


	double alpha = 1.0;
	double beta = 0.0;
	//cudnnSetStream(m_cudnnHandle[0],streams[0]);
	cudaSetDevice(0);

	for(int i=0;i<4;i++){
		checkCUDNN(cudnnConvolutionForward(m_cudnnHandle[0], //
					&alpha, //
					m_srcTensorDesc[0], //
					deviceInData, //the input data of current layer ,also is the output data of preLayer.
					m_filterDesc[0], //
					deviceWeight, //
					m_convDesc[0], //
					algo, //
					m_fwdDataWorkSpace, //
					m_fwdDataSizeInBytes, //
					&beta, //
					m_dstTensorDesc[0], //
					deviceOutData //
					));
	}
	//cudnnSetStream(m_cudnnHandle[1],streams[1]);
	cudaSetDevice(1);
	for(int i=0;i<4;i++){
		checkCUDNN(cudnnConvolutionForward(m_cudnnHandle[1], //
					&alpha, //
					m_srcTensorDesc[1], //
					deviceInData1, //the input data of current layer ,also is the output data of preLayer.
					m_filterDesc[1], //
					deviceWeight1, //
					m_convDesc[1], //
					algo, //
					m_fwdDataWorkSpace1, //
					m_fwdDataSizeInBytes, //
					&beta, //
					m_dstTensorDesc[1], //
					deviceOutData1 //
					));
	}
	cudaDeviceSynchronize();

//	pthread_t t3,t4;
//
//	ret = pthread_create(&t3, NULL, func5, NULL);
//	if (ret != 0) {
//		LOG(INFO)<<"Error";
//	}
//
//	ret = pthread_create(&t4, NULL, func6, NULL);
//	if (ret != 0) {
//		LOG(INFO)<<"Error";
//	}
//
//	pthread_join(t1,NULL);
//	pthread_join(t2,NULL);

	cudaSetDevice(0);
	checkCudaErrors(cudaFree(deviceInData));
	checkCudaErrors(cudaFree(deviceOutData));
	checkCudaErrors(cudaFree(deviceWeight));
	checkCudaErrors(cudaFree(m_fwdDataWorkSpace));

	checkCUDNN(cudnnDestroy(m_cudnnHandle[0]));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(m_convDesc[0]));
	checkCUDNN(cudnnDestroyFilterDescriptor(m_filterDesc[0]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc[0]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc[0]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_biasTensorDesc[0]));

	cudaSetDevice(1);
	checkCudaErrors(cudaFree(deviceInData1));
	checkCudaErrors(cudaFree(deviceOutData1));
	checkCudaErrors(cudaFree(deviceWeight1));
	checkCudaErrors(cudaFree(m_fwdDataWorkSpace1));

	checkCUDNN(cudnnDestroy(m_cudnnHandle[1]));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(m_convDesc[1]));
	checkCUDNN(cudnnDestroyFilterDescriptor(m_filterDesc[1]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_srcTensorDesc[1]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_dstTensorDesc[1]));
	checkCUDNN(cudnnDestroyTensorDescriptor(m_biasTensorDesc[1]));

//	cudaStreamDestroy(streams[0]);
//	cudaStreamDestroy(streams[1]);

}

void testD2DOnDiffDevice(){

	double startTime,endTime;
	long size=1024*1024*10;
	long sizeC=1024*1024*500;
	double* deviceA=NULL;
	double* deviceB=NULL;
	double* deviceA2=NULL;
	double* deviceB2=NULL;
	int* deviceC=NULL;

	int nStreams=3;
	cudaStream_t streams[nStreams];
	cublasHandle_t cublasHandle;

	double *d_A,*d_B,*d_C;
    //fc7
    int M = 512;  //batch_size
	int N = 4096; //in
	int Q = 4096; //out

	long size_A = (M * N); //input_data
	long size_B = (N * Q); //weight
	long size_C = (M * Q); //output_data

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&deviceA,size*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&deviceA2,size*sizeof(double)));

	checkCudaErrors(cudaSetDevice(1));
	for(int i=0;i<nStreams;++i){
		cudaStreamCreate(&streams[i]);
	}
	checkCublasErrors(cublasCreate(&cublasHandle));
	checkCudaErrors(cudaMalloc((void**)&deviceB,size*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&deviceB2,size*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&deviceC,sizeC*sizeof(int)));

	checkCudaErrors(cudaMalloc(&d_A, size_A * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_B, size_B * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_C, size_C * sizeof(double)));

	startTime = omp_get_wtime();
	//checkCudaErrors(cudaMemcpyAsync(deviceB,deviceA,size*sizeof(double),cudaMemcpyDeviceToDevice,streams[0]));
	checkCudaErrors(cudaMemcpyAsync(deviceB2,deviceA2,size*sizeof(double),cudaMemcpyDeviceToDevice,streams[1]));
	double alpha = 1.0;
	double beta = 0.0;
    checkCublasErrors(cublasSetStream(cublasHandle,streams[2]));
    checkCublasErrors(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, Q, M, N, &alpha, d_B, Q, d_A, N, &beta, d_C, Q));
	//myKernel<<<1,1,0,streams[2]>>>(deviceC);
    cudaDeviceSynchronize();
	endTime = omp_get_wtime();
	cout << "Runtime3:" << endTime - startTime << endl;

	cudaSetDevice(0);
	cudaFree(deviceA);
	cudaFree(deviceA2);

	cudaSetDevice(1);
	for(int i=0;i<nStreams;++i){
		cudaStreamDestroy(streams[i]);
	}
	 checkCublasErrors(cublasDestroy(cublasHandle));
	cudaFree(deviceB2);
	cudaFree(deviceB);
	cudaFree(deviceC);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
}

int main(int argc, char *argv[]) {

	//testKernelWithStreams();
	//testGemmsOnDiffDevice();
	//testGemmsOnSameDevice();
	//testConnvOnSameDevice();
	//testConnvOnDiffDevice();
	//testD2DOnDiffDevice();
	//testDgemm();

	cudaStream_t* stream=NULL;
	assert(stream==NULL);
	return 0;
}

