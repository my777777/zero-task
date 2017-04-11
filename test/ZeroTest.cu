/*
 * ZeroTest.cu
 *
 *  Created on: Dec 26, 2016
 *      Author: zys
 */
#include "ZeroTest.cuh"

__global__ void fillKernel(int *a, long n, long offset) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n) {
		register long delay=0;
		for (int i = 0; i < 1000; i++) {
			delay = 100000000;
			while (delay > 0) {
				delay--;
			}
		}
		a[tid] = delay + offset + tid;
	}
}

__global__ void fillKernel2(int *a, long n, long offset) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n) {
		register long delay=0;
		for (int i = 0; i < 1000; i++) {
			delay = 100000000;
			while (delay > 0) {
				delay--;
			}
		}
		a[tid] = delay + offset + tid;
	}
}

