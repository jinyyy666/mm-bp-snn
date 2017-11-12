#ifndef _NET_SPIKING_CUH_
#define _NET_SPIKING_CUH_

#include "common/cuMatrix.h"
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include "common/cuMatrixVector.h"

/*
 * function               : read the network weight from checkpoint
 * parameter              :
 * path                   : the path for the checkpoint file
 */
void cuReadSpikingNet(const char* path);

/*
 * function: trainning the network
 */

void cuTrainSpikingNetwork(cuMatrixVector<bool>&x, 
	cuMatrix<int>*y ,
	cuMatrixVector<bool>& testX,
	cuMatrix<int>* testY,
	int batch,
	int nclasses,
	std::vector<float>&nlrate,
	std::vector<float>&nMomentum,
	std::vector<int>&epoCount,
	cublasHandle_t handle);

void buildSpikingNetwork(int trainLen, int testLen);

void cuFreeSpikingNet();

void cuFreeSNNMemory(
	int batch,
	cuMatrixVector<bool>&trainX, 
	cuMatrixVector<bool>&testX);

void getSpikingNetworkCost(int* y);

#endif
