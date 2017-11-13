#include "DataLayerSpiking.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
//#include <thread>
#include "../common/Config.h"
#include "../common/cuBase.h"
#include "../common/util.h"


/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(outputDim * endTime, 1024));
*/
__global__ void g_dataLayer_spiking_feedforward(
	bool** inputs,
	bool* outputs,
    int outputArea,
    int outputCols);

DataLayerSpiking::DataLayerSpiking(std::string name){
	m_name = name;
    myId = 0;

    ConfigDataSpiking* config = (ConfigDataSpiking*)Config::instance()->getLayerByName(m_name);
	inputDim  = config->m_inputNeurons;
	outputDim = inputDim;
    endTime   = Config::instance()->getEndTime();
	batch     = Config::instance()->getBatchSize();
	inputAmount = Config::instance()->getChannels();
	outputAmount= inputAmount;
	outputs = new cuMatrix<bool>(batch, outputDim * endTime, outputAmount);
    outputs_time = new cuMatrix<int>(batch, outputDim * endTime, outputAmount);

    fireCount = new cuMatrix<int>(batch, outputDim, outputAmount);

    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < batch; j++){
            batchSpeeches[i].push_back(new cuMatrix<bool>(endTime, inputDim, Config::instance()->getChannels()));
        }
        batchSpeeches[i].toGpu();
    }

	checkCudaErrors(cudaStreamCreate(&stream1));

	Layers::instance()->set(m_name, this);
}

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(outputDim * endTime, 1024));
*/

__global__ void g_dataLayer_spiking_feedforward(
	bool** inputs,
	bool* outputs,
    int outputArea,
    int outputCols)
{
	int batchId = blockIdx.x;
    int ok      = blockIdx.y;

    int outputAmount = gridDim.y;

	bool* input  = inputs[batchId];
	bool* output = outputs + ok * outputArea+ batchId * outputCols * outputAmount;
	for(int i = 0; i < outputCols; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < outputCols){
			output[idx] = input[idx];
		}
	}
}

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(outputDim, 1024));
*/
__global__ void g_dataLayer_get_fireCount(
    bool* outputs,
    int* batchfireCount,
    int outputDim,
    int endTime)
{
	int batchId = blockIdx.x;

    bool* output = outputs + batchId * endTime * outputDim;
    int* fireCount = batchfireCount + batchId * outputDim;

    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        int sum = 0;
        for(int time = 0; time < endTime; ++time)   sum += output[o_idx + time * outputDim];
        fireCount[o_idx] = sum;
    }
}


//* simply copy the input data to the output
void DataLayerSpiking::feedforward(){
	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(min(outputDim * endTime, 1024));
	
	g_dataLayer_spiking_feedforward<<<block, thread>>>(
		batchSpeeches[myId].m_devPoint, 
		outputs->getDev(),
		outputs->getArea(),
        outputs->cols);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("DataLayerSpiking:feedforward");

    //* get the fire counts for transforming the binary response to spike times    
    thread = dim3(min(outputDim, 1024));
    g_dataLayer_get_fireCount<<<block, thread>>>(
        outputs->getDev(),
        fireCount->getDev(),
        outputDim,
        endTime);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("DataLayerSpiking:g_dataLayer_get_fireCount");
    

    g_response_2_spiketime<<<block, thread>>>(
        outputs->getDev(),
        outputs_time->getDev(),
        outputDim,
        endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("DataLayerSpiking:g_response_2_spiketime");

}; 

void DataLayerSpiking::trainData()
{
}

void DataLayerSpiking::testData()
{
}


void DataLayerSpiking::synchronize(){
    myId = 1 - myId;
    cudaStreamSynchronize(this->stream1);
}

//* get the input spike trains in batch from the input speeches streams
void DataLayerSpiking::getBatchSpikesWithStreams(cuMatrixVector<bool>& inputs, int start){
    int id = 1 - this->myId;
    for(size_t i = 0; i < this->batchSpeeches[id].size(); i++){
        memcpy(this->batchSpeeches[id][i]->getHost(), inputs[i + start]->getHost(), sizeof(bool) * this->batchSpeeches[id][i]->getLen());
        this->batchSpeeches[id][i]->toGpu(this->stream1);
        //this->batchSpeeches[i]->toGpu();
    }
}
