#include "DataLayerSpiking.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
//#include <thread>
#include "../common/Config.h"
#include "../common/cuBase.h"
#include "../common/util.h"
#include "../dataAugmentation/cuTransformation.cuh"

#define CONST_SPIKING_SCALE (5.5 * 255.0f)

curandGenerator_t rand_gen_device;
const curandRngType_t gen_t = CURAND_RNG_PSEUDO_DEFAULT;

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(outputSize * endTime, 1024));
*/
__global__ void g_DataLayerSpiking_feedforward(
	bool** inputs,
	bool* outputs,
    int outputArea,
    int outputCols);

DataLayerSpiking::DataLayerSpiking(std::string name){
	m_name = name;
    myId = 0;

    ConfigDataSpiking* config = (ConfigDataSpiking*)Config::instance()->getLayerByName(m_name);
	inputSize  = config->m_inputNeurons;
	outputSize = inputSize;
    endTime   = Config::instance()->getEndTime();
	batch     = Config::instance()->getBatchSize();

    inputDim    = Config::instance()->getImageSize();
    outputDim   = Config::instance()->getImageSize();
	inputAmount = Config::instance()->getChannels();
	outputAmount= inputAmount;
	outputs = new cuMatrix<bool>(batch, endTime * outputSize, outputAmount);
    outputs_time = new cuMatrix<int>(batch, outputSize * endTime, outputAmount);

    fireCount = new cuMatrix<int>(batch, outputSize, outputAmount);

    cu_randomNum = new cuMatrix<float>(batch, endTime * inputSize, 1);

    bool has_distortion = Config::instance()->applyPreproc();
    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < batch; j++){
            batchSpeeches[i].push_back(new cuMatrix<bool>(endTime, inputSize, Config::instance()->getChannels()));
            if(has_distortion){
                batchSamplesFloat[i].push_back(new cuMatrix<float>(outputDim, outputDim, Config::instance()->getChannels()));
            }
        }
        batchSpeeches[i].toGpu();
        if(has_distortion)
            batchSamplesFloat[i].toGpu();
    }
    if(has_distortion){
        for(int i = 0; i < batch; ++i){
            processOutputs.push_back(new cuMatrix<float>(outputDim, outputDim, Config::instance()->getChannels()));
        }
        processOutputs.toGpu();
    }

	checkCudaErrors(cudaStreamCreate(&stream1));

	Layers::instance()->set(m_name, this);

	curandStatus_t curandstatus = curandCreateGenerator(&rand_gen_device, gen_t);
	if(curandstatus != CURAND_STATUS_SUCCESS)
	{
		char logStr[1024];
		sprintf(logStr, "DataLayerSpiking::curandCreateGenerator fail\n");
		LOG(logStr, "Result/log.txt");
		assert(0);
	}
    
}


/*
 * dim3 block = dim3(batch, inputSize);
 * dim3 thread= dim3(min(1024, endTime));
*/
__global__ void g_DataLayerSpiking_poissonCode(
    float** preprocs,
    bool** inputs,
    float* _randoms,
    int batch,
    int inputSize,
    int endTime)
{
    int batchId = blockIdx.x;
    int i_idx = blockIdx.y;
    int speechSize = endTime * inputSize;

    float * random = _randoms + batchId * speechSize;
    float * preproc = preprocs[batchId];
    bool * input = inputs[batchId];
    float distorted = preproc[i_idx];
    float freq = ((distorted + 1) * 255.0f / 2) / CONST_SPIKING_SCALE; // map back to freq range;
    for(int t = 1; t < endTime; t += blockDim.x)
    {
        int time = t + threadIdx.x;
        float r = random[time * inputSize + i_idx];
        if(r < freq)    input[time * inputSize + i_idx] = true;
        else    input[time * inputSize + i_idx] = false;
    }

}

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(outputSize * endTime, 1024));
*/

__global__ void g_DataLayerSpiking_feedforward(
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
 * dim3 thread= dim3(min(outputSize, 1024));
*/
__global__ void g_DataLayerSpiking_get_fireCount(
    bool* outputs,
    int* batchfireCount,
    int outputSize,
    int endTime)
{
	int batchId = blockIdx.x;

    bool* output = outputs + batchId * endTime * outputSize;
    int* fireCount = batchfireCount + batchId * outputSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize){
            int sum = 0;
            for(int time = 0; time < endTime; ++time)   sum += output[o_idx + time * outputSize];
            fireCount[o_idx] = sum;
        }
    }
}


//* simply copy the input data to the output
void DataLayerSpiking::feedforward(){
	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(min(outputSize * endTime, 1024));
	
	g_DataLayerSpiking_feedforward<<<block, thread>>>(
		batchSpeeches[myId].m_devPoint, 
		outputs->getDev(),
		outputs->getArea(),
        outputs->cols);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("DataLayerSpiking:feedforward");

    //* get the fire counts for transforming the binary response to spike times    
    thread = dim3(min(outputSize, 1024));
    g_DataLayerSpiking_get_fireCount<<<block, thread>>>(
        outputs->getDev(),
        fireCount->getDev(),
        outputSize,
        endTime);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("DataLayerSpiking:g_DataLayerSpiking_get_fireCount");
    

    g_response_2_spiketime<<<block, thread>>>(
        outputs->getDev(),
        outputs_time->getDev(),
        outputs->getArea(),
        outputSize,
        endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("DataLayerSpiking:g_response_2_spiketime");

}; 

//* apply the distortation here
void DataLayerSpiking::getBatchSpikesWithPreproc(cuMatrixVector<bool>& inputs, int start)
{
    assert(Config::instance()->applyPreproc() == true);
    
    // generate the uniform random number
    generateRandom(clock() + start);

    // cp the float raw sample to Gpu
    int id = 1 - this->myId;
    for(size_t i = 0; i < this->batchSamplesFloat[id].size(); i++){
        memcpy(this->batchSamplesFloat[id][i]->getHost(), inputs[i + start]->getHostRawImg(), sizeof(float) * this->batchSamplesFloat[id][i]->getLen());
        this->batchSamplesFloat[id][i]->toGpu(this->stream1);
        this->batchSpeeches[id][i]->toGpu(this->stream1);
    }
    // apply the distortation
    cuApplyDistortion(batchSamplesFloat[id].m_devPoint, processOutputs.m_devPoint, batch, outputDim); 

    // map the distorted samples to spike times
    g_DataLayerSpiking_poissonCode<<<dim3(batch, inputSize), dim3(min(1024, endTime))>>>(processOutputs.m_devPoint, batchSpeeches[id].m_devPoint, cu_randomNum->getDev(), batch, inputSize, endTime);
    /* // do the same thing by CPU
    for(size_t i = 0; i < this->processOutputs.size(); i++){
        processOutputs[i]->toCpu();
        convertToSpikeTimes(processOutputs[i], inputs[i+start]->getSpikeTimes(), outputDim, endTime);
    }
    */
    // show the distorted image
    if (Config::instance()->getImageShow()) {
		for (int ff = batch - 1; ff >= 0; ff--) {
			showImg(batchSamplesFloat[id][ff], 5);
            processOutputs[ff]->toCpu();
			showImg(processOutputs.m_vec[ff], 5);
			cv::waitKey(0);
		}
	}
}

void DataLayerSpiking::testData()
{
}

//* generate the random numbers for map preproc samples to poisson spike trains
void DataLayerSpiking::generateRandom(unsigned long long seed)
{
	curandGenerateUniform(rand_gen_device, cu_randomNum->getDev(), cu_randomNum->getLen());
}

void DataLayerSpiking::synchronize(){
    myId = 1 - myId;
    cudaStreamSynchronize(this->stream1);
}

//* get the input spike trains in batch from the input speeches streams
void DataLayerSpiking::getBatchSpikes(cuMatrixVector<bool>& inputs, int start){
    int id = 1 - this->myId;
    for(size_t i = 0; i < this->batchSpeeches[id].size(); i++){
        inputs[i+start]->sparseToDense();
        memcpy(this->batchSpeeches[id][i]->getHost(), inputs[i + start]->getHost(), sizeof(bool) * this->batchSpeeches[id][i]->getLen());
        this->batchSpeeches[id][i]->toGpu(this->stream1);
        inputs[i+start]->freeCpuMem();
        //this->batchSpeeches[i]->toGpu();
    }
}


void DataLayerSpiking::loadBatchSpikes(cuMatrixVector<bool>& inputs, int start){
    if(Config::instance()->applyPreproc() == true)
        getBatchSpikesWithPreproc(inputs, start);
    else
        getBatchSpikes(inputs, start);
}
