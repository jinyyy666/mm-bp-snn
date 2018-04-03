#include "SoftMaxSpiking.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../common/util.h"
#include "../readData/readSpeechData.h"
#include <fstream>
#include <assert.h>
#include <math.h>

//#define DEBUG
#define I_IDX 0
#define O_IDX 0

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(1024, outputSize));
 */
__global__ void g_getSoftMaxP(
    float* softMaxP, 
    int* batchFireCount, 
    int outputSize);
    
/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_getSoftMaxCost_output(
    int*   fireCount,
    float* groundTruth,
    float* cost,
    int*   y,
    int    batch,
    int    cols);


void SoftMaxSpiking::calCost()
{
    cost->gpuClear();
    if(predict == NULL){
        printf("Warning::Try to compute the cost when the predict is not properly set!\n ");
        return;
    }
    g_getCost_1<<<dim3(1), dim3(256), sizeof(float) * 256>>>(softMaxP->getDev(),
            groundTruth->getDev(),
            cost->getDev(),
            predict,
            softMaxP->rows,
            softMaxP->cols,
            batch);
    cudaStreamSynchronize(0);
    getLastCudaError("SoftMaxSpiking:g_getCost_1");
}

void SoftMaxSpiking::feedforward()
{
    if((inputs == NULL))
    {
        printf("SoftMaxSpiking init error\n");
        exit(0);
    }

    // fast input response
    g_cast_bool_2_float<<<dim3(batch, endTime), min(1024, inputSize)>>>(inputs->getDev(), endTime, inputSize, inputs->cols, inputs->channels, batch, inputs_float->getDev());
    matrixMul(w, inputs_float, inputs_resp_tmp); //input_resp_tmp rows:outputSize; cols:endTime*batch
    g_transform_2_batch<<<dim3(batch, outputSize), min(1024, endTime)>>>(inputs_resp_tmp->getDev(), endTime, outputSize, batch, inputs_resp->getDev());

    // convert (batch, inputDim2*endTime, amount) to (batch, amount*inputDim2*endTime, 1)
    g_convert_spiketimes<<<dim3(batch, endTime), min(1024, inputSize)>>>(inputs_time->getDev(), endTime, inputSize, inputs_time->cols, inputs_time->channels, batch, inputs_time_format->getDev());
    // convert (batch, inputDim2, amount) to (batch, amount*inputDim2, 1)
    g_convert_firecounts<<<dim3(batch), min(1024, inputSize)>>>(preFireCount->getDev(), preFireCount->getArea(), inputSize, preFireCount->cols, preFireCount->channels, batch, preFireCount_format->getDev());

    dim3 thread= dim3(min(1024, outputSize));
    dim3 block = dim3(batch);
    ConfigSpiking * config = (ConfigSpiking*) Config::instance()->getLayerByName(m_name); 
    int dummyFreq = config->getBiasFreq();

    g_Spiking_feedforward<<<block, thread>>>(
            inputs_resp->getDev(),
            w->getDev(),
            w_laterial == NULL ? NULL : w_laterial->getDev(),
            b->getDev(),
            outputs->getDev(),
            fireCount->getDev(),
            inputSize,
            outputSize,
            endTime,
            threshold,
            dummyFreq,
            T_REFRAC,
            tau->getDev(),
            res->getDev(),
            TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("SoftMaxSpiking::g_Spiking_feedforward");

    block = dim3(batch, 1);
    thread = dim3(min(outputSize, 1024));

    // transform the binary response matrix to the spike times
    g_response_2_spiketime<<<block, thread>>>(
            outputs->getDev(),
            outputs_time->getDev(),
            outputs->getArea(),
            outputSize,
            endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("SoftMaxSpiking:g_response_2_spiketime");

    g_getSoftMaxP<<<block, thread, sizeof(float) * outputSize * 2>>>(softMaxP->getDev(), fireCount->getDev(), outputSize);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("SoftMaxSpiking:g_getSoftMaxP");
   
}

void SoftMaxSpiking::backpropagation()
{ 
    // compute the cost function
    g_getCost_1<<<dim3(1), dim3(256), sizeof(float) * 256>>>(softMaxP->getDev(), groundTruth->getDev(), cost->getDev(), predict, softMaxP->rows, softMaxP->cols, batch);
    cudaStreamSynchronize(0);
    getLastCudaError("SoftMaxSpiking::g_getCost_1");

    // compute the delta (error)
    g_getSoftMaxDelta<<<dim3(1), dim3(256)>>>(curDelta->getDev(), softMaxP->getDev(), groundTruth->getDev(), curDelta->getLen());
    cudaStreamSynchronize(0);
    getLastCudaError("SoftMaxSpiking::g_getSoftMaxDelta");

    // apply the sample weights
    g_boostWeight_output<<<dim3(batch), dim3(outputSize)>>>(curDelta->getDev(), sample_weights, curDelta->getLen());
    cudaStreamSynchronize(0);
    getLastCudaError("SoftMaxSpiking::g_boostWeight_output");

    // compute the lateral factors if applicable
    if(lateralFactor != NULL && w_laterial != NULL){
        int threads = min(outputSize, 1024);
        g_getLateralFactor_output<<<dim3(batch, outputSize), threads, sizeof(float) * threads>>>(
            outputs_time->getDev(),
            fireCount->getDev(),
            lateralW,
            predict,
            lateralFactor->getDev(),
            threshold,
            outputSize,
            endTime,
            T_REFRAC,
            tau->getDev(),
            TAU_S);
        cudaStreamSynchronize(0);
        getLastCudaError("SoftMaxSpiking::g_getLateralFactor_output");
    }

    // modify the output spikes of the target neuron if it does not fire
    // tricky: modify both the spike trains and output fire counts!
    g_modifySpikes<<<dim3(batch), dim3(min(outputSize, 1024))>>>(outputs->getDev(), predict, fireCount->getDev(), DESIRED_LEVEL, endTime, outputSize);
    cudaStreamSynchronize(0);
    getLastCudaError("SoftMaxSpiking::g_modifySpikes");

    // retransform the binary matrix to the spike times since the outputs might be changed
    g_response_2_spiketime<<<dim3(batch, 1), dim3(min(outputSize, 1024))>>>(
            outputs->getDev(),
            outputs_time->getDev(),
            outputs->getArea(),
            outputSize,
            endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("SoftMaxSpiking:g_response_2_spiketime");

    // pre compute the accumulative synaptic effect, and effect ratio (if applicable)
    dim3 thread = dim3(min(1024, outputSize));
    dim3 block  = dim3(batch, inputSize);
    cudaFuncSetCacheConfig(g_Spiking_synaptic_effect, cudaFuncCachePreferL1);
    g_Spiking_synaptic_effect<<<block, thread>>>(
        inputs_time_format->getDev(),
        outputs_time->getDev(),
        preFireCount_format->getDev(),
        fireCount->getDev(),
        w->getDev(),
        accEffect->getDev(),
        effectRatio == NULL ? NULL : effectRatio->getDev(),
        inputSize,
        outputSize,
        endTime,
        T_REFRAC,
        tau->getDev(),
        TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_SoftMaxSpiking_synaptic_effect");
   
    // divide the curDelta by vth
    block = dim3(batch, 1);
    thread = dim3(min(1024, outputSize));
    g_divide_by_threshold<<<block, thread>>>(curDelta->getDev(), curDelta->getArea(), curDelta->cols, threshold);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_divide_by_threshold");
   
    // compute preDelta: curDelta: batch * outputSize; w: outputSize * inputSize
    if(preDelta == NULL){
        ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
        assert(config->m_input == "data");
    }
    else{
        if(effectRatio != NULL){
            matrixMul(curDelta, effectRatio, preDelta_format);
        }
        else{
            matrixMul(curDelta, w, preDelta_format);
        }
        // preDelta_format: (batch, channels * size, 1) -> preDelta: (batch, size, channels)
        block = batch;
        thread = min(512, preDelta->channels * preDelta->cols);
        g_preDeltaFormat<<<block, thread>>>(preDelta_format->getDev(), preDelta->getDev(),
            preDelta->rows, preDelta->cols, preDelta->channels);
        cudaStreamSynchronize(0);
        getLastCudaError("g_preDeltaFormat");
    } 
}

/*
* block   =  batch
* threads =  outputSize
* shared : sizeof(float) * outputSize * 2
*/
__global__ void g_getSoftMaxP(float* softMaxP, int* batchFireCount, int outputSize)
{
	int batchId = blockIdx.x;
    int tid = threadIdx.x;
	extern __shared__ float _share[];
	float * _max = _share;
	float * _sum = _share + blockDim.x;

	float* sp = softMaxP + batchId * outputSize;
    int* fireCount = batchFireCount + batchId * outputSize;

	_sum[tid] = 0.0;
	_max[tid] = float(fireCount[tid]);
    __syncthreads();
	
    int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < skip && (tid + skip) < len)
		{
            _max[tid] = max(_max[tid], _max[tid + skip]);
		}
		len = skip;
	}
	__syncthreads();

    float prob = __expf(float(fireCount[tid]) - _max[0]);
    sp[tid] = prob; 
    _sum[tid] = prob;

	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < skip && (tid + skip) < len)
		{
			_sum[tid] += _sum[tid + skip];
		}
		len = skip;
	}
	__syncthreads();
    sp[tid] /= _sum[0];
}


SoftMaxSpiking::SoftMaxSpiking(std::string name):
Spiking(name)
{
    softMaxP = new cuMatrix<float>(batch, outputSize, 1);
}
