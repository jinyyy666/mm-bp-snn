#include "Spiking.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../common/util.h"
#include "../readData/readSpeechData.h"
#include <fstream>
#include <assert.h>
#include <math.h>

//#define DEBUG

/*
 * Device func for accumulate the spike response
 *
*/
__device__ float d_Spiking_accumulate_spikes(
    int inputDim,
    int outputDim,
    bool* input,
    bool* output,
    int o_idx,
    float* weights,
    float* weights_lat,
    float* biases,
    int t,
    int dummyFreq);

/*
 * Device func for spike gradient for each pair of binary spike response
 */
__device__ float d_Spiking_gradient(
    bool* output,
    bool* input,
    float delta,
    int o_idx,
    int i_idx,
    int outputDim,
    int inputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

/*
 * Device func for spike gradient for each pair of spike train in times
 */
__device__ float d_Spiking_gradient_spiketime(
    int* output_time,
    int* input_time,
    int n_ospikes,
    int n_ispikes,
    float delta,
    int o_idx,
    int i_idx,
    float lat_factor,
    int outputDim,
    int inputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

/*
 * Device func for spike gradient for bias in times
 */
__device__ float d_Spiking_bias_gradient_spiketime(
    int* output_time,
    int n_ospikes,
    float delta,
    int o_idx,
    int dummyFreq,
    int outputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);


/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_weight_cp_test(float** ws, int nrows, int ncols);
    
/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_getCost_output(
    int*   fireCount,
    float* groundTruth,
    float* cost,
    int*   y,
    int    batch,
    int    cols,
    float UNDESIRED_LEVEL,
    float DESIRED_LEVEL,
    float MARGIN);

/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_getDelta_output(
    float* outputDelta,
    int*   fireCount,
    float* groundTruth,
    int    len,
    float  MARGIN);

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(outputDim);
 */
__global__ void g_boostWeight_output(
    float* outputDelta,
    float* sample_weights,
    int len);


/*
 * dim3 block = dim3(batch, outputDim);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_getLateralFactor_output(
    int* outputs_time,
    int* batchFireCount,
    float w0,
    int* y,
    float* batchLFactor,
    float vth,
    int outputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);
  
/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(1024, outputDim));
 */
__global__ void g_getMaxCount(
    int* fireCount,
    int* maxCount,
    int cols); 

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(1024, outputDim));
 */
__global__ void g_modifySpikes(
    bool* outputs,
    int* y,
    int* fireCount,
    int target_level,
    int endTime,
    int outputDim);


/*
 * dim3 block = dim3(batch, inputDim, outputAmount);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_wgrad(
        bool* inputs,
        bool* outputs,
        float* curDelta,
        float** wgradTmp,
        int inputDim,
        int outputDim,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);

/*
 * dim3 block = dim3(batch, inputDim, outputAmount);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_wgrad_spiketime(
        float* batchAccEffect,
        float* curDelta,
        float* latFactors,
        float** wgradTmp,
        int inputDim,
        int outputDim);


/*
 * dim3 block = dim3(outputDim);
 * dim3 thread= dim3(batch);
 */
__global__ void g_Spiking_bgrad_spiketime(
        int* outputs_time,
        int* batchFireCount,
        float* curDelta,
        float** bgradTmp,
        int outputDim,
        int endTime,
        int dummyFreq,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);


/*
 * block = dim3(outputDim * inputDim, outputAmount);
 * thread= dim3(batch);
*/
__global__ void g_Spiking_gradAdd(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
    float* w_sq_sum,
	int batch,
	float lambda,
    float beta,
    float limit,
    int inputDim,
	int wArea);


/*
 * dim3 block = dim3(batch, inputDim, outputAmount);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_synaptic_effect(
        int* inputs_time,
        int* outputs_time,
        int* batchPreFireCount,
        int* batchFireCount,
        float* w,
        float* batchAccEffect,
        float* effectRatio,
        int inputDim,
        int outputDim,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);


/*
 *	blocks : dim3(batch, div),
 *	threads: dim3(min(outputDim, 1024), 1);
 */
__global__ void g_Spiking_feedforward(
	bool*  inputs,
	float** ws,
    float** ws_lat,
    float** bs,
	bool*  outputs,
    int*    fireCount,
	int inputDim,
	int outputDim,
    int endTime,
	int inputAmount,
	int outputAmount,
    float vth,
    int dummyFreq,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);


/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_Spiking_sgd_vecAdd(
    float** v_m, 
    float** wgrad, 
    float** w, 
    int lenw, 
    float momentum, 
    float lr);

/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_Spiking_adam_vecAdd(
    float** g1_ws, 
    float** g2_ws, 
    float*  b1_t,
    float*  b2_t,
    float** _wgrad,
    float** _w,
    int lenw, 
    float lr);


/*
 * dim3 block = dim3(batch, inputDim);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_debug_spiketime(
    int* inputs_time,
    int* outputs_time,
    int* batchPreFireCount,
    int* batchFireCount,
    int inputDim,
    int outputDim,
    int endTime);


void Spiking::calCost()
{
    cost->gpuClear();
    if(predict == NULL){
        printf("Warning::Try to compute the cost when the predict is not properly set!\n ");
        return;
    }
    g_getCost_output<<<dim3(1), dim3(256), sizeof(float) * 256>>>(fireCount->getDev(),
            groundTruth->getDev(),
            cost->getDev(),
            predict,
            batch,
            fireCount->cols,
            UNDESIRED_LEVEL,
            DESIRED_LEVEL,
            MARGIN);
    cudaStreamSynchronize(0);
    getLastCudaError("Spiking:g_getCost_output");
}

void Spiking::feedforward()
{
    if((inputs == NULL))
    {
        printf("Spiking init error\n");
        exit(0);
    }

    int remain = min(1024 / outputDim, outputAmount); //1
    dim3 thread= dim3(1024, remain);

    int div = (outputAmount + remain - 1) / remain;//1
    dim3 block = dim3(batch, div);
    float ** w_lat_dev = (w_laterial.empty()? NULL : w_laterial.m_devPoint);

    ConfigSpiking * config = (ConfigSpiking*) Config::instance()->getLayerByName(m_name); 
    int dummyFreq = config->getBiasFreq();
    g_Spiking_feedforward<<<block, thread>>>(
            inputs->getDev(),
            w.m_devPoint,
            w_lat_dev,
            b.m_devPoint,
            outputs->getDev(),
            fireCount->getDev(),
            inputDim,
            outputDim,
            endTime,
            inputAmount,
            outputAmount,
            threshold,
            dummyFreq,
            T_REFRAC,
            TAU_M,
            TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("Spiking::g_Spiking_feedforward");

    block = dim3(batch, outputAmount);
    thread = dim3(min(outputDim, 1024));

    // transform the binary response matrix to the spike times
    g_response_2_spiketime<<<block, thread>>>(
            outputs->getDev(),
            outputs_time->getDev(),
            outputDim,
            endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("Spiking:g_response_2_spiketime");

}

void Spiking::backpropagation()
{
    // reduce to get the max fire count for each sample in the batch
    int threads = min(1024, outputDim);
    g_getMaxCount<<<dim3(batch), dim3(threads), sizeof(int) * threads>>>(fireCount->getDev(), maxCount->getDev(), fireCount->cols);  
    cudaStreamSynchronize(0);
    getLastCudaError("Spiking::g_getMaxCount");

    if(m_name == std::string("output")){
        // compute the cost function
        g_getCost_output<<<dim3(1), dim3(256), sizeof(float) * 256>>>(fireCount->getDev(), groundTruth->getDev(), cost->getDev(), predict, batch, fireCount->cols, UNDESIRED_LEVEL, DESIRED_LEVEL, MARGIN);
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_getCost_output");

        // compute the delta (error)
        g_getDelta_output<<<dim3(1), dim3(256)>>>(curDelta->getDev(), fireCount->getDev(), groundTruth->getDev(), curDelta->getLen(), MARGIN);
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_getDelta_output");

        // apply the sample weights
        g_boostWeight_output<<<dim3(batch), dim3(outputDim)>>>(curDelta->getDev(), sample_weights, curDelta->getLen());
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_boostWeight_output");

        // compute the lateral factors if applicable
        if(lateralFactor != NULL && !w_laterial.empty()){
            threads = min(outputDim, 1024);
            g_getLateralFactor_output<<<dim3(batch, outputDim), threads, sizeof(float) * threads>>>(
                outputs_time->getDev(),
                fireCount->getDev(),
                lateralW,
                predict,
                lateralFactor->getDev(),
                threshold,
                outputDim,
                endTime,
                T_REFRAC,
                TAU_M,
                TAU_S);
            cudaStreamSynchronize(0);
            getLastCudaError("Spiking::g_getLateralFactor_output");
        }

        // modify the output spikes of the target neuron if it does not fire
        // tricky: modify both the spike trains and output fire counts!
        g_modifySpikes<<<dim3(batch), dim3(min(outputDim, 1024))>>>(outputs->getDev(), predict, fireCount->getDev(), DESIRED_LEVEL, endTime, outputDim);
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_modifySpikes");

        // retransform the binary matrix to the spike times since the outputs might be changed
        g_response_2_spiketime<<<dim3(batch), dim3(min(outputDim, 1024))>>>(
                outputs->getDev(),
                outputs_time->getDev(),
                outputDim,
                endTime);
        checkCudaErrors(cudaStreamSynchronize(0));
        getLastCudaError("Spiking:g_response_2_spiketime");
    }
    // pre compute the accumulative synaptic effect, and effect ratio (if applicable)
    dim3 thread = dim3(min(1024, outputDim));
    dim3 block  = dim3(batch, inputDim);
    cudaFuncSetCacheConfig(g_Spiking_synaptic_effect, cudaFuncCachePreferL1);
    g_Spiking_synaptic_effect<<<block, thread>>>(
        inputs_time->getDev(),
        outputs_time->getDev(),
        preFireCount->getDev(),
        fireCount->getDev(),
        w[0]->getDev(),
        accEffect->getDev(),
        effectRatio == NULL ? NULL : effectRatio->getDev(),
        inputDim,
        outputDim,
        endTime,
        T_REFRAC,
        TAU_M,
        TAU_S);

    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_Spiking_synaptic_effect");

    // compute preDelta: curDelta: batch * outputDim; w: outputDim * inputDim
    assert(w.size() == 1);
    if(preDelta == NULL){
        ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
        assert(config->m_input == "data");
    }
    else{
       if(effectRatio != NULL){
            matrixMul(curDelta, effectRatio, preDelta);
        }
        else{
            matrixMul(curDelta, w[0], preDelta);
        }
    }    
    // need more code to multi-channel input, simply learn: FullConnect.cu
}

/*
 * block = dim3(outputDim, outputAmount);
 * thread= dim3(min(inputDim, 1024));
*/
__global__ void g_Spiking_calSquareSum(
    float** w,
    float* w_sq_sum,
    int outputDim,
    int inputDim,
    float weight_limit)
{
    extern __shared__ float _sum[];
    int ok = blockIdx.y;
    int o_id = blockIdx.x;
    int tid = threadIdx.x;

    _sum[tid] = 0;
    __syncthreads();
    for(int i = 0; i < inputDim; i += blockDim.x)
    {
        int id = i + tid;
        if(id < inputDim)
        { 
            int wid = id + o_id * inputDim;
            float weight = w[ok][wid];
            _sum[tid] += (weight/weight_limit) * (weight/weight_limit);
        }
    }
    __syncthreads();
    int len = blockDim.x;
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
    if(tid == 0)
        w_sq_sum[o_id] = _sum[0] / inputDim;
}

/*
 * block = dim3(outputDim * inputDim, outputAmount);
 * thread= dim3(batch);
*/
__global__ void g_Spiking_gradAdd(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
    float* w_sq_sum,
	int batch,
	float lambda,
    float beta,
    float limit,
    int inputDim,
	int wArea)
{
	extern __shared__ float _sum[];

	int ok  = blockIdx.y;
	int wid = blockIdx.x;
	int tid = threadIdx.x;

	_sum[tid] = 0;
	__syncthreads();
	float* wgradTmp = _WgradTmp[ok];
	for(int i = 0; i < batch; i += blockDim.x)
	{
		int b = i + threadIdx.x;
		if(b < batch)
		{
			_sum[threadIdx.x] += wgradTmp[b * wArea + wid];
		}
	}
	__syncthreads();
	int len = blockDim.x;
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
	if(tid == 0)
	{
        float sq_sum = w_sq_sum[wid / inputDim];
		Wgrad[ok][wid] = _sum[0] / batch + lambda*beta*(w[ok][wid]/limit)*__expf(beta*(sq_sum - 1));
	}
}

void Spiking::getGrad()
{
    dim3 thread = dim3(min(1024, outputDim));
    dim3 block  = dim3(batch, inputDim);
    cudaFuncSetCacheConfig(g_Spiking_wgrad_spiketime,cudaFuncCachePreferL1);

    g_Spiking_wgrad_spiketime<<<block, thread>>>(
        accEffect->getDev(),
        curDelta->getDev(),
        lateralFactor == NULL ? NULL : lateralFactor->getDev(),
        wgradTmp.m_devPoint,
        inputDim,
        outputDim);

    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_wgrad_spiketime");

#ifdef DEBUG
    g_Spiking_debug_spiketime<<<block, thread>>>(inputs_time->getDev(), outputs_time->getDev(), preFireCount->getDev(), fireCount->getDev(), inputDim, outputDim, endTime);
    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_debug_spiketime");
#endif
    
    block = dim3(outputDim);
    thread = dim3(min(inputDim, 1024));

    g_Spiking_calSquareSum<<<block, thread, sizeof(float) * min(inputDim, 1024)>>>(
        w.m_devPoint,
        weightSqSum->getDev(),
        outputDim,
        inputDim,
        weightLimit);
    checkCudaErrors(cudaStreamSynchronize(0));    
	getLastCudaError("g_Spiking_calSquareSum");
 
	block  = dim3(outputDim * inputDim, outputAmount);
	thread = dim3(batch);

	g_Spiking_gradAdd<<<block, thread, sizeof(float) * batch>>>(
		wgradTmp.m_devPoint,
		wgrad.m_devPoint,
		w.m_devPoint,
        weightSqSum->getDev(),
		batch,
		lambda,
        beta,
        weightLimit,
        inputDim,
		w[0]->getArea());

	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_gradAdd");
    
    // add the bias derivation here:
    ConfigSpiking * config = (ConfigSpiking*) Config::instance()->getLayerByName(m_name); 
    if(config->hasBias()){
        thread = dim3(min(1024, outputDim));
        block  = dim3(batch);
 
        int dummyFreq = config->getBiasFreq();
        g_Spiking_bgrad_spiketime<<<block, thread>>>(
            outputs_time->getDev(),
            fireCount->getDev(),
            curDelta->getDev(),
            bgradTmp.m_devPoint,
            outputDim,
            endTime,
            dummyFreq,
            T_REFRAC,
            TAU_M,
            TAU_S);

    	checkCudaErrors(cudaStreamSynchronize(0));
	    getLastCudaError("g_Spiking_bgrad_spiketime");
        
        block  = dim3(outputDim);
        thread = dim3(batch);
 
        g_Spiking_gradAdd<<<block, thread, sizeof(float) * batch>>>(
            bgradTmp.m_devPoint,
            bgrad.m_devPoint,
            b.m_devPoint,
            weightSqSum->getDev(),
            batch,
            0.0f,
            0.0f,
            weightLimit,
            inputDim,
            b[0]->getArea());
    }

}	


/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_Spiking_adam_vecAdd(float** g1_ws, float** g2_ws, float* b1_t, float* b2_t, float** _wgrad, float** _w, int lenw, float lr)
{
    int ok = blockIdx.x;
    float* g1_w  = g1_ws[ok];
    float* g2_w  = g2_ws[ok];
    float* w     = _w[ok];
    float* wgrad = _wgrad[ok];
    int idx = threadIdx.x;
    float b1t = b1_t[0];
    float b2t = b2_t[0];
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float eps = 1.e-8f;
    __syncthreads();

    for(int i = 0; i < lenw; i += blockDim.x * gridDim.x)
    {
        int id = i + idx;
        if(id < lenw)
        {
            float weight_grad = wgrad[id];
            float g1 = b1 * g1_w[id] + (1 - b1) * weight_grad;
            float g2 = b2 * g2_w[id] + (1 - b2) * weight_grad * weight_grad;
            w[id]  -= lr * (g1/(1.f - b1t)) / ((float)sqrtf(g2/(1. - b2t)) + eps);
            g1_w[id] = g1;
            g2_w[id] = g2;
        }
    }
    if(threadIdx.x == 0){
        b1_t[0] = b1t * b1;
        b2_t[0] = b2t * b2;
    }
}

/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_Spiking_sgd_vecAdd(float** momentum_w, float** _wgrad, float** _w, int lenw, float momentum, float lr)
{
    int ok = blockIdx.x;
    float* v_w   = momentum_w[ok];
    float* w     = _w[ok];
    float* wgrad = _wgrad[ok];
    int idx = threadIdx.x;
    for(int i = 0; i < lenw; i += blockDim.x * gridDim.x)
    {
        int id = i + idx;
        if(id < lenw)
        {
            v_w[id] = v_w[id] * momentum + wgrad[id] * lr;
            w[id]  -= v_w[id];
        }
    }
}


void Spiking::updateWeight()
{
	dim3 block  = outputAmount;
	dim3 thread = min(256, w[0]->getLen());
    if(Config::instance()->getOptimizerType() == std::string("adam")){
        g_Spiking_adam_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
            g1_w.m_devPoint,
            g2_w.m_devPoint,
            b1_t->getDev(),
            b2_t->getDev(),
            wgrad.m_devPoint,
            w.m_devPoint,
            w[0]->getLen(),
            Config::instance()->getLrate());
    } 
    else{
        g_Spiking_sgd_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
            momentum_w.m_devPoint,
            wgrad.m_devPoint, 
            w.m_devPoint,
            w[0]->getLen(), 
            Config::instance()->getMomentum(),
            Config::instance()->getLrate());

        ConfigSpiking * config = (ConfigSpiking*) Config::instance()->getLayerByName(m_name); 
        if(config->hasBias()){
            block = outputAmount;
            thread = min(256, b[0]->getLen());
            g_Spiking_sgd_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
                momentum_b.m_devPoint,
                bgrad.m_devPoint,
                b.m_devPoint,
                b[0]->getLen(),
                Config::instance()->getMomentum(),
                Config::instance()->getLrate());
        }
    }
}

/*
 * dim3 block = dim3(batch)
 * dim3 thread = min(1024, outputDim)
 */
__global__ void g_Spiking_delta_vth(int* batchFireCount, float * vth, float * vthDeltaTmp, int outputDim, float DESIRED_LEVEL, float MARGIN)
{
    int batchId = blockIdx.x;
    float * vthDelta = vthDeltaTmp + batchId * outputDim;
    int * fireCount = batchFireCount + batchId * outputDim;
    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputDim)
        {
            if(fireCount[o_idx] == 0 && vth[o_idx] > 5.0f)
                vthDelta[o_idx] = 0.0001 * outputDim;
            else if(fireCount[o_idx] > DESIRED_LEVEL + MARGIN && vth[o_idx] < 25.0f)
                vthDelta[o_idx] = -0.0001 * outputDim;
        }
    }
}

/*
 * dim3 block = dim3(outputDim)
 * dim3 thread = min(batch)
 */
__global__ void g_Spiking_delta_vthAdd(float* vthDeltaTmp, float* vthDelta, int batch, int vthArea)
{
    extern __shared__ float _sum[];
        
    int vid = blockIdx.x;
    int tid = threadIdx.x;

    _sum[tid] = 0;
    __syncthreads();
    for(int i = 0; i < batch; i += blockDim.x)
    {
        int b = i + tid;
        if(b < batch)
        {
            _sum[tid] += vthDeltaTmp[b * vthArea + vid];
        }
    }
    __syncthreads();
    int len = blockDim.x;
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
    if(tid == 0)
        vthDelta[vid] = _sum[0] / batch;
}


void Spiking::getDeltaVth()
{
    dim3 thread = dim3(min(1024, outputDim));
    dim3 block = dim3(batch);
    cudaFuncSetCacheConfig(g_Spiking_delta_vth, cudaFuncCachePreferL1);
    g_Spiking_delta_vth<<<block, thread>>>(
        fireCount->getDev(),
        vth->getDev(),
        vthDeltaTmp->getDev(),
        outputDim,
        DESIRED_LEVEL,
        MARGIN);
    
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_Spiking_delta_vth");
    
    block  = dim3(outputDim);
	thread = dim3(batch);
   
    g_Spiking_delta_vthAdd<<<block, thread, sizeof(float) * batch>>>(
        vthDeltaTmp->getDev(),
        vthDelta->getDev(),
        batch,
        vth->getArea());

    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_Spiking_delta_vthAdd");

}

/*
 * dim3 block = dim3(1)
 * dim3 thread = min(256, outputDim)
 */
__global__ void g_Spiking_vth_vecAdd(float * vthDelta, float * vth, int lenVth)
{
    for(int i = 0; i < lenVth; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < lenVth)
        {
            vth[id] -= vthDelta[id];
        }
    }
}

void Spiking::updateVth()
{
    dim3 block = 1;
    dim3 thread = min(256, outputDim);
    g_Spiking_vth_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
        vthDelta->getDev(),
        vth->getDev(),
        vth->getLen());
}

__global__ void g_weight_cp_test(float ** ws, int nrows, int ncols)
{
    int weight_size = nrows * ncols;
    float * w = ws[0];
    for(int i = 0; i < weight_size; i += blockDim.x)
    {
        int idx = i + threadIdx.x;
        if(idx < weight_size && fabsf(w[idx]) > 1e-5)
        {
            printf("Accessing row: %d , col: %d,  the weight is %f \n", idx /ncols, idx % ncols, w[idx]);
        }
    }
}

Spiking::Spiking(std::string name)
{
	m_name = name;
	ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
	SpikingLayerBase * preLayer = (SpikingLayerBase*)Layers::instance()->get(config->m_input);

	inputs = preLayer->getSpikingOutputs();
    inputs_time = preLayer->getSpikingTimeOutputs();
	preDelta = preLayer->getCurDelta();
    preFireCount = preLayer->getFireCount();
	
    inputAmount  = preLayer->outputAmount;
    outputAmount = inputAmount;

	inputDim  = preLayer->outputDim;
	outputDim = config->m_numNeurons;
    endTime   = Config::instance()->getEndTime(); 
	batch     = Config::instance()->getBatchSize();
	lambda    = Config::instance()->getLambda();
    beta      = Config::instance()->getBeta();
    T_REFRAC  = config->m_t_ref;
    TAU_M     = config->m_tau_m;
    TAU_S     = config->m_tau_s;    

    weightLimit = Config::instance()->getWeightLimit();

    UNDESIRED_LEVEL = config->m_undesired_level;
    DESIRED_LEVEL   = config->m_desired_level;
    MARGIN          = config->m_margin; 

	outputs  = new cuMatrix<bool>(batch, outputDim * endTime, outputAmount);
    outputs_time = new cuMatrix<int>(batch, outputDim * endTime, outputAmount);

	curDelta = new cuMatrix<float>(batch, outputDim, outputAmount);
    fireCount= new cuMatrix<int>(batch, outputDim, outputAmount);
    weightSqSum = new cuMatrix<float>(outputDim, 1, 1);
    maxCount    = new cuMatrix<int>(batch, 1, 1);
    accEffect   = new cuMatrix<float>(batch, outputDim * inputDim, 1); 

    predict = NULL;

    // only for the output
    if(config->m_name == std::string("output")){
        groundTruth   = new cuMatrix<float>(batch, outputDim, 1);
        cost          = new cuMatrix<float>(1, 1, 1);
    }
    else{
        groundTruth   = NULL;
        cost          = NULL;
    }
    assert(outputDim > 0 && inputDim > 0);

	for(int i = 0; i < outputAmount; i++){
		w.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
		b.push_back(new cuMatrix<float>(outputDim, 1, 1));
		wgrad.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
		bgrad.push_back(new cuMatrix<float>(outputDim, 1, 1));
		wgradTmp.push_back(new cuMatrix<float>(batch, outputDim * inputDim, 1));
        bgradTmp.push_back(new cuMatrix<float>(batch, outputDim, 1));
        
        if(config->hasLaterialWeight() == true){
            w_laterial.push_back(new cuMatrix<float>(outputDim, outputDim, 1));
        }
	}
    
    threshold = config->m_vth;
    vth = new cuMatrix<float>(outputDim, 1, 1);
    vthDelta = new cuMatrix<float>(outputDim, 1, 1);
    vthDeltaTmp = new cuMatrix<float>(batch, outputDim, 1);

	w.toGpu();
	b.toGpu();
    b[0]->toGpu();
	wgrad.toGpu();
	bgrad.toGpu();
	wgradTmp.toGpu();
    bgradTmp.toGpu();

    if(config->hasLaterialWeight() == true){
        w_laterial.toGpu();
    }
   
    // lateral inihibition factor for the output
    lateralFactor = NULL;
    lateralW = 0.0f;
    if(config->hasLaterialInh() == true && config->m_name == std::string("output")){
        lateralFactor = new cuMatrix<float>(batch, outputDim, 1);
        lateralW = config->m_localInbStrength;
    }

    // use the e^k_{i|j} / o^{k-1}_j for estimating the grad of effect w.r.t to fire count
    // notice that this variable is w[i][j] * e^k_{i|j} / o^{k-1}_j ! 
    effectRatio = NULL;
    if(Config::instance()->useEffectRatio()){
        if(batch > 1){
            printf("Must set batch size to 1 if use effect ratio for grad of synaptic effect.\n");
            printf("Current batch size: %d\n", batch);
            assert(batch <= 1);
        }
        effectRatio = new cuMatrix<float>(outputDim, inputDim, 1);
    }

	for(int i = 0; i < outputAmount; i++){
		momentum_w.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
		momentum_b.push_back(new cuMatrix<float>(outputDim, 1, 1));
        g1_w.push_back(new cuMatrix<float>(outputDim, inputDim, 1)); // for adam
        g1_b.push_back(new cuMatrix<float>(outputDim, 1, 1));
        g2_w.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
        g2_b.push_back(new cuMatrix<float>(outputDim, 1, 1));       
	}
	momentum_w.toGpu();
	momentum_b.toGpu();
    g1_w.toGpu();
    g1_b.toGpu();
    g2_w.toGpu();
    g2_b.toGpu();

    b1_t = new cuMatrix<float>(1, 1, 1);
    b2_t = new cuMatrix<float>(1, 1, 1);
    for(int i = 0; i < b1_t->getLen(); i++){
        b1_t->getHost()[i] = 0.9f;
        b2_t->getHost()[i] = 0.999f;
    }
    b1_t->toGpu();
    b2_t->toGpu();

	this->initRandom();

    if(Config::instance()->getIsGradientChecking())
        this->loadRef(); // for verification purpose

    Layers::instance()->set(m_name, this);
}

void Spiking::save(FILE* file)
{
    for(int a = 0; a < (int)w.size(); a++){

        w[a]->toCpu();
        b[a]->toCpu();

        for(int c = 0; c < w[a]->channels; c++){
            for(int i = 0; i < w[a]->rows; i++){
                for(int j = 0; j < w[a]->cols; j++){
                    fprintf(file, "%f ", w[a]->get(i, j, c));
                }
            }
        }
        if(!w_laterial.empty()){
            for(int c = 0; c < w_laterial[a]->channels; c++){
                for(int i = 0; i < w_laterial[a]->rows; i++){
                    for(int j = 0; j < w_laterial[a]->cols; j++){
                        fprintf(file, "%f ", w_laterial[a]->get(i, j, c));
                    }
                }
            } 
        }

        for(int c = 0; c < b[a]->channels; c++){
            for(int i = 0; i < b[a]->rows; i++){
                for(int j = 0; j < b[a]->cols; j++){
                    fprintf(file, "%f ", b[a]->get(i, j, c));
                }
            }
        }
    }
}

void Spiking::clearMomentum()
{
	for(int i = 0; i < (int)momentum_b.size(); i++){
		momentum_b[i]->gpuClear();
	}
	for(int i = 0; i < (int)momentum_w.size(); i++){
		momentum_w[i]->gpuClear();
	}
}

void Spiking::verify(const std::string& phrase)
{
    printf("Verify for the layer: %s at %s phrase.\n", m_name.c_str(), phrase.c_str());
    if(phrase == std::string("train"))
    {
        if(!output_train_ref.empty()){
            outputs->toCpu();
            checkMatrixIsSame(output_train_ref[0], outputs, outputDim);
        }
        
    }
    else if(phrase == std::string("test"))
    {
        if(!w_ref.empty()){
            w[0]->toCpu();
            checkMatrixIsSame(w_ref[0], w[0]);
        }
        if(!w_laterial_ref.empty() && !w_laterial.empty()){
            w_laterial[0]->toCpu();
            checkMatrixIsSame(w_laterial_ref[0], w_laterial[0]);
        }
 
        if(!b_ref.empty()){
            b[0]->toCpu();
            checkMatrixIsSame(b_ref[0], b[0]);
        }
    
        if(!output_test_ref.empty()){
            outputs->toCpu();
            checkMatrixIsSame(output_test_ref[0], outputs, outputDim);
        }
    }
    printf("Verification for the layer: %s at %s phrase. Pased!!\n", m_name.c_str(), phrase.c_str());
}

//* load the reference weights and output spikes for verification
void Spiking::loadRef()
{
    if(batch != 1){
        printf("Only do the verification for one batch and one sample!\n");
        exit(0);
    }
    ConfigSpiking * config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
    if(config->m_ref_weight_path != std::string("NULL")){
        w_ref.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
        initFromDumpfile(config->m_ref_weight_path, w_ref);
        if(config->hasBias()){
            b_ref.push_back(new cuMatrix<float>(outputDim, 1, 1));
            initBiasFromDumpfile(config->m_ref_weight_path, b_ref);
        }
    }

    if(config->m_ref_lweight_path != std::string("NULL")){
        w_laterial_ref.push_back(new cuMatrix<float>(outputDim, outputDim, 1));
        initFromDumpfile(config->m_ref_lweight_path, w_laterial_ref);
    }

    if(config->m_ref_output_train_path != std::string("NULL")){
        read_each_speech_dump(config->m_ref_output_train_path, output_train_ref, endTime, outputDim);
        assert(output_train_ref.size() == 1 && output_train_ref[0] != NULL);
        output_train_ref[0]->rows = 1;
        output_train_ref[0]->cols = endTime * outputDim;
    }

    if(config->m_ref_output_test_path != std::string("NULL")){
        read_each_speech_dump(config->m_ref_output_test_path, output_test_ref, endTime, outputDim);
        assert(output_test_ref.size() == 1 && output_test_ref[0] != NULL);
        output_test_ref[0]->rows = 1;
        output_test_ref[0]->cols = endTime * outputDim;
   }

}

void Spiking::initRandom()
{
    //srand(clock());
    ConfigSpiking * config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
    float initW = config->m_initW;

    //  	for(int i = 0; i < w.size(); i++){
    //  		initMatrix(w[i], initW);
    //  	}

    if(config->isGaussian()){
        for(int i = 0; i < (int)w.size(); i++){
            float epsilon = initW;
            for(int c = 0; c < w[i]->channels; c++)
            {
                float r1 = 0.5f + 4.0f * (rand()) / RAND_MAX;
                float r2 = 0.5f + 4.0f * (rand()) / RAND_MAX;
                createGaussian(w[i]->getHost() + c * w[i]->getArea(), r1,r2,
                        outputDim, inputDim, w[i]->channels,
                        epsilon);
            }
            w[i]->toGpu();
        }
    }
    else if(config->isBernoulli()){
        for(int i = 0; i < (int)w.size(); i++){
            for(int j = 0; j < w[i]->getLen(); j++){
                w[i]->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
                //printf("%f ", w[i]->hostData[j]);
            }//printf("\n");
            w[i]->toGpu();
        }
    }
    else if(config->isFixed()){
        // one input connects to nconnect randomly selected outputs, with initW/-initW
        int nconnect = config->m_weightConnect;
        assert(nconnect > 0);
        for(int a = 0; a < w.size(); a++){
            for(int c = 0; c < w[a]->channels; ++c){
                for(int i = 0; i < w[a]->rows; ++i){
                    for(int t = 0; t < nconnect; ++t){
                        int j = rand() % inputDim;
                        if(rand() % 2 == 0)
                            w[a]->set(i, j, c, initW);
                        else
                            w[a]->set(i, j, c, -1.0*initW);
                        //printf("input_%d to reservoir_%d : %f\n", j, i, w[a]->get(i, j, c));
                    }
                }
            }
            w[a]->toGpu();
        }
    }
    else if(config->isExternal()){
        initFromDumpfile(config->m_weightPath, w);
    }
    if(config->hasLaterialWeight()){
        initLaterial();
    }

    // initialize vth
    float thres = config->m_vth;
    for(int i = 0; i < outputDim; ++i){
        vth->set(i, 0, 0, thres);
    }
    vth->toGpu();
}

void Spiking::initFromCheckpoint(FILE* file)
{
    float val = 0;
    for(int a = 0; a < (int)w.size(); a++){
        for(int c = 0; c < w[a]->channels; c++){
            for(int i = 0; i < w[a]->rows; i++){
                for(int j = 0; j < w[a]->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
                        char logStr[256];
                        sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                        LOG(logStr, "Result/log.txt");
                        assert(0);
                    }
                    w[a]->set(i, j, c, val);
                }
            }
        }

        if(!w_laterial.empty()){
            for(int c = 0; c < w_laterial[a]->channels; c++){
                for(int i = 0; i < w_laterial[a]->rows; i++){
                    for(int j = 0; j < w_laterial[a]->cols; j++){
                        if(fscanf(file, "%f", &val) == EOF)
                        {
                            char logStr[256];
                            sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                            LOG(logStr, "Result/log.txt");
                        }
                        w_laterial[a]->set(i, j, c, val);
                    }
                }
            } 
        }

        for(int c = 0; c < b[a]->channels; c++){
            for(int i = 0; i < b[a]->rows; i++){
                for(int j = 0; j < b[a]->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
                        char logStr[256];
                        sprintf(logStr, "scanf fail for layer: %s\n", m_name.c_str());
                        LOG(logStr, "Result/log.txt");
                        assert(0);
                    }
                    b[a]->set(i, j, c, val);
                }
            }
        }

        w[a]->toGpu();
        b[a]->toGpu();
    }
}

//* initial the weights from the dumped file by the CPU sim
void Spiking::initFromDumpfile(const std::string& filename, cuMatrixVector<float>& cuW)
{
    ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        printf("Cannot open the file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
 
    assert(cuW.size() == 1);
    std::vector<std::vector<float> > weights(cuW[0]->rows, std::vector<float>(cuW[0]->cols, 0.0f));
   
    int idx; 
    float weight;
    std::string pre_name, post_name;
    while(f_in>>idx>>pre_name>>post_name>>weight){
        int pre = extractNeuronIndex(pre_name);
        int post = extractNeuronIndex(post_name);
        if(post >= weights.size() || pre >= weights[0].size()){
            if(pre == weights[0].size() && post < weights.size()){ // this is related to bias    
                continue;
            }
            else{
                printf("Read the file: %s, in line: %d\n", filename.c_str(), idx);
                printf("Post: %d, OutputDim: %d\n Pre: %d, InputDim: %d\n", post, (int)weights.size(), pre, (int)weights[0].size());
                assert(post < weights.size() && pre < weights[0].size());
            }
        }
        weights[post][pre] += weight;
    }

	for(int a = 0; a < (int)cuW.size(); a++){
		for(int c = 0; c < cuW[a]->channels; c++){
			for(int i = 0; i < cuW[a]->rows; i++){
				for(int j = 0; j < cuW[a]->cols; j++){
					cuW[a]->set(i, j, c, weights[i][j]);
				}
			}
		}
		cuW[a]->toGpu();
    }
    // verify that the weights is correctly copied!
    for(int i = 0; i < weights.size(); ++i){
        for(int j = 0; j < weights[0].size(); ++j){
            assert(fabsf(cuW[0]->get(i, j, 0) - weights[i][j]) < 1e-4);
        }
    }
}

//* initial the bias weights from the dumped file by the CPU sim
void Spiking::initBiasFromDumpfile(const std::string& filename, cuMatrixVector<float>& cuW)
{
    ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        printf("Cannot open the file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    assert(cuW.size() == 1);

    int idx; 
    float weight;
    std::string pre_name, post_name;
    while(f_in>>idx>>pre_name>>post_name>>weight){
        int pre = extractNeuronIndex(pre_name);
        int post = extractNeuronIndex(post_name);
        if(pre == inputDim && post < outputDim){ // this is related to bias
            cuW[0]->set(post, 0, 0, weight); 
        }
    }
    cuW[0]->toGpu();
}

void Spiking::initLaterial()
{
    ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
    if(config->m_laterialType == "RESERVOIR"){
        initFromDumpfile(config->m_lweightPath, w_laterial);
        //initReservoirConnection(config->m_reservoirDim);
    }
    else if(config->m_laterialType == "LOCAL_INHIBITION"){
        initLocalInhibition(config->m_localInbStrength); 
    }
}

// intialize the reservoir connections
// TODO: improve the randomness of the reservoir (the bad random seed we used now!)
void Spiking::initReservoirConnection(const std::vector<int>& reservoirDim)
{
    assert(reservoirDim.size() == 3);
    assert(w_laterial.size() == 1);
    int d1 = reservoirDim[0], d2 = reservoirDim[1], d3 = reservoirDim[2];
    int num = d1 * d2 * d3;
    if(num != outputDim){
        printf("The reservoir dim: %d x %d x %d = %d does not match the number neuron: %d!\n",d1, d2, d3, num, outputDim);
        exit(EXIT_FAILURE);
    }
    // adopted from the CPU code:
    srand(5);
    std::vector<bool> excitatory(num, false);
    std::vector<dim3> coordinates;
    for(int i = 0; i < excitatory.size(); ++i){
        if(rand() % 100 < 20) excitatory[i] = false;
        else    excitatory[i] = true;
    }
    for(int i = 0; i < d1; ++i){
        for(int j = 0; j < d2; ++j){
            for(int k = 0; k < d3; ++k){
                int index = (i * d2 + j) * d3 + k;
                assert(index < excitatory.size());
                coordinates.push_back(dim3(i, j, k));
            }
        }
    }
    double c, a;
    double distsq, dist;
    const double factor2 = 1.5;
    for(int i = 0; i < num; ++i){
        for(int j = 0; j < num; ++j){
            if(excitatory[i]){
                if(excitatory[j]){
                    c = 0.3 * factor2;
                    a = 1;
                }
                else{
                    c = 0.2 * factor2;
                    a = 1;
                }
            }
            else{
                if(excitatory[j]){
                    c = 0.4 * factor2;
                    a = -1;
                }
                else{
                    c = 0.1 * factor2;
                    a = -1;
                }
            }
            distsq = 0;
            dist = coordinates[i].x -  coordinates[j].x;
            distsq += dist * dist;
            dist = coordinates[i].y -  coordinates[j].y;
            distsq += dist * dist;
            dist = coordinates[i].z -  coordinates[j].z;
            distsq += dist * dist;
            if(rand() % 100000 < 100000 * c * exp(-distsq / 4)){
                //printf("reservoir_%d to reservoir_%d %f\n", i , j, a);
                w_laterial[0]->set(j, i, 0, a);
            }
        }
    }
    w_laterial[0]->toGpu();
}

void Spiking::initLocalInhibition(float strength)
{
    for(int a = 0; a < (int)w_laterial.size(); a++){
		for(int c = 0; c < w_laterial[a]->channels; c++){
			for(int i = 0; i < w_laterial[a]->rows; i++){
				for(int j = 0; j < w_laterial[a]->cols; j++){
                    if(i == j)  continue;
					w_laterial[a]->set(i, j, c, -1*strength);
				}
			}
		}
		w_laterial[a]->toGpu();
    }
}

/* the device function to realize: weights * spikes(:, t - 1) + recurrent_weights * o_spikes(t - 1)
 * I only consider the first order dynamics 
 * inputDim  : number of input neurons
 * outputDim : number of output neurons
*/
__device__ float d_Spiking_accumulate_spikes(
    int inputDim,
    int outputDim,
    bool* input,
    bool* output,
    int o_idx,
    float* weights,
    float* weights_lat,
    float* biases,
    int t,
    int dummyFreq)
{
    int idx = threadIdx.x;
    if(idx >= outputDim * inputDim){
        return 0;
    }  
    float response = 0.0f;
    // effect from the forward-connects
    for(int i = 0; i < inputDim; ++i){
        response += input[i + (t - 1) * inputDim] ? weights[i + o_idx * inputDim] : 0; 
    }
    // effect from the bias
    if(t % dummyFreq == 0){
        response += biases[idx];
    }    

    if(weights_lat != NULL){
        // effect from the recurrent connections:
        for(int i = 0; i < outputDim; ++i)
            response += output[i + (t - 1) * outputDim] ? weights_lat[i + o_idx * outputDim] : 0;
    }
    return response;
}


/* given each input and output spike train, 
 * compute the accumulative synaptic effect as the gradient
 * input: input spikes: endTime * inputDim
 * output: output spikes: endTime * outputDim
 */
__device__ float d_Spiking_gradient(
    bool* output,
    bool* input,
    float delta,
    int o_idx,
    int i_idx,
    int outputDim,
    int inputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    float acc_response = 0.0f;
    int t_post_last = 1;
    for(int t_post = 1; t_post < endTime; t_post++){
        if(output[o_idx + t_post * outputDim] != true) continue;
        float sum = 0.0f;

        int ub = t_post;
        int lb = max(1, int(t_post - 4*TAU_M));
        for(int t_pre = lb; t_pre < ub; ++t_pre){
            if(input[i_idx + t_pre * inputDim] != true)    continue;
            int pre_time = t_pre + T_REFRAC;
            if(pre_time > t_post)   continue;
            int s = t_post - t_post_last;
            int t = t_post - pre_time;
            float factor = exp(-1*max(t - s, 0)/TAU_S)/(1 - TAU_S/TAU_M);
            sum += factor * (exp(-1*min(s, t)/TAU_M) - exp(-1*min(s, t)/TAU_S));
        }
        t_post_last = t_post + T_REFRAC;
        acc_response += sum;
    }
    float delta_w = delta * acc_response;
    return delta_w;
 
}

/* given each input and output spike train of spike times, 
 * compute the accumulative synaptic effect
 * input: input spikes: endTime * inputDim
 * output: output spikes: endTime * outputDim
 */
__device__ float d_Spiking_accumulate_effect(
    int* output_time,
    int* input_time,
    int n_ospikes,
    int n_ispikes,
    int o_idx,
    int i_idx,
    int outputDim,
    int inputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    float acc_response = 0.0f;
    int t_post_last = 1;
    for(int i = 0; i < n_ospikes; ++i){
        int t_post = output_time[o_idx * endTime + i];
        float sum = 0.0f;
        
        int ub = t_post;
        int lb = max(1, int(t_post - 4*TAU_M));
        for(int j = 0; j < n_ispikes; ++j){
            int t_pre = input_time[i_idx * endTime + j];
            if(t_pre < lb || t_pre >= ub)    continue;

            int pre_time = t_pre + T_REFRAC;
            if(pre_time > t_post)   continue;
            int s = t_post - t_post_last;
            int t = t_post - pre_time;
            float factor = exp(-1*max(t - s, 0)/TAU_S)/(1 - TAU_S/TAU_M);
            sum += factor * (exp(-1*min(s, t)/TAU_M) - exp(-1*min(s, t)/TAU_S));
        }
        t_post_last = t_post + T_REFRAC;
        acc_response += sum;
    }
    if(n_ospikes == 0 && n_ispikes != 0)
        acc_response = 0.1;
    return acc_response;
}

/* given each input and output spike train of spike times, 
 * compute the accumulative synaptic effect as the gradient
 * input: input spikes: endTime * inputDim
 * output: output spikes: endTime * outputDim
 */
__device__ float d_Spiking_gradient_spiketime(
    int* output_time,
    int* input_time,
    int n_ospikes,
    int n_ispikes,
    float delta,
    int o_idx,
    int i_idx,
    float lat_factor,
    int outputDim,
    int inputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    float acc_response = d_Spiking_accumulate_effect(output_time, input_time, n_ospikes, n_ispikes, o_idx, i_idx, outputDim, inputDim, endTime, T_REFRAC, TAU_M, TAU_S);
    float delta_w = delta * acc_response * lat_factor;
    return delta_w;
 
}


/* compute the gradient for the bias
 * input: input spikes: endTime * inputDim
 * output: output spikes: endTime * outputDim
 */
__device__ float d_Spiking_bias_gradient_spiketime(
    int* output_time,
    int n_ospikes,
    float delta,
    int o_idx,
    int dummyFreq,
    int outputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    float acc_response = 0.0f;
    int t_post_last = 1;
    for(int i = 0; i < n_ospikes; ++i){
        int t_post = output_time[o_idx * endTime + i];
        float sum = 0.0f;
        
        int ub = t_post;
        int lb = max(1, int(t_post - 4*TAU_M));
        for(int j = dummyFreq; j < endTime; j += dummyFreq){
            int t_pre = j;
            if(t_pre < lb || t_pre >= ub)    continue;

            int pre_time = t_pre + T_REFRAC;
            if(pre_time > t_post)   continue;
            int s = t_post - t_post_last;
            int t = t_post - pre_time;
            float factor = exp(-1*max(t - s, 0)/TAU_S)/(1 - TAU_S/TAU_M);
            sum += factor * (exp(-1*min(s, t)/TAU_M) - exp(-1*min(s, t)/TAU_S));
        }
        t_post_last = t_post + T_REFRAC;
        acc_response += sum;
    }
    float delta_b = delta * acc_response;
    return delta_b;
 
}


/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_getCost_output(
    int*   fireCount, 
    float* groundTruth, 
    float* cost, 
    int*   y, 
    int    batch, 
    int    cols,
    float  UNDESIRED_LEVEL,
    float  DESIRED_LEVEL,
    float  MARGIN)
{
    extern __shared__ float _sum[];
    int len = batch * cols;
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len){
            groundTruth[id] =  UNDESIRED_LEVEL;
        }
    }
    __syncthreads();
    for(int i = 0; i < batch; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < batch){
            int yy = y[id];
            groundTruth[id * cols + yy] = DESIRED_LEVEL;
        }
    }
    _sum[threadIdx.x] = 0;
    __syncthreads();
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len)
        {
            float diff = fabsf(float(fireCount[id]) - groundTruth[id]);
            _sum[threadIdx.x] += diff > MARGIN ? diff * diff : 0; 
        }
    }
    len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1)>>1;
        if(threadIdx.x < skip && (threadIdx.x + skip) < len)
        {
            _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = skip;
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        cost[0] = _sum[0];
    }
}

 
/*
 * dim3 block = dim3(1);
 * dim3 thread= dim3(256);
 */
__global__ void g_getDelta_output(float* outputDelta, int* fireCount, float* groundTruth, int len, float MARGIN)
{
    for(int i = 0; i < len; i += blockDim.x)
    {
        int id = i + threadIdx.x;
        if(id < len)
        {
            float diff = fabsf(float(fireCount[id]) - groundTruth[id]);
            outputDelta[id] = diff > MARGIN ? fireCount[id] - groundTruth[id] : 0;
        }
    }
}

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(outputDim);
 */
__global__ void g_boostWeight_output(float* outputDelta, float* sample_weights, int len)
{
    int batchId = blockIdx.x;
    float sample_weight = sample_weights[batchId];
    int outputDim = blockDim.x;
    int tid = threadIdx.x;
    int target = tid + batchId * outputDim;
    if(target < len)
        outputDelta[target] *= sample_weight;

}


/*
 * dim3 block = dim3(batch, outputDim);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_getLateralFactor_output(
    int* outputs_time,
    int* batchFireCount,
    float w0,
    int* y,
    float* batchLFactor,
    float vth,
    int outputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    extern __shared__ float d_sum[];
    int tid = threadIdx.x;
    d_sum[tid] = 0;
    __syncthreads();

    int batchId = blockIdx.x;
    int j_idx   = blockIdx.y;
    
    int outputSize2 = endTime * outputDim;
    int* output_time = outputs_time + batchId * outputSize2;
    int* output_fireCount = batchFireCount + batchId * outputDim;
    int cls = y[batchId];

    float * lateral_factors = batchLFactor + batchId * outputDim;

    int f_cnt_j = output_fireCount[j_idx];
    float d_j = (f_cnt_j > 0 || (f_cnt_j == 0 && j_idx == cls)) ? 1 / vth : 0;

    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int l_idx = i + tid;
        if(l_idx < outputDim && j_idx != l_idx)
        {
            int f_cnt_l = output_fireCount[l_idx];
            float d_l = (f_cnt_l > 0 || (f_cnt_l == 0 && l_idx == cls)) ? 1 / vth : 0;
            // j --> l
            float e_jl = d_Spiking_accumulate_effect(output_time, output_time, f_cnt_l, f_cnt_j, l_idx, j_idx, outputDim, outputDim, endTime, T_REFRAC, TAU_M, TAU_S);
            float effect_ratio_jl = (f_cnt_j == 0 || f_cnt_l == 0) ? 1 : e_jl / f_cnt_j;
            
            // l --> j
            float e_lj = d_Spiking_accumulate_effect(output_time, output_time, f_cnt_j, f_cnt_l, j_idx, l_idx, outputDim, outputDim, endTime, T_REFRAC, TAU_M, TAU_S);
            float effect_ratio_lj = (f_cnt_l == 0 || f_cnt_j == 0) ? 1 : e_lj / f_cnt_l;

            d_sum[tid] += effect_ratio_jl * d_l * effect_ratio_lj * d_j; 
        }
    }
    int len = blockDim.x;  
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len)
        {
            d_sum[tid] += d_sum[tid + skip];
        }
        len = skip;
    }
    if(tid == 0)
    {
        lateral_factors[j_idx] = 1.0f / (1 - d_sum[0] * w0 * w0);
    }
}
 

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(outputDim);
 */
__global__ void g_getMaxCount(int* fireCount, int* maxCount, int cols)
{
    extern __shared__ int _max[];
    int batchId = blockIdx.x;
    int len = blockDim.x;
    int id = threadIdx.x;
    
    _max[id] = 0;
    __syncthreads();
    
    for(int tid = 0; tid < cols; tid += blockDim.x){
        int ttid = tid + id;
        if(ttid < cols){
            _max[threadIdx.x] = max(_max[threadIdx.x], fireCount[ttid + batchId * cols]);
        }
    }

    _max[id] = fireCount[id + batchId * cols];
    while(len != 1)
    { 
        __syncthreads();
        int skip = (len + 1)>>1;
        if(id < skip && (id + skip) < len)
        {
            _max[id] = max(_max[id], _max[id + skip]);
        }
        len = skip;
    }
    __syncthreads();
    if(id == 0)
    {
        maxCount[batchId] = _max[0];
    } 
}


/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(1024, outputDim));
 */
__global__ void g_modifySpikes(bool* outputs, int* y, int* fireCount, int target_level, int endTime, int outputDim)
{
    int batchId = blockIdx.x;
    int target = y == NULL ? -1 : y[batchId];
    int mCnt = target_level; //maxCount[batchId]; 
    bool* outputSpikes = outputs + batchId * endTime * outputDim;
    for(int id = 0; id < outputDim; id += blockDim.x){
        int o_idx = id + threadIdx.x;
        if(o_idx < outputDim)
        {
            if(fireCount[o_idx + batchId * outputDim] == 0)
            {
                int count = 0;
                int interval = (mCnt == 0 || target != o_idx) ? endTime - 1 : endTime / mCnt;
                for(int t = interval; t < endTime; t += interval)
                {
                    outputSpikes[o_idx + t * outputDim] = true;
                    count++;
                }
                fireCount[o_idx + batchId * outputDim] = count;
            }
        }
    }
}


/*
 * dim3 block = dim3(batch, div);
 * dim3 thread= dim3(min(outputDim, 1024), min(1024/ outputDim, outputAmount)));
 */
__global__ void g_Spiking_feedforward(
	bool*  inputs,
	float** ws,
	float** ws_lat,
    float** bs,
	bool*  outputs,
    int* fireCount,
	int inputDim,
	int outputDim,
    int endTime,
	int inputAmount,
	int outputAmount,
    float vth,
    int dummyFreq, 
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
	int batchId = blockIdx.x;
	int ok = blockIdx.y * blockDim.y + threadIdx.y;
	if(ok >= outputAmount) return;

    int outputSize2 = endTime * outputDim;
	int inputSize2  = endTime* inputDim;
    int outputArea  = endTime * outputDim;
    int inputArea   = endTime * inputDim;

	bool* curOutput    = outputs + ok * outputArea + batchId * outputSize2 * outputAmount;
    bool* curInput     = inputs + ok * inputArea + batchId * inputSize2 * outputAmount;
    int* curFireCount = fireCount + ok * outputDim + batchId * outputDim * outputAmount; 
    float* w = ws[ok];
    float* w_l = ws_lat == NULL ? NULL : ws_lat[ok];
    float* b = bs[ok];

    // simulate the spiking train
    for(int tidx = 0; tidx < outputDim; tidx += blockDim.x)
    {
        int o_idx = tidx + threadIdx.x;
        if(o_idx < outputDim)
        {
            float v  = 0.0f;
            float ep = 0.0f;
            float threshold = vth;
            int t_ref= 0;
            float response = 0.0f;
            int fire_count = 0;
            for(int t = 0; t < endTime; t++){
                // 1. leakage
                v  -= v / TAU_M;
                ep -= ep / TAU_S;
                if(t == 0)
                {
                    curOutput[o_idx + t * outputDim] = false;
                    continue;
                }

                // 2. receive the spike inputs
                __syncthreads(); // make sure all the threads has generated the spikes for the last time step
                response = d_Spiking_accumulate_spikes(inputDim, outputDim, curInput, curOutput, o_idx, w, w_l, b, t, dummyFreq);
                
                // 3. Add up the response to ep (state variable)
                ep += response;

                // 4. Update the vmem accordingly
                v += ep/TAU_S;
                if(t_ref > 0){
                    v = 0;
                    t_ref--;
                }
            
                // 5. Fire or not
                curOutput[o_idx + t * outputDim] = v > threshold ?  true : false;
                t_ref = v > threshold ? T_REFRAC : t_ref;
                fire_count += v > threshold ? 1 : 0;
                v = v > threshold ? 0 : v;
            }
            curFireCount[o_idx] = fire_count; 
        }
    }
}


/*
 * dim3 block = dim3(batch, inputDim, outputAmount);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_wgrad(
        bool* inputs,
        bool* outputs,
        float* curDelta,
        float** wgradTmp,
        int inputDim,
        int outputDim,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;
    int ok      = blockIdx.z;

    int wSize        = outputDim * inputDim;
    int inputSize2   = endTime * inputDim;
    int outputSize2  = endTime * outputDim;
    int curDeltaSize = outputDim;

    float* wgrad  = wgradTmp[ok] + batchId * wSize;
    bool* input    = inputs + batchId * inputSize2;
    bool* output   = outputs + batchId * outputSize2;
    float* cDelta = curDelta + batchId * curDeltaSize;
    
    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputDim)
        {
            float delta_w = d_Spiking_gradient(output, input, cDelta[o_idx], o_idx, i_idx, outputDim, inputDim, endTime, T_REFRAC, TAU_M, TAU_S);
            wgrad[i_idx + o_idx * inputDim] = delta_w;
        }
    }

}

/*
 * dim3 block = dim3(batch, inputDim, outputAmount);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_wgrad_spiketime(
        float* batchAccEffect,
        float* curDelta,
        float* latFactor,
        float** wgradTmp,
        int inputDim,
        int outputDim)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;
    int ok      = blockIdx.z;

    int wSize        = outputDim * inputDim;
    int curDeltaSize = outputDim;

    float* wgrad  = wgradTmp[ok] + batchId * wSize;
    float* acc_effect     = batchAccEffect + batchId * wSize;
    float* cDelta = curDelta + batchId * curDeltaSize;
    float* lFactor = latFactor == NULL ? NULL : latFactor + batchId * curDeltaSize;

    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputDim)
        {
            float latFac = lFactor == NULL ? 1.0f : lFactor[o_idx];
            float delta_w = cDelta[o_idx] * acc_effect[i_idx + o_idx * inputDim] * latFac;

            wgrad[i_idx + o_idx * inputDim] = delta_w;
        }
    }
}

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(1024, outputDim));
 */
__global__ void g_Spiking_bgrad_spiketime(
        int* outputs_time,
        int* batchFireCount,
        float* curDelta,
        float** bgradTmp,
        int outputDim,
        int endTime,
        int dummyFreq,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int ok      = 0;

    int bSize = outputDim;
    int outputSize2  = endTime * outputDim;
    int curDeltaSize = outputDim;

    float* bgrad  = bgradTmp[ok] + batchId * bSize;
    int* output_time      = outputs_time + batchId * outputSize2;
    int* output_fireCount = batchFireCount + batchId * outputDim;
    float* cDelta = curDelta + batchId * curDeltaSize;
    
    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputDim)
        {
            float delta_b = d_Spiking_bias_gradient_spiketime(output_time, output_fireCount[o_idx], cDelta[o_idx], o_idx, dummyFreq, outputDim, endTime, T_REFRAC, TAU_M, TAU_S);
            bgrad[o_idx] = delta_b;
        }
    }

}


/*
 * dim3 block = dim3(batch, inputDim);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_synaptic_effect(
        int* inputs_time,
        int* outputs_time,
        int* batchPreFireCount,
        int* batchFireCount,
        float* w,
        float* batchAccEffect,
        float* effectRatio,
        int inputDim,
        int outputDim,
        int endTime,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;

    int wSize        = outputDim * inputDim;
    int inputSize2   = endTime * inputDim;
    int outputSize2  = endTime * outputDim;

    int* input_time       = inputs_time + batchId * inputSize2;
    int* output_time      = outputs_time + batchId * outputSize2;
    int* input_fireCount  = batchPreFireCount + batchId * outputDim;
    int* output_fireCount = batchFireCount + batchId * outputDim;
    float* acc_effect     = batchAccEffect + batchId * wSize;

    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputDim)
        {
            float e = d_Spiking_accumulate_effect(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, outputDim, inputDim, endTime, T_REFRAC, TAU_M, TAU_S);
            acc_effect[i_idx + o_idx * inputDim] = e;
            if(effectRatio != NULL){
                int o_cnt = output_fireCount[o_idx];
                int i_cnt = input_fireCount[i_idx];
                float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : e / float(i_cnt);
                effectRatio[i_idx + o_idx * inputDim] = ratio * w[i_idx + o_idx * inputDim];
            }
        }
    }
}


/*
 * dim3 block = dim3(batch, inputDim, outputAmount);
 * dim3 thread= min(1024, outputDim);
 */
__global__ void g_Spiking_debug_spiketime(
        int* inputs_time,
        int* outputs_time,
        int* batchPreFireCount,
        int* batchFireCount,
        int inputDim,
        int outputDim,
        int endTime)
{
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;

    int inputSize2   = endTime * inputDim;
    int outputSize2  = endTime * outputDim;
 
    int* input_time       = inputs_time + batchId * inputSize2;
    int* output_time      = outputs_time + batchId * outputSize2;
    int* input_fireCount  = batchPreFireCount + batchId * outputDim;
    int* output_fireCount = batchFireCount + batchId * outputDim;

    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputDim)
        {
            if(i_idx == 154 && o_idx == 56){
                printf("%d fires: ", i_idx);
                for(int i = 0; i < input_fireCount[i_idx]; i++)    printf("%d\t", input_time[i_idx * endTime + i]);
                printf("\n");
                printf("%d fires: ", o_idx);
                for(int j = 0; j < output_fireCount[o_idx]; j++)    printf("%d\t", output_time[o_idx * endTime + j]);
                printf("\n");
            }
        }
    } 
}
