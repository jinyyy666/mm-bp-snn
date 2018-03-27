#include "PoolingSpiking.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"
#include "../common/util.h"

__global__ void g_PoolingSpiking_fast_input_response(
	bool* conv,
    float* inputs_resp,
	int convDim,
	int poolDim,
    int endTime,
	int poolingSkip,
	int poolingSize,
    int convArea,
	int poolArea,
	int kAmount);


__global__ void g_PoolingSpiking_feedforward(
	float* inputs_resp,
	bool* pool,
    int*  fireCount,
	int poolDim,
    int endTime,
	int poolArea,
	int kAmount,
    float vth,
    int T_REFRAC,
    float* tau,
    float* res,
    float TAU_S);

/*
 * function: upper pooling with psize == pskip
 * blocks : dim3(batch, outputAmount, outputDim * outputDim)
 * threads: dim3(psize * psize);
 */
__global__ void g_PoolingSpiking_backpropagation_no_atomic(
    int* _inputs_time,
    int* _outputs_time,
    int* batchPreFireCount,
    int* batchFireCount,
    float* _pool,
    float* _conv,
	int poolDim,
    int convDim,
    int endTime,
    int convArea,
    int poolArea,
    int pSkip,
    int pSize,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

/*
 * function: upper pooling with psize != pskip
 * blocks : dim3(batch, outputAmount, outputDim * outputDim)
 * threads: dim3(psize * psize);
 */
__global__ void g_PoolingSpiking_backpropagation(
    int* _inputs_time,
    int* _outputs_time,
    int* batchPreFireCount,
    int* batchFireCount,
    float* _pool,
    float* _conv,
	int poolDim,
    int convDim,
    int endTime,
    int convArea,
    int poolArea,
    int pSkip,
    int pSize,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);


void PoolingSpiking::loadRef()
{
    if(batch != 1){
        printf("Only do the verification for one batch and one sample!\n");
        exit(0);
    }
    ConfigPoolingSpiking * config = (ConfigPoolingSpiking*)Config::instance()->getLayerByName(m_name);
    if(config->m_ref_output_train_path != std::string("NULL")){
        output_train_ref = new cuMatrix<bool>(1, endTime * outputDim * outputDim, outputAmount);
        readSpikesFromDumpfile(config->m_ref_output_train_path, output_train_ref);
    }
    if(config->m_ref_output_test_path != std::string("NULL")){
        output_test_ref = new cuMatrix<bool>(1, endTime * outputDim * outputDim, outputAmount);
        readSpikesFromDumpfile(config->m_ref_output_test_path, output_test_ref);
    }
}

void PoolingSpiking::verify(const std::string& phrase)
{
    printf("Verify for the layer: %s at %s phrase.\n", m_name.c_str(), phrase.c_str());
    if(phrase == std::string("train"))
    {
        if(output_train_ref != NULL){
            outputs->toCpu();
            checkMatrixIsSame(output_train_ref, outputs, outputDim*outputDim);
        } 
    }
    else if(phrase == std::string("test"))
    {
        if(output_test_ref != NULL){
            outputs->toCpu();
            checkMatrixIsSame(output_test_ref, outputs, outputDim*outputDim);
        }
    }
    printf("Verification for the layer: %s at %s phrase. Pased!!\n", m_name.c_str(), phrase.c_str());
}

void PoolingSpiking::intrinsicPlasticity()
{
    int outputSize2 = outputDim * outputDim;
    dim3 thread = dim3(min(1024, outputSize2));
    dim3 block  = dim3(batch, outputAmount);
    float u = 0.2;
    g_intrinsic_plasticity<<<block, thread>>>(fireCount->getDev(), taugradTmp->getDev(), resgradTmp->getDev(), tau->getDev(), res->getDev(), endTime, tau->getArea(), outputSize2, T_REFRAC, threshold, u);
    //checkCudaErrors(cudaStreamSynchronize(0));
    //getLastCudaError("ConvSpiking::g_intrinsic_plasticity");

    thread = batch;
    g_intrinsic_plasticity_gradadd<<<dim3(outputSize2, outputAmount), thread, 2 * sizeof(float) * batch>>>(taugradTmp->getDev(), taugrad->getDev(), resgradTmp->getDev(), resgrad->getDev(), batch, taugradTmp->getArea(), outputSize2);
    //checkCudaErrors(cudaStreamSynchronize(0));
    //getLastCudaError("ConvSpiking::g_intrinsic_plasticity_add");


    block = min((tau->getLen() + 255)/ 256, 5120);
    thread = 256;
    g_intrinsic_plasticity_update<<<block, thread, 0, Layers::instance()->get_stream()>>>(
        taugrad->getDev(),
        resgrad->getDev(),
        tau->getDev(),
        res->getDev(),
        tau->getLen(),
        0.5);
    //checkCudaErrors(cudaStreamSynchronize(0));
    //getLastCudaError("ConvSpiking::g_intrinsic_plasticity");
}

void PoolingSpiking::feedforward()
{
	int threadx = min(outputDim * outputDim , 512);
	if(threadx <= 16) threadx = 16;
	else if(threadx <= 256) threadx = 64;
	int remain = 512 / threadx;
	int div = (outputAmount + remain - 1) / remain;

    // fast input response
    g_PoolingSpiking_fast_input_response<<<dim3(batch, div, endTime), dim3(threadx, remain)>>>(
        inputs->getDev(),
        inputs_resp->getDev(),
        inputDim,
        outputDim,
        endTime,
        pskip,
        psize,
        inputs->getArea(),
        inputs_resp->getArea(),
        outputAmount);
    //checkCudaErrors(cudaStreamSynchronize(0));
    //getLastCudaError("PoolSpiking::g_PoolingSpiking_fast_input_response");

	dim3 block = dim3(batch, div);     // remain * div ~~= outputAmount / remain
	dim3 thread= dim3(threadx, remain);

    g_PoolingSpiking_feedforward<<<block, thread>>>(
        inputs_resp->getDev(),
        outputs->getDev(),
        fireCount->getDev(),
        outputDim,
        endTime,
        outputs->getArea(),
        outputAmount,
        threshold,
        T_REFRAC,
        tau->getDev(),
        res->getDev(),
        TAU_S);
    //checkCudaErrors(cudaStreamSynchronize(0));
    //getLastCudaError("PoolSpiking::g_PoolingSpiking_feedforward");

    int outputDim2 = outputDim * outputDim;
    block = dim3(batch, outputAmount);
    thread = dim3(min(outputDim2, 1024));

    // transform the binary response matrix to the spike times
    g_response_2_spiketime<<<block, thread>>>(
            outputs->getDev(),
            outputs_time->getDev(),
            outputs->getArea(),
            outputDim2,
            endTime);
    //checkCudaErrors(cudaStreamSynchronize(0));
    //getLastCudaError("PoolingSpiking:g_response_2_spiketime");

}

void PoolingSpiking::backpropagation()
{
    dim3 block = dim3(batch, outputAmount, outputDim * outputDim);
    dim3 thread= dim3(psize * psize);
    if(psize == pskip){
        /*no need to clear preDelta*/
        g_PoolingSpiking_backpropagation_no_atomic<<<block, thread>>>(
            inputs_time->getDev(),
            outputs_time->getDev(),
            preFireCount->getDev(),
            fireCount->getDev(),
            curDelta->getDev(),
            preDelta->getDev(),
            outputDim,
            inputDim,
            endTime,
            inputs->getArea(),
            outputs->getArea(),
            pskip,
            psize,
            T_REFRAC,
            TAU_M,
            TAU_S);
        //checkCudaErrors(cudaStreamSynchronize(0));
        //getLastCudaError("PoolingSpiking::g_PoolingSpiking_backpropagation_no_atomic");
    }else{
        preDelta->gpuClear();
        g_PoolingSpiking_backpropagation<<<block, thread>>>(
            inputs_time->getDev(),
            outputs_time->getDev(),
            preFireCount->getDev(),
            fireCount->getDev(),
            curDelta->getDev(),
            preDelta->getDev(),
            outputDim,
            inputDim,
            endTime,
            inputs->getArea(),
            outputs->getArea(),
            pskip,
            psize,
            T_REFRAC,
            TAU_M,
            TAU_S);
        //checkCudaErrors(cudaStreamSynchronize(0));
        //getLastCudaError("PoolingSpiking::g_PoolingSpiking_backpropagation");
    }
	
}


PoolingSpiking::PoolingSpiking(std::string name)
{	
	cost = NULL;
	m_name = name;
	ConfigPoolingSpiking* config = (ConfigPoolingSpiking*)Config::instance()->getLayerByName(m_name);
	SpikingLayerBase * preLayer = (SpikingLayerBase*)Layers::instance()->get(config->m_input);

	psize = config->m_size;
	pskip = config->m_skip;

	inputs = preLayer->getSpikingOutputs();
    inputs_time = preLayer->getSpikingTimeOutputs();
    preDelta = preLayer->getCurDelta();
    preFireCount = preLayer->getFireCount();

	inputDim = preLayer->outputDim;
	outputDim = (inputDim + pskip - 1) / pskip;
	inputAmount = preLayer->outputAmount;
	outputAmount = inputAmount;
	
	batch= Config::instance()->getBatchSize();
    endTime   = Config::instance()->getEndTime();
    T_REFRAC  = config->m_t_ref;
    TAU_M     = config->m_tau_m;
    TAU_S     = config->m_tau_s;    
    threshold = config->m_vth;

	outputs       = new cuMatrix<bool>(batch, endTime * outputDim * outputDim, outputAmount);
	outputs_time  = new cuMatrix<int>(batch, outputDim * outputDim * endTime, outputAmount);
    inputs_resp   = new cuMatrix<float>(batch, endTime * outputDim * outputDim, outputAmount);

	curDelta = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
    fireCount= new cuMatrix<int>(batch, outputDim * outputDim, outputAmount);

    tau        = new cuMatrix<float>(1,  outputDim * outputDim, outputAmount);
    res        = new cuMatrix<float>(1,  outputDim * outputDim, outputAmount);
    taugrad    = new cuMatrix<float>(1,  outputDim * outputDim, outputAmount);
    resgrad    = new cuMatrix<float>(1,  outputDim * outputDim, outputAmount);
    taugradTmp     = new cuMatrix<float>(batch,  outputDim * outputDim, outputAmount);
    resgradTmp     = new cuMatrix<float>(batch,  outputDim * outputDim, outputAmount); 
    for(int i = 0; i < tau->getLen(); i++){
        tau->getHost()[i] = TAU_M;
        res->getHost()[i] = TAU_M;
    }
    tau->toGpu();
    res->toGpu();

    output_train_ref = NULL;
    output_test_ref = NULL;
    if(Config::instance()->getIsGradientChecking())
        this->loadRef(); // for verification purpose

	Layers::instance()->set(m_name, this);
}

void PoolingSpiking::saveTauRes(FILE* file)
{
    tau->toCpu();
    int len = tau->getLen();
    fprintf(file, "The tau in %s: ", m_name.c_str());
    for(int c = 0; c < tau->channels; ++c)
        for(int i = 0; i < tau->rows; ++i)
            for(int j = 0; j < tau->cols; ++j)
                fprintf(file, "%f ", tau->get(i, j, c));

    fprintf(file, "\n");

    res->toCpu();
    len = res->getLen();
    fprintf(file, "The res in %s: ", m_name.c_str());
    for(int c = 0; c < res->channels; ++c)
        for(int i = 0; i < res->rows; ++i)
            for(int j = 0; j < res->cols; ++j)
                fprintf(file, "%f ", res->get(i, j, c));

    fprintf(file, "\n");
}

/*
 * blocks : dim3(batch, outputAmount / remain, endTime), remain can be 1, 2, 16, or 64
 * threads: dim3(min(outputDim * outputDim, 1024), remain);
 */
__global__ void g_PoolingSpiking_fast_input_response(
	bool* conv,
    float* inputs_resp,
	int convDim,
	int poolDim,
    int endTime,
	int poolingSkip,
	int poolingSize,
    int convArea,
	int poolArea,
	int kAmount)
{
	int batchId = blockIdx.x;
	int k  = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z;
	if(k >= kAmount)return;
    int convSize2  = convDim * convDim;
    int poolSize2  = poolDim * poolDim;

    bool* curConv = conv   + convArea * k + batchId * convSize2 * endTime;
    float* curResponse = inputs_resp + poolArea * k + batchId * poolSize2 * endTime;
   
	for(int tidx = 0; tidx < poolSize2; tidx += blockDim.x)
	{
		int o_idx = tidx + threadIdx.x;
		if(o_idx < poolSize2)
		{
			int x = o_idx / poolDim;
			int y = o_idx % poolDim;

			int curX = x * poolingSkip;
			int curY = y * poolingSkip;
			cuAssert(curX < convDim && curY < convDim);

			int lenx = min(convDim, curX + poolingSize);
			int leny = min(convDim, curY + poolingSize);
            float response = 0.0f;
            for(int i = curX; i < lenx; i++){
                for(int j = curY; j < leny; j++){
                    int i_idx = i * convDim + j;
                    float val = curConv[i_idx + t * convSize2];
                    response += val;
                }
            }
            curResponse[o_idx + t * poolSize2] = response;
        }
    }

}


/*
 * blocks : dim3(batch, outputAmount / remain), remain can be 1, 2, 16, or 64
 * threads: dim3(min(outputDim * outputDim, 512), remain);
 */
__global__ void g_PoolingSpiking_feedforward(
	float* inputs_resp,
	bool* pool,
    int*  fireCount,
	int poolDim,
    int endTime,
	int poolArea,
	int kAmount,
    float vth,
    int T_REFRAC,
    float* tau,
    float* res,
    float TAU_S)
{
	int batchId = blockIdx.x;
	int k  = blockIdx.y * blockDim.y + threadIdx.y;
	if(k >= kAmount)return;

	int poolSize2  = poolDim * poolDim;

	float* curResp = inputs_resp  + poolArea * k + batchId * poolSize2 * endTime;
	bool* curPool = pool   + poolArea * k + batchId * poolSize2 * endTime;
    int* curFireCount = fireCount + k * poolArea / endTime + batchId * poolSize2;

	/*pooling*/
	for(int tidx = 0; tidx < poolSize2; tidx += blockDim.x)
	{
		int o_idx = tidx + threadIdx.x;
		if(o_idx < poolSize2)
		{
            float v  = 0.0f;
            float ep = 0.0f;
            float threshold = vth - 1e-6;
            int t_ref= 0;
            float response = 0.0f;
            int fire_count = 0;
            float TAU_M = tau[k * poolSize2 + o_idx];
            float r = res[k * poolSize2 + o_idx];

            for(int t = 0; t < endTime; t++){
                v  -= v / TAU_M;
                ep -= ep / TAU_S;
                if(t == 0){
                    curPool[o_idx + t * poolSize2] = false;
                    continue;
                }
                response = curResp[o_idx + (t - 1) * poolSize2];

                ep += response * r / TAU_M;
                v += ep/TAU_S;  
                if(t_ref > 0){
                    v = 0;
                    t_ref--;
                }

                curPool[o_idx + t * poolSize2] = v > threshold ?  true : false;
                t_ref = v > threshold ? T_REFRAC : t_ref;
                fire_count += v > threshold ? 1 : 0;
                v = v > threshold ? 0 : v;
            }
            curFireCount[o_idx] = fire_count;
        }
	}
}


/*
 * function: upper pooling with psize != pskip
 * blocks : dim3(batch, outputAmount, outputDim * outputDim) 
 * threads: dim3(psize * psize);
 */
__global__ void g_PoolingSpiking_backpropagation(
    int* _inputs_time,
    int* _outputs_time,
    int* batchPreFireCount,
    int* batchFireCount,
    float* _pool,
    float* _conv,
	int poolDim,
    int convDim,
    int endTime,
    int convArea,
    int poolArea,
    int pSkip,
    int pSize,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    int batchId = blockIdx.x;
    int ok = blockIdx.y;
    int o_idx = blockIdx.z;
    int ik = ok;

	int poolSize2 = poolDim * poolDim;
	int convSize2 = convDim * convDim;
    int preDeltaArea = convArea / endTime;
    int curDeltaArea = poolArea / endTime;

    int* input_time = _inputs_time + convArea * ik + batchId * convSize2 * endTime;
    int* output_time = _outputs_time + poolArea * ok + batchId * poolSize2 * endTime;
    int* input_fireCount = batchPreFireCount + ik * convArea / endTime + batchId * convSize2;
    int* output_fireCount = batchFireCount + ok * poolArea / endTime + batchId * poolSize2;
    float* conv = _conv + ok * preDeltaArea + batchId * convSize2;
    float* pool = _pool + ok * curDeltaArea + batchId * poolSize2;

    int x = o_idx / poolDim;
    int y = o_idx % poolDim;

    int curX = x * pSkip;
    int curY = y * pSkip;

    int i = curX + threadIdx.x / pSize;
    int j = curY + threadIdx.x % pSize;

    float val = pool[o_idx] / (pSize * pSize);
    if(i < convDim && j < convDim){
        int i_idx = i * convDim + j;
        float e = d_Spiking_accumulate_effect(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, poolSize2, convSize2, endTime, T_REFRAC, TAU_M, TAU_S);
        int o_cnt = output_fireCount[o_idx];
        int i_cnt = input_fireCount[i_idx];
        float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : e / float(i_cnt);
		atomicAdd(conv + i * convDim + j, val * ratio);
	}
}


/*
 * function: upper pooling with psize == pskip
 * blocks : dim3(batch, outputAmount, outputDim * outputDim) 
 * threads: dim3(psize * psize);
 */
__global__ void g_PoolingSpiking_backpropagation_no_atomic(
    int* _inputs_time,
    int* _outputs_time,
    int* batchPreFireCount,
    int* batchFireCount,
    float* _pool,
    float* _conv,
	int poolDim,
    int convDim,
    int endTime,
    int convArea,
    int poolArea,
    int pSkip,
    int pSize,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    int batchId = blockIdx.x;
    int ok = blockIdx.y;
    int o_idx = blockIdx.z;
    int ik = ok;

	int poolSize2 = poolDim * poolDim;
	int convSize2 = convDim * convDim;
    int preDeltaArea = convArea / endTime;
    int curDeltaArea = poolArea / endTime;

    int* input_time = _inputs_time + convArea * ik + batchId * convSize2 * endTime;
    int* output_time = _outputs_time + poolArea * ok + batchId * poolSize2 * endTime;
    int* input_fireCount = batchPreFireCount + ik * convArea / endTime + batchId * convSize2;
    int* output_fireCount = batchFireCount + ok * poolArea / endTime + batchId * poolSize2;
    float* conv = _conv + ok * preDeltaArea + batchId * convSize2;
    float* pool = _pool + ok * curDeltaArea + batchId * poolSize2;

    int x = o_idx / poolDim;
    int y = o_idx % poolDim;

    int curX = x * pSkip;
    int curY = y * pSkip;

    int i = curX + threadIdx.x / pSize;
    int j = curY + threadIdx.x % pSize;

    float val = pool[o_idx] / (pSize * pSize);
    if(i < convDim && j < convDim){
        int i_idx = i * convDim + j;
        float e = d_Spiking_accumulate_effect(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, poolSize2, convSize2, endTime, T_REFRAC, TAU_M, TAU_S);
        int o_cnt = output_fireCount[o_idx];
        int i_cnt = input_fireCount[i_idx];
        float ratio = i_cnt == 0 || o_cnt == 0 ? 1 : e / float(i_cnt);
		conv[i * convDim + j] = val * ratio;
    }
}
