#include "PoolingSpiking.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"
#include "../common/util.h"

__global__ void g_PoolingSpiking_feedforward(
	bool* conv,
	bool* pool,
    int*  fireCount,
	int convDim,
	int poolDim,
    int endTime,
	int poolingSkip,
	int poolingSize,
	int convArea,
	int poolArea,
	int kAmount,
    float vth,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);

/*
 * function: upper pooling with psize == pskip
 * blocks : dim3(min(256, (poolDeltalen + threadx) / threadx)) 
 * threads: dim3(threadx);
 */
__global__ void g_PoolingSpiking_backpropagation_no_atomic(float* _pool, float* _conv,
	int poolDim, int convDim, int poolingSkip, int poolingSize, int poolDeltalen);


/*
 * function: upper pooling with psize != pskip
 * blocks : dim3(min(256, (poolDeltalen + threadx) / threadx)) 
 * threads: dim3(threadx);
 */
__global__ void g_PoolingSpiking_backpropagation(float* _pool, float* _conv,
	int poolDim, int convDim, int poolingSkip, int poolingSize, int poolDeltalen);


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


void PoolingSpiking::feedforward()
{
	int threadx = min(outputDim * outputDim , 512);
	if(threadx <= 16) threadx = 16;
	else if(threadx <= 256) threadx = 64;
	int remain = 512 / threadx;
	int div = (outputAmount + remain - 1) / remain;

	dim3 block = dim3(batch, div);     // remain * div ~~= outputAmount / remain
	dim3 thread= dim3(threadx, remain);

    g_PoolingSpiking_feedforward<<<block, thread>>>(
        inputs->getDev(),
        outputs->getDev(),
        fireCount->getDev(),
        inputDim,
        outputDim,
        endTime,
        pskip,
        psize,
        inputs->getArea(),
        outputs->getArea(),
        outputAmount,
        threshold,
        T_REFRAC,
        TAU_M,
        TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("PoolSpiking::g_PoolingSpiking_feedforward");

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
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("PoolingSpiking:g_response_2_spiketime");

}

void PoolingSpiking::backpropagation()
{
    int curDeltalen = curDelta->getLen();
    int threadx = outputDim * outputDim;
    threadx = 1024 / threadx * threadx;
    dim3 block = dim3(std::min(256, (curDeltalen + threadx) / threadx));
    dim3 thread= dim3(threadx);
    if(psize == pskip){
        /*no need to clear preDelta*/
        g_PoolingSpiking_backpropagation_no_atomic<<<block, thread>>>(
            curDelta->getDev(), preDelta->getDev(), 
            outputDim, inputDim, pskip, psize, curDeltalen);
        checkCudaErrors(cudaStreamSynchronize(0));
        getLastCudaError("PoolingSpiking::g_PoolingSpiking_backpropagation_no_atomic");
    }else{
        preDelta->gpuClear();
        g_PoolingSpiking_backpropagation<<<block, thread>>>(curDelta->getDev(), preDelta->getDev(), 
            outputDim, inputDim, pskip, psize, curDeltalen);
        checkCudaErrors(cudaStreamSynchronize(0));
        getLastCudaError("PoolingSpiking::g_PoolingSpiking_backpropagation");
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
    preDelta = preLayer->getCurDelta();

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

	curDelta = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
    fireCount= new cuMatrix<int>(batch, outputDim * outputDim, outputAmount);

    output_train_ref = NULL;
    output_test_ref = NULL;
    if(Config::instance()->getIsGradientChecking())
        this->loadRef(); // for verification purpose

	Layers::instance()->set(m_name, this);
}


/*
 * blocks : dim3(batch, outputAmount / remain), remain can be 1, 2, 16, or 64
 * threads: dim3(min(outputDim * outputDim, 512), remain);
 */
__global__ void g_PoolingSpiking_feedforward(
	bool* conv,
	bool* pool,
    int*  fireCount,
	int convDim,
	int poolDim,
    int endTime,
	int poolingSkip,
	int poolingSize,
	int convArea,
	int poolArea,
	int kAmount,
    float vth,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
	int batchId = blockIdx.x;
	int k  = blockIdx.y * blockDim.y + threadIdx.y;
	if(k >= kAmount)return;

	int convSize2  = convDim * convDim;
	int poolSize2  = poolDim * poolDim;

	bool* curConv = conv   + convArea * k + batchId * convSize2 * endTime;
	bool* curPool = pool   + poolArea * k + batchId * poolSize2 * endTime;
    int* curFireCount = fireCount + k * poolArea / endTime + batchId * poolSize2;

	/*pooling*/
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

            float v  = 0.0f;
            float ep = 0.0f;
            float threshold = vth - 1e-6;
            int t_ref= 0;
            float response = 0.0f;
            int fire_count = 0;

            for(int t = 0; t < endTime; t++){
                v  -= v / TAU_M;
                ep -= ep / TAU_S;
                if(t == 0){
                    curPool[o_idx + t * poolSize2] = false;
                    continue;
                }
                response = 0.0f;
                for(int i = curX; i < lenx; i++){
                    for(int j = curY; j < leny; j++){
                        int i_idx = i * convDim + j;
                        float val = curConv[i_idx + (t - 1) * convSize2];
                        response += val;
                    }
                }

                ep += response;
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
 * blocks : dim3(min(256, (poolDeltalen + threadx) / threadx)) 
 * threads: dim3(threadx);
 */
__global__ void g_PoolingSpiking_backpropagation(float* _pool, float* _conv,
	int poolDim, int convDim, int pSkip, int pSize, int poolDeltalen)
{
	int poolSize2 = poolDim * poolDim;
	int convSize2 = convDim * convDim;
	for(int i = 0; i < poolDeltalen; i += gridDim.x * blockDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < poolDeltalen)
		{
			int convId = id / poolSize2;
			int idx    = id % poolSize2;

			float* pool = _pool   + poolSize2 * convId;
			float* conv = _conv   + convSize2 * convId;

			int x = idx / poolDim;
			int y = idx % poolDim;

			int curX = x * pSkip;
			int curY = y * pSkip;

			int lenx = min(convDim, curX + pSize);
			int leny = min(convDim, curY + pSize);

			float val = pool[idx] / (pSize * pSize);
			for(int i = curX; i < lenx; i++)
			{
				for(int j = curY; j < leny; j++)
				{
					cuAssert(i < convDim && j < convDim);
					atomicAdd(conv + i * convDim + j, val);
				}
			}
		}
	}
}


/*
 * function: upper pooling with psize == pskip
 * blocks : dim3(min(256, (poolDeltalen + threadx) / threadx)) 
 * threads: dim3(threadx);
 */
__global__ void g_PoolingSpiking_backpropagation_no_atomic(float* _pool, float* _conv,
	int poolDim, int convDim, int pSkip, int pSize, int poolDeltalen)
{
	int poolSize2 = poolDim * poolDim;
	int convSize2 = convDim * convDim;
	for(int i = 0; i < poolDeltalen; i += gridDim.x * blockDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < poolDeltalen)
		{
			int convId = id / poolSize2;
			int idx    = id % poolSize2;

			float* pool = _pool   + poolSize2 * convId;
			float* conv = _conv   + convSize2 * convId;

			int x = idx / poolDim;
			int y = idx % poolDim;

			int curX = x * pSkip;
			int curY = y * pSkip;

			int lenx = min(convDim, curX + pSize);
			int leny = min(convDim, curY + pSize);

			float val = pool[idx] / (pSize * pSize);
			for(int i = curX; i < lenx; i++)
			{
				for(int j = curY; j < leny; j++)
				{
					assert(i < convDim && j < convDim);
					conv[i * convDim + j] = val;
				}
			}
		}
	}
}
