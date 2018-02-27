#include "ConvSpiking.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../common/util.h"

//#define DEBUG
#define ROW 4
#define COL 2
#define IN_CH 0
#define OUT_CH 4

/*
 * blocks  : dim3(batch, div, endTime);
 * threads :  dim3(min(outputDim * outputDim, 1024), remain));
 */
__global__ void g_ConvSpiking_fast_input_response(
        bool*  inputs,
        float** ws,
        float** bs,
        float*  inputs_resp,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int endTime,
        int inputAmount,
        int outputAmount,
        int inputArea,
        int responseArea);


/*
*	blocks : dim3(batch, outputAmount),
*	threads: dim3(min(outputDim * outputDim, 1024));
*/
__global__ void g_ConvSpiking_feedforward(
        float*  inputs_resp,
        bool*  outputs,
        int* fireCount,
        int outputDim,
        int endTime,
        int outputAmount,
        int outputArea,
        float vth,
        int T_REFRAC,
        float TAU_M,
        float TAU_S);

/*
 * dim3 block = dim3(batch, inputAmount);
 * dim3 thread= dim3(min(inputDim * inputDim, 1024), 1);
 */
__global__ void g_ConvSpiking_backpropagation(
        float* _curDelta,
        float** ws,
        float* _preDelta,
        int     curDim,
        int     preDim,
        int     preAmount,
        int     curAmount,
        int     kernelSize,
        int     padding,
        int     curArea,
        int     preArea);

/*
 * dim3 block = dim3(batch, outputAmount*inputAmount, kernelSize * kernelSize);
 * dim3 thread= min(outputDim * outputDim, 512);
 */
__global__ void g_ConvSpiking_wgrad(
        int*    _inputs_time,
        int*    _outputs_time,
        int*    batchPreFireCount,
        int*    batchFireCount,
        float*  _curDelta,
        float** wgradTmp,
        int     inputDim,
        int     curDeltaDim,
        int     endTime,
        int     kernelSize,
        int     padding,
        int     inputAmount,
        int     inputArea,
        int     outputArea,
        int     curDeltaArea,
        int     wgradTmpArea,
        int     T_REFRAC,
        float   TAU_M,
        float   TAU_S);

/*
*blocks  : dim3(kernelAmount2)
*threads : dim3(256)
*shared  : sizeof(float) * 256
*/
__global__ void g_ConvSpiking_Bgrad(float* delta,
	float** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea);


void ConvSpiking::calCost()
{
	cost->gpuClear();
	g_getCost_3<<<dim3(w.size()), dim3(32), sizeof(float) * 32>>>(cost->getDev(), 
		w.m_devPoint, 
		lambda,
		w[0]->getLen());
	cudaStreamSynchronize(0);
	getLastCudaError("ConvSpiking:getCost");
}

void ConvSpiking::feedforward()
{
	if((inputs == NULL))
	{
		printf("ConvSpiking init error\n");
		assert(0);
	}

    int outputDim2 = outputDim * outputDim;
    int remain = min(1024 / outputDim2, outputAmount); //1
    int div = (outputAmount + remain - 1) / remain;//32

    // fast input response: compute the convolved spikes for each time step
    g_ConvSpiking_fast_input_response<<<dim3(batch, div, endTime), dim3(min(outputDim2, 1024), remain)>>>(
        inputs->getDev(), 
        w.m_devPoint,
        b.m_devPoint,
        inputs_resp->getDev(),
        inputDim,
        kernelSize,
        padding,
        outputDim,
        endTime,
        inputAmount,
        outputAmount,
        inputs->getArea(),
        inputs_resp->getArea());
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("ConvSpiking::g_ConvSpiking_fast_input_response");

    dim3 thread= dim3(min(outputDim2, 1024), remain);
    dim3 block = dim3(batch, div);
    
    g_ConvSpiking_feedforward<<<block, thread>>>(
        inputs_resp->getDev(),
        outputs->getDev(),
        fireCount->getDev(),
        outputDim,
        endTime,
        outputAmount,
        outputs->getArea(),
        threshold,
        T_REFRAC,
        TAU_M,
        TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("ConvSpiking::g_ConvSpiking_feedforward");

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
    getLastCudaError("ConvSpiking:g_response_2_spiketime");

}

void ConvSpiking::backpropagation()
{	
    g_divide_by_threshold<<<dim3(batch, outputAmount), dim3(min(1024, outputDim*outputDim))>>>(curDelta->getDev(), curDelta->getArea(), curDelta->cols, threshold);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_divide_by_threshold");
    
	if(Config::instance()->getLayerByName(m_name)->m_input == std::string("data"))
		return;

    int inputDim2 = inputDim * inputDim;
    int remain = min(1024 / inputDim2, inputAmount); // 1

    int div = (inputAmount + remain - 1) / remain; // inputAmount
    dim3 block = dim3(batch, div);
    dim3 thread= dim3(min(inputDim2, 1024), remain);
   
    g_ConvSpiking_backpropagation<<<block, thread>>>(
        curDelta->getDev(),
        w.m_devPoint,
        preDelta->getDev(),
        outputDim,
        inputDim,
        inputAmount,
        outputAmount,
        kernelSize,
        padding,
        curDelta->getArea(),
        preDelta->getArea());
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("ConvSpiking::g_ConvSpiking_backpropagation");
}

/*
 * block = dim3(outputAmount, kernelSize * kernelSize * inputAmount);
 * thread= dim3(batch);
*/
__global__ void g_ConvSpiking_wgradAdd(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
    float*  w_sq_sum,
	int kernelSize,
	int batch,
	float lambda,
    float beta,
    float wlimit,
	int wgradTmpArea,
	int wArea)
{
	extern __shared__ float _sum[];
	int ok = blockIdx.x;
	int kernelSize2 = kernelSize * kernelSize;
    int wgradArea = wArea;
	int kid= blockIdx.y % kernelSize2;
    int c = blockIdx.y / kernelSize2;
	int tid = threadIdx.x;

	float* wgradTmp = _WgradTmp[ok];
    int skip = c * wgradTmpArea + kid;

	_sum[tid] = 0;
	__syncthreads();
	for(int i = 0; i < batch; i += blockDim.x)
	{
		int b = i + threadIdx.x;
		if(b < batch)
		{
			_sum[threadIdx.x] += wgradTmp[b * kernelSize2 + skip];
		}
	}
	__syncthreads();
	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < skip && (tid + skip < len))
		{
			_sum[tid] += _sum[tid + skip];
		}
		len = skip;
	}
	if(tid == 0)
	{
        float sq_sum = w_sq_sum[ok];
		Wgrad[ok][kid + c * wgradArea] = _sum[0] / batch + lambda*beta*(w[ok][kid + c * wArea]/wlimit)*__expf(beta*(sq_sum - 1));

#ifdef DEBUG        
        //        i              j    ik        ok
        if(kid == ROW*kernelSize + COL && c == IN_CH && ok == OUT_CH)
            printf("Wgrad: %f\n", Wgrad[ok][kid + c * wgradArea]);
#endif
	}
}

/*
 * block = dim3(outputAmount);
 * thread= dim3(inputAmount);
 */
__global__ void g_ConvSpiking_calSquareSum(
    float** ws,
    float*  w_sq_sum,
    int     weightArea,
    int     inputAmount,
    int     kernelSize,
    float   weight_limit)
{
    extern __shared__ float _sum[];
    int kernelSize2 = kernelSize * kernelSize;
    int ok = blockIdx.x;
    int ik = threadIdx.x;
    int tid = threadIdx.x;
    float* w = ws[ok] + ik * weightArea;

    _sum[tid] = 0;
    __syncthreads();
    for(int i = 0; i < kernelSize; ++i){
        for(int j = 0; j < kernelSize; ++j){
            float  weight = w[i * kernelSize + j];
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
        w_sq_sum[ok] = _sum[0] / (inputAmount * kernelSize2);
}


void ConvSpiking::getGrad()
{
    dim3 block    = dim3(batch, outputAmount * inputAmount, kernelSize * kernelSize);
    int n_threads = min(outputDim * outputDim, 512);
    dim3 thread   = n_threads;
    cudaFuncSetCacheConfig(g_ConvSpiking_wgrad,cudaFuncCachePreferL1);

    g_ConvSpiking_wgrad<<<block, thread, sizeof(float)*n_threads>>>(
        inputs_time->getDev(),
        outputs_time->getDev(),
        preFireCount->getDev(),
        fireCount->getDev(),
        curDelta->getDev(),
        wgradTmp.m_devPoint,
        inputDim,
        outputDim,
        endTime,
        kernelSize,
        padding,
        inputAmount,
        inputs->getArea(),
        outputs->getArea(),
        curDelta->getArea(),
        wgradTmp[0]->getArea(),
        T_REFRAC,
        TAU_M,
        TAU_S);

    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("g_ConvSpiking_wgrad");
	
    block = dim3(outputAmount);
    thread = dim3(inputAmount);

    g_ConvSpiking_calSquareSum<<<block, thread, sizeof(float) * inputAmount>>>(
        w.m_devPoint,
        weightSqSum->getDev(),
        w[0]->getArea(),
        inputAmount,
        kernelSize,
        weightLimit);
    checkCudaErrors(cudaStreamSynchronize(0));    
	getLastCudaError("g_ConvSpiking_calSquareSum");

	block  = dim3(outputAmount, kernelSize * kernelSize * inputAmount);
	thread = dim3(batch);

	g_ConvSpiking_wgradAdd<<<block, thread, sizeof(float) * batch>>>(
		wgradTmp.m_devPoint,
		wgrad.m_devPoint,
		w.m_devPoint,
        weightSqSum->getDev(),
		kernelSize,
		batch,
		lambda,
        beta,
        weightLimit,
		wgradTmp[0]->getArea(),
		w[0]->getArea());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_ConvSpiking_wgradAdd");
	
}

void ConvSpiking::updateWeight()
{
	dim3 block  = outputAmount;
	dim3 thread = min(512, w[0]->getLen());
    
    if(Config::instance()->getOptimizerType() == std::string("adam")){
        g_adam_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
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
        g_sgd_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
            momentum_w.m_devPoint,
            wgrad.m_devPoint, 
            w.m_devPoint,
            w[0]->getLen(), 
            Config::instance()->getMomentum(),
            Config::instance()->getLrate());
    }
}

ConvSpiking::ConvSpiking(std::string name)
{
	m_name = name;
	ConfigConvSpiking* config = (ConfigConvSpiking*)Config::instance()->getLayerByName(m_name);
	SpikingLayerBase * preLayer = (SpikingLayerBase*)Layers::instance()->get(config->m_input);

    inputs = preLayer->getSpikingOutputs();
    inputs_time = preLayer->getSpikingTimeOutputs();
	preDelta = preLayer->getCurDelta();
    preFireCount = preLayer->getFireCount();

	inputAmount = preLayer->outputAmount;
	outputAmount = config->m_amount;

	kernelSize = config->m_kernelSize;
    padding = config->m_padding;

	inputDim  = preLayer->outputDim;
	outputDim = (inputDim + 1 - kernelSize) + padding * 2;
	batch     = Config::instance()->getBatchSize();
    endTime   = Config::instance()->getEndTime(); 
	batch     = Config::instance()->getBatchSize();
	lambda    = Config::instance()->getLambda();
    beta      = Config::instance()->getBeta();
    T_REFRAC  = config->m_t_ref;
    TAU_M     = config->m_tau_m;
    TAU_S     = config->m_tau_s;    
    threshold = config->m_vth;

    weightLimit = Config::instance()->getWeightLimit();

    outputs  = new cuMatrix<bool>(batch, endTime * outputDim * outputDim, outputAmount);
    outputs_time = new cuMatrix<int>(batch, outputDim * outputDim * endTime, outputAmount);
    inputs_resp = new cuMatrix<float>(batch, endTime * outputDim * outputDim, outputAmount);

	curDelta = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
    fireCount= new cuMatrix<int>(batch, outputDim * outputDim, outputAmount);
    weightSqSum = new cuMatrix<float>(outputAmount, 1, 1);

	for(int i = 0; i < outputAmount; i++){
		w.push_back(new cuMatrix<float>(kernelSize, kernelSize, inputAmount));
		b.push_back(new cuMatrix<float>(1, 1, 1));
		wgrad.push_back(new cuMatrix<float>(kernelSize, kernelSize, inputAmount));
		bgrad.push_back(new cuMatrix<float>(1, 1, 1));
		wgradTmp.push_back(new cuMatrix<float>(batch, kernelSize * kernelSize, inputAmount));
	}

	w.toGpu();
	b.toGpu();
	wgrad.toGpu();
	bgrad.toGpu();
	wgradTmp.toGpu();

	for(int i = 0; i < outputAmount; i++){
		momentum_w.push_back(new cuMatrix<float>(kernelSize, kernelSize, inputAmount));
		momentum_b.push_back(new cuMatrix<float>(1, 1, 1));
        g1_w.push_back(new cuMatrix<float>(kernelSize, kernelSize, inputAmount)); // for adam
        g1_b.push_back(new cuMatrix<float>(1, 1, 1));
        g2_w.push_back(new cuMatrix<float>(kernelSize, kernelSize, inputAmount));
        g2_b.push_back(new cuMatrix<float>(1, 1, 1));       
	}
	momentum_w.toGpu();
	momentum_b.toGpu();
    g1_w.toGpu();
    g1_b.toGpu();
    g2_w.toGpu();
    g2_b.toGpu();
    b1_t = new cuMatrix<float>(outputAmount, 1, 1);
    b2_t = new cuMatrix<float>(outputAmount, 1, 1);
    for(int i = 0; i < outputAmount; i++){
        b1_t->getHost()[i] = 0.9;
        b2_t->getHost()[i] = 0.999;
    }
    b1_t->toGpu();
    b2_t->toGpu();

	this->initRandom();

    output_train_ref = NULL;
    output_test_ref = NULL;
    if(Config::instance()->getIsGradientChecking())
        this->loadRef(); // for verification purpose

	Layers::instance()->set(m_name, this);
}

void ConvSpiking::save(FILE* file)
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

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < b[a]->rows; i++){
				for(int j = 0; j < b[a]->cols; j++){
					fprintf(file, "%f ", b[a]->get(i, j, c));
				}
			}
		}
	}
}

void ConvSpiking::clearMomentum()
{
	for(int i = 0; i < (int)momentum_b.size(); i++){
		momentum_b[i]->gpuClear();
	}
	for(int i = 0; i < (int)momentum_w.size(); i++){
		momentum_w[i]->gpuClear();
	}
}

//* load the reference weights and output spikes for verification
void ConvSpiking::loadRef()
{
    if(batch != 1){
        printf("Only do the verification for one batch and one sample!\n");
        exit(0);
    }
    ConfigConvSpiking * config = (ConfigConvSpiking*)Config::instance()->getLayerByName(m_name);
    if(config->m_ref_weight_path != std::string("NULL")){
        for(int i = 0; i < outputAmount; ++i)
            w_ref.push_back(new cuMatrix<float>(kernelSize, kernelSize, inputAmount));
        initFromDumpfile(config->m_ref_weight_path, w_ref);
    }
    if(config->m_ref_output_train_path != std::string("NULL")){
        output_train_ref = new cuMatrix<bool>(1, endTime * outputDim * outputDim, outputAmount);
        readSpikesFromDumpfile(config->m_ref_output_train_path, output_train_ref);
    }
    if(config->m_ref_output_test_path != std::string("NULL")){
        output_test_ref = new cuMatrix<bool>(1, endTime * outputDim * outputDim, outputAmount);
        readSpikesFromDumpfile(config->m_ref_output_test_path, output_test_ref);
    }

}

void ConvSpiking::initRandom()
{
    ConfigConvSpiking * config = (ConfigConvSpiking*)Config::instance()->getLayerByName(m_name);
    float initW = config->m_initW;

    if(config->isGaussian()){
        for(int i = 0; i < (int)w.size(); i++){
            float epsilon = initW;
            for(int c = 0; c < w[i]->channels; c++)
            {
                createGaussian(w[i]->getHost() + c * w[i]->getArea(),
                        kernelSize, kernelSize, w[i]->channels, epsilon);
            }
            w[i]->toGpu();
        }
    }
    else if(config->isExternal()){
        initFromDumpfile(config->m_weightPath, w);
    }
    else{
        for(int i = 0; i < (int)w.size(); i++){
            for(int j = 0; j < w[i]->getLen(); j++){
                w[i]->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
            }
            w[i]->toGpu();
        }
    }
}

void ConvSpiking::initFromCheckpoint(FILE* file)
{
    float val = 0;
    for(size_t a = 0; a < w.size(); a++){
        for(int c = 0; c < w[a]->channels; c++){
            for(int i = 0; i < w[a]->rows; i++){
                for(int j = 0; j < w[a]->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
                        LOG("scanf fail", "result/log.txt");
                    }
                    w[a]->set(i, j, c, val);
                }
            }
        }

        for(int c = 0; c < b[a]->channels; c++){
            for(int i = 0; i < b[a]->rows; i++){
                for(int j = 0; j < b[a]->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
                        LOG("scanf fail", "result/log.txt");
                    }
                    b[a]->set(i, j, c, val);
                }
            }
        }
        w[a]->toGpu();
        b[a]->toGpu();
    }
}

//* initialize the wight from the dump file by Matlab sim
void ConvSpiking::initFromDumpfile(const std::string& filename, cuMatrixVector<float>& cuW)
{
    assert(filename != std::string("NULL"));
    FILE *file = fopen(filename.c_str(), "r");

    char logStr[256];
    if(file == NULL){
        sprintf(logStr, "Cannot open file: %s", filename.c_str());  LOG(logStr, "Result/log.txt");
        assert(0);
    }
    float val = 0;
    for(size_t a = 0; a < cuW.size(); a++){
        for(int c = 0; c < cuW[a]->channels; c++){
            for(int i = 0; i < cuW[a]->rows; i++){
                for(int j = 0; j < cuW[a]->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
                        sprintf(logStr, "Reading weight failed for %s @row: %d\t@col: %d\t@channel: %d\t@outputAmount: %d", filename.c_str(), i, j, c, int(a));
                        LOG(logStr, "Result/log.txt");
                        assert(0);
                    }
                    cuW[a]->set(i, j, c, val);
                }
            }
        }
        cuW[a]->toGpu();
    }
}


void ConvSpiking::verify(const std::string& phrase)
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
        if(!w_ref.empty()){
            for(int i = 0; i < outputAmount; ++i){
                w[i]->toCpu();
                checkMatrixIsSame(w_ref[i], w[i], i);
            }
        }    
        if(output_test_ref != NULL){
            outputs->toCpu();
            checkMatrixIsSame(output_test_ref, outputs, outputDim*outputDim);
        }
    }
    printf("Verification for the layer: %s at %s phrase. Pased!!\n", m_name.c_str(), phrase.c_str());
}

__device__ float d_ConvSpiking_accumulate_spikes(
    int inputDim,
    int inputArea,
    int inputAmount,
    int kernelSize,
    bool* inputs,
    int x, 
    int y,
    int padding,
    int ok,
    int batchId,
    float ** ws,
    int t,
    int endTime)
{
    int inputSize2 = inputDim * inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    
    float response = 0.0f;
    // for all inputAmount (channels)
    for(int c = 0; c < inputAmount; c++){         
        bool* curInput = inputs + c * inputArea + batchId * inputSize2 * endTime;
        float* w = ws[ok] + c * kernelSize2;
        for(int i = 0; i < kernelSize; i++){
            int xx = x + i - padding;
            for(int j = 0; j < kernelSize; j++){
                int yy = y + j - padding;
                if(xx >= 0 && xx < inputDim && yy >= 0 && yy < inputDim){
                    int i_idx = xx * inputDim + yy;
                    response += curInput[i_idx + t * inputSize2] ?  w[i * kernelSize + j] : 0;
                }
            }
        }
    }
    return response;
}

/*
 * dim3 block = dim3(batch, div, endTime);
 * dim3 thread= dim3(min(outputDim * outputDim, 1024), remain));
 */
__global__ void g_ConvSpiking_fast_input_response(
        bool*  inputs,
        float** ws,
        float** bs,
        float*  inputs_resp,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int endTime,
        int inputAmount,
        int outputAmount,
        int inputArea,
        int responseArea)
{
    int batchId = blockIdx.x;
    int t = blockIdx.z;
    int ok = blockIdx.y * blockDim.y + threadIdx.y;
    if(ok >= outputAmount)return;
    int outputSize2 = outputDim * outputDim;

    float* curResp = inputs_resp + ok * responseArea + batchId * outputSize2 * endTime;
    for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
    {
        int o_idx = tidx + threadIdx.x;
        if(o_idx < outputSize2)
        {
            int x = o_idx / outputDim;
            int y = o_idx % outputDim;
            curResp[o_idx + t * outputSize2] = d_ConvSpiking_accumulate_spikes(inputDim, inputArea, inputAmount, kernelSize, inputs, x, y, padding, ok, batchId, ws, t, endTime);
        }
    }
}

/*
 * dim3 block = dim3(batch, div);
 * dim3 thread= dim3(min(outputDim * outputDim, 1024), remain));
 */
__global__ void g_ConvSpiking_feedforward(
        float*  inputs_resp,
        bool*  outputs,
        int* fireCount,
        int outputDim,
        int endTime,
        int outputAmount,
        int outputArea,
        float vth,
        int T_REFRAC,
        float TAU_M,
        float TAU_S)
{
    int batchId = blockIdx.x;
    int ok = blockIdx.y * blockDim.y + threadIdx.y;
    if(ok >= outputAmount)return;
    int outputSize2 = outputDim * outputDim;

    bool* curOutput = outputs + ok * outputArea + batchId * outputSize2 * endTime;
    int*  curFireCount = fireCount + ok * outputArea / endTime + batchId * outputSize2; 
    float* curResponse = inputs_resp + ok * outputArea + batchId * outputSize2 * endTime;
    for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
    {
        int o_idx = tidx + threadIdx.x;
        if(o_idx < outputSize2)
        {
            float v  = 0.0f;
            float ep = 0.0f;
            float threshold = vth;
            int t_ref= 0;
            float response = 0.0f;
            int fire_count = 0;

            for(int t = 0; t < endTime; t++){
                v  -= v / TAU_M;
                ep -= ep / TAU_S;
                if(t == 0)
                {
                    curOutput[o_idx + t * outputSize2] = false;
                    continue;
                }
                // get the convoluted the spikes
                response = curResponse[o_idx + (t - 1)*outputSize2];
                ep += response;
                v += ep/TAU_S;

                if(t_ref > 0){
                    v = 0;
                    t_ref--;
                }

                // Fire or not
                curOutput[o_idx + t * outputSize2] = v > threshold ?  true : false;
                t_ref = v > threshold ? T_REFRAC : t_ref;
                fire_count += v > threshold ? 1 : 0;
                v = v > threshold ? 0 : v;
            }
            curFireCount[o_idx] = fire_count;
        }
    }
}


/*
 * dim3 block = dim3(batch, inputAmount);
 * dim3 thread= dim3(min(inputDim * inputDim, 1024), 1);
 */
__global__ void g_ConvSpiking_backpropagation(
        float* _curDelta,
        float** ws,
        float* _preDelta,
        int     curDim,
        int     preDim,
        int     preAmount,
        int     curAmount,
        int     kernelSize,
        int     padding,
        int     curArea,
        int     preArea)
{
    int batchId = blockIdx.x;
    int ik      = blockIdx.y * blockDim.y + threadIdx.y;

    if(ik >= preAmount)        return;

    int curSize2    = curDim     * curDim;
    int preSize2    = preDim     * preDim;
    int kernelSize2 = kernelSize * kernelSize;

    float *preDelta = _preDelta + ik * preArea + batchId * preSize2;
    for (int tidx = 0; tidx < preSize2; tidx += blockDim.x) {
        int idx = tidx + threadIdx.x;
        if (idx < preSize2) {
            int i = idx / preDim;
            int j = idx % preDim;
            float val = 0.0;
            int ok = 0;
            int c = ik;
            while(ok < curAmount){
                float *curDelta = _curDelta + ok * curArea + batchId * curSize2;
                float *w = ws[ok] + c * kernelSize2;
                for (int x = 0; x < kernelSize; x++) {
                    int cx = i - x + padding;
                    for (int y = 0; y < kernelSize; y++) {
                        int cy = j - y + padding;
                        if(cx >= 0 && cx < curDim && cy >= 0 && cy < curDim){
                            val += curDelta[cx * curDim + cy] * w[x * kernelSize + y];
                        }
                    }
                }
                ok += 1;
            }
            preDelta[idx] = val;
        }
    }
}


/*
 * dim3 block = dim3(batch, outputAmount*inputAmount, kernelSize * kernelSize);
 * dim3 thread= min(outputDim * outputDim, 512);
 */
__global__ void g_ConvSpiking_wgrad(
        int*    _inputs_time,
        int*    _outputs_time,
        int*    batchPreFireCount,
        int*    batchFireCount,
        float*  _curDelta,
        float** wgradTmp,
        int     inputDim,
        int     curDeltaDim,
        int     endTime,
        int     kernelSize,
        int     padding,
        int     inputAmount,
        int     inputArea,
        int     outputArea,
        int     curDeltaArea,
        int     wgradTmpArea,
        int     T_REFRAC,
        float   TAU_M,
        float   TAU_S)
{
    extern __shared__ float _sum[];
    int tid = threadIdx.x;
    _sum[tid] = 0;
    __syncthreads();

    int ok = blockIdx.y / inputAmount;
    int ik  = blockIdx.y % inputAmount;
    int k_id = blockIdx.z;
    int batchId  = blockIdx.x;

    int inputSize2    = inputDim * inputDim;
    int curDeltaSize2 = curDeltaDim * curDeltaDim;
    int outputSize2   = curDeltaSize2;
    int kernelSize2   = kernelSize * kernelSize;

    float* wgrad = wgradTmp[ok] + ik * wgradTmpArea + batchId * kernelSize2;

    int* input_time    = _inputs_time + inputArea * ik + batchId * inputSize2 * endTime;
    int* output_time   = _outputs_time + outputArea * ok + batchId * outputSize2 * endTime;
    int* input_fireCount  = batchPreFireCount + ik * inputArea / endTime + batchId * inputSize2;
    int* output_fireCount = batchFireCount + ok * outputArea / endTime + batchId * outputSize2;
    float* curDelta = _curDelta + ok * curDeltaArea + batchId * curDeltaSize2;

    for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
    {
        int o_idx = tidx + threadIdx.x;
        if(o_idx < outputSize2)
        {
            int i = k_id / kernelSize;
            int j = k_id % kernelSize;
           
            int x = o_idx / curDeltaDim;
            int y = o_idx % curDeltaDim;
            
            int cx = i + x - padding;
            int cy = j + y - padding;
            if(cx >= 0 &&  cy >= 0 && cx < inputDim && cy < inputDim){
                int i_idx = cx * inputDim + cy;
                float e = d_Spiking_accumulate_effect(output_time, input_time, output_fireCount[o_idx], input_fireCount[i_idx], o_idx, i_idx, outputSize2, inputSize2, endTime, T_REFRAC, TAU_M, TAU_S);
                _sum[tid] += e * curDelta[x * curDeltaDim + y];
#ifdef DEBUG
                if(i == ROW && j == COL && ik == IN_CH && ok == OUT_CH)
                    printf("Collect x= %d; y = %d; Acc effect: %f\tdelta= %f\n", x,y,e,curDelta[x*curDeltaDim + y]);
#endif
            }
        }
    }
    int len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(tid < skip && (tid + skip) < len){
            _sum[tid] += _sum[tid + skip];
        }
        len = skip;
    }
    if(tid == 0)
        wgrad[k_id] = _sum[0];
}

/*
 * blocks  : dim3(kernelAmount2)
 * threads : dim3(256)
 * shared  : sizeof(float) * 256
 */
__global__ void g_ConvSpiking_Bgrad(float* delta,
        float** bgrad,
        int deltaSize,
        int kernelAmount2,
        int batch,
        int deltaArea)
{
    extern __shared__ float _sum[];
    int k2 = blockIdx.x;
    _sum[threadIdx.x] = 0.0;
    __syncthreads();
    int deltaSize2 = deltaSize * deltaSize;
    int tlen = deltaSize2 * batch;
    int skip = deltaArea * k2;
    for(int i = 0; i < tlen; i += blockDim.x)
    {
        int idx = i + threadIdx.x;
        if(idx < tlen)
        {
            _sum[threadIdx.x] += delta[idx + skip];
        }
    }
    __syncthreads();
    int len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < skip && (threadIdx.x + skip < len))
        {
            _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        else{
            return;
        }
        len = skip;
    }
    if(threadIdx.x == 0)
    {
        bgrad[k2][0] = _sum[0] / batch;
    }
}
