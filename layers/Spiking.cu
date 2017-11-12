#include "Spiking.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../common/util.h"
#include "../readData/readSpeechData.h"
#include <fstream>
#include <assert.h>
#include <math.h>


/*
 * Device func for accumulate the spike response
 *
*/
__device__ float d_Spiking_accumulate_spikes(
    int inputDim,
    int outputDim,
    bool* input,
    bool* output,
    float* weights,
    float* weights_lat,
    int t);

/*
 * Device func for spike gradient for each pair of spike train
 *
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
__global__ void g_getMaxCount_output(
    int* fireCount,
    int* maxCount,
    int cols); 

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(1);
 */
__global__ void g_modifySpikes_output(
    bool* outputs,
    int* y,
    int* fireCount,
    int* maxCount,
    int endTime,
    int outputDim);


/*
 * dim3 block = dim3(batch, inputDim, outputDim);
 * dim3 thread= dim3(256);
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
 * block = dim3(outputDim * inputDim, outputAmount);
 * thread= dim3(batch);
*/
__global__ void g_Spiking_wgradAdd(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
	int batch,
	float lambda,
	int wArea);


/*
 *	blocks : dim3(batch, div),
 *	threads: dim3(min(outputDim, 1024), 1);
 */
__global__ void g_Spiking_feedforward(
	bool*  inputs,
	float** ws,
    float** ws_lat,
	bool*  outputs,
    int*    fireCount,
	int inputDim,
	int outputDim,
    int endTime,
	int inputAmount,
	int outputAmount,
    float vth,
    int T_REFRAC,
    float TAU_M,
    float TAU_S);


/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_Spiking_vecAdd(
    float* v_m, 
    float* wgrad, 
    float* w, 
    int lenw, 
    float momentum, 
    float lr);


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
	getLastCudaError("Spiking:getCost");
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
    g_Spiking_feedforward<<<block, thread>>>(
        inputs->getDev(),
        w.m_devPoint,
        w_lat_dev,
        outputs->getDev(),
        fireCount->getDev(),
        inputDim,
        outputDim,
        endTime,
        inputAmount,
        outputAmount,
        vth,
        T_REFRAC,
        TAU_M,
        TAU_S);
    checkCudaErrors(cudaStreamSynchronize(0));
    getLastCudaError("Spiking::g_Spiking_feedforward");

}

void Spiking::backpropagation()
{
    if(m_name == std::string("output")){
        // compute the cost function
        g_getCost_output<<<dim3(1), dim3(256), sizeof(float) * 256>>>(fireCount->getDev(), groundTruth->getDev(), cost->getDev(), predict, batch, fireCount->cols, UNDESIRED_LEVEL, DESIRED_LEVEL, MARGIN);
	    cudaStreamSynchronize(0);
	    getLastCudaError("Spiking::g_getCost_output");

        // compute the delta (error)
        g_getDelta_output<<<dim3(1), dim3(256)>>>(curDelta->getDev(), fireCount->getDev(), groundTruth->getDev(), curDelta->getLen(), MARGIN);
	    cudaStreamSynchronize(0);
	    getLastCudaError("Spiking::g_getDelta_output");

        // reduce to get the max fire count for each sample in the batch
        g_getMaxCount_output<<<dim3(batch), dim3(outputDim), sizeof(int) * outputDim>>>(fireCount->getDev(), maxCount->getDev(), fireCount->cols);  
	    cudaStreamSynchronize(0);
	    getLastCudaError("Spiking::g_getMaxCount_output");

        // modify the output spikes of the target neuron if it does not fire
        g_modifySpikes_output<<<dim3(batch), dim3(1)>>>(outputs->getDev(), predict, fireCount->getDev(), maxCount->getDev(), endTime, outputDim);
        cudaStreamSynchronize(0);
        getLastCudaError("Spiking::g_modifySpikes_output");
    }

    // compute preDelta: curDelta: batch * outputDIm; w: outputDim * inputDim
    assert(w.size() == 1);
    if(preDelta == NULL){
        ConfigSpiking* config = (ConfigSpiking*)Config::instance()->getLayerByName(m_name);
        assert(config->m_input == "data");
    }
    else{
        matrixMul(curDelta, w[0], preDelta);
    }    
    // need more code to multi-channel input, simply learn: FullConnect.cu
}

/*
 * block = dim3(outputDim * inputDim, outputAmount);
 * thread= dim3(batch);
*/
__global__ void g_Spiking_wgradAdd(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
	int batch,
	float lambda,
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
		Wgrad[ok][wid] = _sum[0] / batch + w[ok][wid] * lambda;
	}
}

void Spiking::getGrad()
{
    dim3 thread = dim3(4);
    dim3 block  = dim3(batch, inputDim, outputDim);
    cudaFuncSetCacheConfig(g_Spiking_wgrad,cudaFuncCachePreferL1);
    g_Spiking_wgrad<<<block, thread, sizeof(float)>>>(
        inputs->getDev(),
        outputs->getDev(),
        curDelta->getDev(),
        wgradTmp.m_devPoint,
        inputDim,
        outputDim,
        endTime,
        T_REFRAC,
        TAU_M,
        TAU_S);

    checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_wgrad");

	block  = dim3(outputDim * inputDim, outputAmount);
	thread = dim3(batch);

	g_Spiking_wgradAdd<<<block, thread, sizeof(float) * batch>>>(
		wgradTmp.m_devPoint,
		wgrad.m_devPoint,
		w.m_devPoint,
		batch,
		lambda,
		w[0]->getArea());

	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_Spiking_wgradAdd");
}	


/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_Spiking_vecAdd(float** momentum_w, float** _wgrad, float** _w, int lenw, float momentum, float lr)
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
	g_Spiking_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(
        momentum_w.m_devPoint,
        wgrad.m_devPoint, 
        w.m_devPoint,
        w[0]->getLen(), 
		Config::instance()->getMomentum(),
		Config::instance()->getLrate());
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
	preDelta = preLayer->getCurDelta();
	
    inputAmount  = preLayer->outputAmount;
    outputAmount = inputAmount;

	inputDim  = preLayer->outputDim;
	outputDim = config->m_numNeurons;
    endTime   = Config::instance()->getEndTime(); 
	batch     = Config::instance()->getBatchSize();
	lambda    = config->m_weightDecay;
    vth       = config->m_vth;
    T_REFRAC  = config->m_t_ref;
    TAU_M     = config->m_tau_m;
    TAU_S     = config->m_tau_s;    

    UNDESIRED_LEVEL = config->m_undesired_level;
    DESIRED_LEVEL   = config->m_desired_level;
    MARGIN          = config->m_margin; 

	outputs  = new cuMatrix<bool>(batch, outputDim * endTime, outputAmount);
	curDelta = new cuMatrix<float>(batch, outputDim, outputAmount);
    fireCount= new cuMatrix<int>(batch, outputDim, outputAmount);
    // only for the output
    if(config->m_name == std::string("output")){
        groundTruth = new cuMatrix<float>(batch, outputDim, 1);
        cost        = new cuMatrix<float>(1, 1, 1);
        maxCount    = new cuMatrix<int>(batch, 1, 1);
    }
    else{
        groundTruth = NULL;
        cost        = NULL;
        maxCount    = NULL;
    }
    assert(outputDim > 0 && inputDim > 0);

	for(int i = 0; i < outputAmount; i++){
		w.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
		b.push_back(new cuMatrix<float>(outputDim, 1, 1));
		wgrad.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
		bgrad.push_back(new cuMatrix<float>(outputDim, 1, 1));
		wgradTmp.push_back(new cuMatrix<float>(batch, outputDim * inputDim, 1));

        if(config->hasLaterialWeight() == true){
            w_laterial.push_back(new cuMatrix<float>(outputDim, outputDim, 1));
        }
	}

	w.toGpu();
	b.toGpu();
	wgrad.toGpu();
	bgrad.toGpu();
	wgradTmp.toGpu();

    if(config->hasLaterialWeight() == true){
        w_laterial.toGpu();
    }

	for(int i = 0; i < outputAmount; i++){
		momentum_w.push_back(new cuMatrix<float>(outputDim, inputDim, 1));
		momentum_b.push_back(new cuMatrix<float>(outputDim, 1, 1));
	}
	momentum_w.toGpu();
	momentum_b.toGpu();

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
                        LOG("scanf fail", "result/log.txt");
                    }
					w[a]->set(i, j, c, val);
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < b[a]->rows; i++){
				for(int j = 0; j < b[a]->cols; j++){
					if(fscanf(file, "%f", &val) != EOF)
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
            printf("Read the file: %s, in line: %d\n", filename.c_str(), idx);
            printf("Post: %d, OutputDim: %d\n Pre: %d, InputDim: %d\n", post, (int)weights.size(), pre, (int)weights[0].size());
            assert(post < weights.size() && pre < weights[0].size());
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
    float* weights,
    float* weights_lat,
    int t)
{
    int idx = threadIdx.x;
    if(idx >= outputDim * inputDim){
        return 0;
    }  
    float response = 0.0f;
    // effect from the forward-connects
    for(int i = 0; i < inputDim; ++i){
        response += input[i + (t - 1) * inputDim] ? weights[i + idx * inputDim] : 0; 
    }
    
    if(weights_lat != NULL){
        // effect from the recurrent connections:
        for(int i = 0; i < outputDim; ++i)
            response += output[i + (t - 1) * outputDim] ? weights_lat[i + idx * outputDim] : 0;
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
        cost[0] = _sum[0]/batch;
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
__global__ void g_getMaxCount_output(int* fireCount, int* maxCount, int cols)
{
    extern __shared__ int _max[];
    int batchId = blockIdx.x;
    int len = blockDim.x;
    int id = threadIdx.x;

    _max[id] = fireCount[id + batchId * cols];
    __syncthreads();
    while(len != 1)
    {
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
 * dim3 thread= dim3(1);
 */
__global__ void g_modifySpikes_output(bool* outputs, int* y, int* fireCount, int* maxCount, int endTime, int outputDim)
{
    int batchId = blockIdx.x;
    int target = y[batchId];
    int mCnt = maxCount[batchId]; 
    bool* outputSpikes = outputs + batchId * endTime * outputDim;
    if(fireCount[target + batchId * outputDim] == 0)
    {
        int interval = mCnt == 0 ? endTime/4 : endTime / mCnt;
        for(int t = interval; t < endTime; t += interval)
        {
            outputSpikes[target + t * outputDim] = true;
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
	bool*  outputs,
    int* fireCount,
	int inputDim,
	int outputDim,
    int endTime,
	int inputAmount,
	int outputAmount,
    float vth,
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

    // simulate the spiking train
    for(int tidx = 0; tidx < outputDim; tidx += blockDim.x)
    {
        int o_idx = tidx + threadIdx.x;
        if(o_idx < outputDim)
        {
            float v  = 0.0f;
            float ep = 0.0f;
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
                response = d_Spiking_accumulate_spikes(inputDim, outputDim, curInput, curOutput, w, w_l, t);
                
                // 3. Add up the response to ep (state variable)
                ep += response;

                // 4. Update the vmem accordingly
                v += ep/TAU_S;
                if(t_ref > 0){
                    v = 0;
                    t_ref--;
                }
            
                // 5. Fire or not
                curOutput[o_idx + t * outputDim] = v > vth ?  true : false;
                t_ref = v > vth ? T_REFRAC : t_ref;
                fire_count += v > vth ? 1 : 0;
                v = v > vth ? 0 : v;
            }
            curFireCount[o_idx] = fire_count; 
        }
    }
}


/*
 * dim3 block = dim3(batch, inputDim, outputDim);
 * dim3 thread= min(256);
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
    extern __shared__ float acc_response[];
    int batchId = blockIdx.x;
    int i_idx   = blockIdx.y;
    int o_idx   = blockIdx.z;

    int wSize        = outputDim * inputDim;
    int inputSize2   = endTime * inputDim;
    int outputSize2  = endTime * outputDim;
    int curDeltaSize = outputDim;

    float* wgrad  = wgradTmp[0] + batchId * wSize;
    bool* input    = inputs + batchId * inputSize2;
    bool* output   = outputs + batchId * outputSize2;
    float* cDelta = curDelta + batchId * curDeltaSize;

    acc_response[0] = 0;
    __syncthreads();

    int t_post_last = -1;
    int thread_handle_pts = (endTime + blockDim.x - 1)/ blockDim.x;
    int tid = threadIdx.x;
    for(int t_post = thread_handle_pts * tid; t_post < thread_handle_pts * (tid + 1); t_post++)
    {
        //int t_post = threadIdx.x + tt_post;
        if(t_post == 0 || t_post > endTime) continue;

        if(output[o_idx + t_post * outputDim] == true){
            // 1. find the last post_spike
            if(t_post_last == -1){
                for(int time = t_post - 1; time > 1; --time){
                    if(output[o_idx + time * outputDim] == true){
                        t_post_last = time + T_REFRAC;
                        break;
                    }
                }
                if(t_post_last == -1)   t_post_last = 1;
            }
            
            
            // 2. Compute the accumulative effect                
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
            atomicAdd(&acc_response[0], sum); 
            t_post_last = t_post + T_REFRAC;
        }
    }
    __syncthreads();
    float delta_w = cDelta[o_idx] * acc_response[0];
    wgrad[i_idx + o_idx * inputDim] = delta_w;

}


