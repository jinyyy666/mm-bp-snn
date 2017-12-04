#include "SoftMaxSpiking.h"
#include "../common/cuBase.h"
#include "../common/cuMatrix.h"
#include "../common/Config.h"
#include "../layers/BranchLayer.h"
#include <math.h>

/*
* function: normalize the fire counts by the max count for SNN
* blocks  : dim3(batch)
* threads : dim3(min(1024, inputDim))
*/
__global__ void g_normalize_fireCount(int * inputs, float * inputs_float, int rows, int cols)
{
    extern __shared__ int _max[];
    int batchId = blockIdx.x;
    int len = blockDim.x;
    int id = threadIdx.x;

    _max[id] = 0;
    __syncthreads();

    for(int tid = 0; tid < cols; tid += blockDim.x){
        int ttid = tid + threadIdx.x;
        if(ttid < cols){
            _max[threadIdx.x] = max(_max[threadIdx.x], inputs[ttid + batchId * cols]);
        }
    }
    int fire_count = inputs[id + batchId * cols];

    while(len != 1)
    { 
        __syncthreads();
        int skip = (len + 1)>>1;
        if(id < skip && (id + skip) < len)
        {
            _max[id] = max(_max[id],  _max[id + skip]);
        }
        len = skip;
    }
    __syncthreads();
    inputs_float[id + batchId * cols] = float(fire_count);///float(_max[0]); // be careful of zero fire count
}


void SoftMaxSpiking::feedforward()
{
	dim3 block  = batch;
	int threads = min(1024, inputDim);
	// normalize the fire counts by the max count
	g_normalize_fireCount<<<block, threads, sizeof(int) * threads>>>(
		inputs->getDev(),
		inputs_float->getDev(), 
		inputs->rows, 
		inputs->cols);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_normalize_fireCount");

	matrixMulTB(inputs_float,
		w, outputs);

	threads = min(512, outputs->cols);
	g_getSoftMaxP<<<outputs->rows, threads, sizeof(float) * threads * 2>>>(
		outputs->getDev(),
		b->getDev(), 
		outputs->cols);
	cudaStreamSynchronize(0);
	getLastCudaError("g_getSoftMaxP");
}

void SoftMaxSpiking::backpropagation()
{
	g_getCost_1<<<dim3(1), dim3(256), sizeof(float) * 256>>>(outputs->getDev(), groundTruth->getDev(),
		cost->getDev(), predict, outputs->rows, outputs->cols, batch);
	cudaStreamSynchronize(0);
	getLastCudaError("g_getCost_1");

	g_getSoftMaxDelta<<<dim3(1), dim3(256)>>>(curDelta->getDev(),
		outputs->getDev(),
		groundTruth->getDev(), curDelta->getLen());
	cudaStreamSynchronize(0);
    getLastCudaError("g_getSoftMaxDelta");

	matrixMul(curDelta, w, preDelta);

}

void SoftMaxSpiking::getGrad()
{
	matrixMulTA(curDelta, inputs_float, wgrad);

	g_getSmrWgrad<<<dim3(1), dim3(256)>>>(wgrad->getDev(),
		w->getDev(), lambda, wgrad->getLen(), batch);
	cudaStreamSynchronize(0);
	getLastCudaError("g_getSmrWgrad");

	if(curDelta->rows > MAX_THREADS)
	{
		printf("getSoftMaxDelta g_getBgrad > MAX_THREADS\n");
		exit(0);
	}
	g_getBgrad<<<dim3(curDelta->cols), dim3(curDelta->rows), 
		sizeof(float) * curDelta->rows>>>(
		curDelta->getDev(), 
		bgrad->getDev(),
		batch);
	cudaStreamSynchronize(0);
	getLastCudaError("g_getBgrad");
}

void SoftMaxSpiking::updateWeight()
{
	g_vecAdd<<<dim3(min((momentum_w->getLen() + 255) / 256, 512)),
		dim3(256), 0, Layers::instance()->get_stream()>>>(
		momentum_w->getDev(), 
		wgrad->getDev(), 
		w->getDev(),
		momentum_b->getDev(), 
		bgrad->getDev(), 
		b->getDev(), 
		wgrad->getLen(),
		bgrad->getLen(),
		Config::instance()->getMomentum(), 
		Config::instance()->getLrate(), Config::instance()->getLrate());
}

void SoftMaxSpiking::clearMomentum()
{
	momentum_b->gpuClear();
	momentum_w->gpuClear();
}

void SoftMaxSpiking::calCost()
{
    cost->gpuClear();
	g_getCost_2<<<dim3(1), dim3(256), sizeof(float) * 256>>>(cost->getDev(),  w->getDev(), lambda,
		w->getLen());
	cudaStreamSynchronize(0);
	getLastCudaError("g_getCost_2");
}


void SoftMaxSpiking::initRandom()
{
    ConfigBase * config = (ConfigBase*)Config::instance()->getLayerByName(m_name);
	float initW = config->m_initW;

	if(config->isGaussian()){
		float epsilon = initW;
		for(int c = 0; c < w->channels; c++){
			float r1 = 0.01f + 5.0f * (rand()) / RAND_MAX;
			float r2 = 0.01f + 5.0f * (rand()) / RAND_MAX;
			createGaussian(w->getHost() + c * w->getArea(), r1,r2,
				w->rows, w->cols, w->channels,
				epsilon);
		}
	}
	else{
		for(int j = 0; j < w->getLen(); j++){
			w->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
		}
	}
	w->toGpu();
}
	
void SoftMaxSpiking::initFromCheckpoint(FILE* file)
{
	float val = 0.0;
	for(int i = 0; i < w->rows; i++){
		for(int j=0; j< w->cols; j++){
            if(fscanf(file, "%f", &val) == EOF){
                printf("scan fail for layer: %s", m_name.c_str());
                assert(0);
            }
			w->set(i,j,0,val);
		}
	}
	
	for(int i = 0; i < b->rows; i++){
		for(int j = 0; j < b->cols; j++){
            if(fscanf(file, "%f ", &val) == EOF){
                printf("scan fail for layer: %s", m_name.c_str());
                assert(0);
            }
			b->set(i,j,0, val);
		}
	}
	w->toGpu();
	b->toGpu();
}

void SoftMaxSpiking::save(FILE* file)
{
	w->toCpu();
	b->toCpu();
	for(int c = 0; c < w->channels; c++){
		for(int i = 0; i< w->rows; i++){
			for(int j=0; j< w->cols; j++){
				fprintf(file, "%f ", w->get(i,j,c)); 
			}
		}
	}

	for(int c = 0; c < b->channels; c++){
		for(int i = 0; i < b->rows; i++){
			for(int j = 0; j < b->cols;  j++){
				fprintf(file, "%f ", b->get(i,j,c));
			}
		}
	}
}

SoftMaxSpiking::SoftMaxSpiking(std::string name)
{
	m_name = name;
	ConfigSoftMaxSpiking* config = (ConfigSoftMaxSpiking*)Config::instance()->getLayerByName(m_name);
	SpikingLayerBase * preLayer = (SpikingLayerBase*)Layers::instance()->get(config->m_input);
	
	inputs = preLayer->getFireCount();
	preDelta = preLayer->getCurDelta();

	batch = Config::instance()->getBatchSize();
	lambda = config->m_weightDecay;

	inputDim = preLayer->outputDim;
	outputDim = config->m_numClasses;

	inputs_float = new cuMatrix<float>(inputs->rows, inputs->cols, 1);
	outputs = new cuMatrix<float>(batch, outputDim, 1);
	curDelta= new cuMatrix<float>(batch, outputDim, 1);

    w     = new cuMatrix<float>(outputDim, inputDim, 1);
	wgrad = new cuMatrix<float>(outputDim, inputDim, 1);

	b     = new cuMatrix<float>(outputDim, 1, 1);
	bgrad = new cuMatrix<float>(outputDim, 1, 1);

	momentum_w = new cuMatrix<float>(outputDim, inputDim, 1);
	momentum_b = new cuMatrix<float>(outputDim, 1, 1);

	groundTruth = new cuMatrix<float>(batch, outputDim, 1);

	this->initRandom();
	Layers::instance()->set(m_name, this);
}
