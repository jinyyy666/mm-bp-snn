#include "net_spiking.cuh"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "common/util.h"
#include <time.h>
#include "common/Config.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include "common/MemoryMonitor.h"
#include "common/cuBase.h"
#include "layers/LayerBase.h"
#include "layers/DataLayerSpiking.h"
#include "layers/Spiking.h"
#include "layers/SoftMaxSpiking.h"
#include <queue>
#include <set>


int cuSCurCorrect;
cuMatrix<int>*  cuSCorrect = NULL;
cuMatrix<int>*  cuSVote = NULL;
cuMatrix<bool>* cuSPredictions = NULL;
cuMatrix<int>*  cuSTrCorrect = NULL;
cuMatrix<int>*  cuSTrVote = NULL;
cuMatrix<bool>* cuSTrPredictions = NULL;
cuMatrix<float>* cuSampleWeight = NULL;
std::vector<ConfigBase*> spiking_que;

void cuSaveSpikingNet()
{	
    FILE *pOut = fopen("Result/checkPoint.txt", "w");
    for(int i = 0; i < (int)spiking_que.size(); i++){
        LayerBase* layer = Layers::instance()->get(spiking_que[i]->m_name);
        layer->save(pOut);
    }
    fclose(pOut);
};

void cuFreeSpikingNet()
{
}

void cuReadSpikingNet(const char* path)
{	
    FILE *pIn = fopen(path, "r");

    for(int i = 0; i < (int)spiking_que.size(); i++){
        LayerBase* layer = Layers::instance()->get(spiking_que[i]->m_name);
        layer->initFromCheckpoint(pIn);
    }

    fclose(pIn);
};

void buildSpikingNetwork(int trainLen, int testLen)
{
    /*BFS*/
    std::queue<ConfigBase*>qqq;
    std::set<ConfigBase*> inque;
    for(int i = 0; i < (int)Config::instance()->getFirstLayers().size(); i++){
        qqq.push(Config::instance()->getFirstLayers()[i]);
        inque.insert(Config::instance()->getFirstLayers()[i]);
    }

    char logStr[1024];
    sprintf(logStr, "\n\n******************layer nexts start********************\n");
    LOG(logStr, "Result/log.txt");
    std::set<ConfigBase*>finish;
    while(!qqq.empty()){
        ConfigBase* top = qqq.front();
        qqq.pop();
        finish.insert(top);
        spiking_que.push_back(top);

        if(top->m_type == std::string("DATASPIKING")){
            new DataLayerSpiking(top->m_name);
        }
        else if(top->m_type == std::string("SPIKING")){
            new Spiking(top->m_name);
        }
        else if(top->m_type == std::string("SOFTMAXSPIKING")){
            new SoftMaxSpiking(top->m_name);
        }

        sprintf(logStr, "layer %15s:", top->m_name.c_str());
        LOG(logStr, "Result/log.txt");
        for(int n = 0; n < (int)top->m_next.size(); n++){
            if(inque.find(top->m_next[n]) == inque.end()){
                qqq.push(top->m_next[n]);
                inque.insert(top->m_next[n]);
            }
            sprintf(logStr, "%s ", top->m_next[n]->m_name.c_str());
            LOG(logStr, "Result/log.txt");
        }sprintf(logStr, "\n");
        LOG(logStr, "Result/log.txt");
    }

    sprintf(logStr, "\n\n******************layer nexts end********************\n");
    LOG(logStr, "Result/log.txt");

    //* correct and cuSVote for tracking the test results
	if(cuSCorrect == NULL)
	{
		cuSCorrect = new cuMatrix<int>(1,1,1);
		cuSVote    = new cuMatrix<int>(testLen, Config::instance()->getClasses(), 1);
        cuSPredictions = new cuMatrix<bool>(testLen, 1, 1);
	}
    //* cuSTrCorrect and cuSTrVote for tracking the training results
    if(cuSTrCorrect == NULL)
    {
        cuSTrCorrect = new cuMatrix<int>(1,1,1);
        cuSTrVote = new cuMatrix<int>(trainLen, Config::instance()->getClasses(), 1);
        cuSTrPredictions = new cuMatrix<bool>(trainLen, 1, 1);
    }
    // boost weighted training
    if(cuSampleWeight == NULL)
    {
        cuSampleWeight = new cuMatrix<float>(trainLen, 1, 1);
        for(int i = 0; i < cuSampleWeight->getLen(); i++){
            cuSampleWeight->getHost()[i] = 1.0f;
        }
        cuSampleWeight->toGpu();
    }
}

void cuFreeSNNMemory(
        int batch,
        cuMatrixVector<bool>&trainX, 
        cuMatrixVector<bool>&testX)
{
}

/*
 * Get the network prediction result
 * block = dim3(1)
 * thread = dim3(batch)
 */
__global__ void g_getPredict(int* batchfireCount, int cols,  int start, int* vote)
{
    int batchid = threadIdx.x;
    if(batchid < start) return;
    int* p = batchfireCount + batchid * cols;
    int* votep = vote + batchid * cols;

    int r = 0;
    int maxCount = 0;
    for(int i = 0; i < cols; i++)
    {
        int cnt = p[i];
        if(maxCount < cnt)
        {
            maxCount = cnt;
            r = i;
        }
    }
    votep[r]++;
}

/*
* Get the predict based on softmax
* dim3(1),dim3(batch)
*/
__global__ void g_getPredict_softmax(float* softMaxP, int cols,  int start, int* vote)
{
	int id = threadIdx.x;
	if(id < start) return;
	float* p = softMaxP + id * cols;
	int* votep= vote     + id * cols;

	int r = 0;
	float maxele = log(p[0]);
	for(int i = 1; i < cols; i++)
	{
		float val = log(p[i]);
		if(maxele < val)
		{
			maxele = val;
			r = i;
		}
	}
	votep[r]++;
}

//* get the prediction from the spiking output layer
void outputPredict(int* vote, int start)
{
    for(int i = 0; i < (int)spiking_que.size(); i++){
        if(spiking_que[i]->m_name == std::string("output")){
            g_getPredict<<<dim3(1), Config::instance()->getBatchSize()>>>(
                    Layers::instance()->get(spiking_que[i]->m_name)->getFireCount()->getDev(),
                    Layers::instance()->get(spiking_que[i]->m_name)->getFireCount()->cols,
                    start,
                    vote);
            cudaStreamSynchronize(0);
            getLastCudaError("g_getPredict");
        }
        if(spiking_que[i]->m_type == std::string("SOFTMAXSPIKING")){
			g_getPredict_softmax<<<dim3(1), Config::instance()->getBatchSize()>>>(
				Layers::instance()->get(spiking_que[i]->m_name)->getOutputs()->getDev(),
				Layers::instance()->get(spiking_que[i]->m_name)->getOutputs()->cols,
				start,
				vote);
			cudaStreamSynchronize(0);
            getLastCudaError("g_getPredict_softmax");
		}
    }
   
}



void getSpikingNetworkCost(int* y, float* weights, int* vote, int start)
{
    /*feedforward*/
    for(int i = 0; i < (int)spiking_que.size(); i++){
        if(spiking_que[i]->m_name == std::string("output") || spiking_que[i]->m_type == std::string("SOFTMAXSPIKING")){
            SpikingLayerBase* output = (SpikingLayerBase*)Layers::instance()->get(spiking_que[i]->m_name);
            output->setPredict(y);
            output->setSampleWeight(weights);
        }
    }

    for(int i = 0; i < (int)spiking_que.size(); i++){
        LayerBase* layer = Layers::instance()->get(spiking_que[i]->m_name);
        layer->feedforward();
    }
    
    /*record the prediction*/
    outputPredict(vote, start);

    /*backpropagation*/
    bool has_dynamic_threshold = Config::instance()->hasDynamicThreshold(); 
    for(int i = (int)spiking_que.size() - 1; i >=0; i--){
        ConfigBase* top = spiking_que[i];
        if(top->m_name == std::string("reservoir")) continue;

        SpikingLayerBase* layer = (SpikingLayerBase*)Layers::instance()->get(top->m_name);

        layer->backpropagation();
        layer->getGrad();
        layer->updateWeight();
        if(has_dynamic_threshold && top->m_name != std::string("output"))
        {
            layer->getDeltaVth();
            layer->updateVth();
        }
    }
    cudaStreamSynchronize(Layers::instance()->get_stream());
    getLastCudaError("updateWB");
}

void resultPredict(int* y, int* vote, int start)
{
    /*feedforward*/
    for(int i = 0; i < (int)spiking_que.size(); i++){
        if(spiking_que[i]->m_name == std::string("output") || spiking_que[i]->m_type == std::string("SOFTMAXSPIKING")){
            SpikingLayerBase* output = (SpikingLayerBase*)Layers::instance()->get(spiking_que[i]->m_name);
            output->setPredict(y);
        }
    }

    for(int i = 0; i < (int)spiking_que.size(); i++){
        LayerBase* layer = Layers::instance()->get(spiking_que[i]->m_name);
        layer->feedforward();
    }

    /*obtain the prediction predict*/
    outputPredict(vote, start);
}

void gradientChecking(bool**x, int*y, int batch, int nclasses, cublasHandle_t handle)
{
}

/*
 * block = (testX.size() + batch - 1) / batch
 * thread = batch
 */
void __global__ g_getSpikeVotingResult(int* voting, int* y, int* correct, bool* predictions, int len, int nclasses)
{
    for(int i = 0; i < len; i += blockDim.x * gridDim.x)
    {
        int idx = i + blockDim.x * blockIdx.x + threadIdx.x;
        if(idx < len)
        {
            int* pvoting = voting + idx * nclasses;
            int _max = pvoting[0];
            int rid  = 0;
            for(int j = 1; j < nclasses; j++)
            {
                if(pvoting[j] > _max)
                {
                    _max = pvoting[j];
                    rid  = j;
                }
            }
            if(rid == y[idx])
            {
                atomicAdd(correct, 1);
                predictions[idx] = true;
            }
        }
    }
}

/*
 * block = 1
 * thread = nclasses
 */
void __global__ g_boostWeightUpdate(float* weights, bool* predictions, int* y, int len, int nclasses)
{
	extern __shared__ float sums[];
    float * sum_weights = (float*)sums;
    float * error_weighted = (float*)&sums[nclasses];

    int tid = threadIdx.x;
    sum_weights[tid] = 0;
    error_weighted[tid] = 0;
    __syncthreads();

    // 1. compute the sum of the boosting weight for each class
    for(int i = 0; i < len; i += blockDim.x)
    {
        int idx = i + tid;
        if(idx < len)
        {
            int cls = y[idx];
            float w = weights[idx];
            atomicAdd(&sum_weights[cls], w);
        }
    }
    __syncthreads();
    
    // 2. compute the weighted error for each class 
    for(int i = 0; i < len; i += blockDim.x)
    {
        int idx = i + tid;
        if(idx < len)
        {
            int cls = y[idx];
            bool prediction = predictions[idx];
            float w = weights[idx];
            atomicAdd(&error_weighted[cls], w*(!prediction)/sum_weights[cls]);
        } 
    }
    __syncthreads();

    // 3. update the boost weight for each training sample
    for(int i = 0; i < len; i += blockDim.x)
    {
        int idx = i + tid;
        if(idx < len)
        {
            bool prediction = predictions[idx];
            int cls = y[idx];
            float w = weights[idx];
            float stage = error_weighted[cls]/20.0f;
            float new_w = w * __expf(stage * (!prediction));
            weights[idx] = new_w;
            /*
            if(prediction)
                printf("Sample: %d predicts correctly old sample weight: %f new sample weight %f\n", cls, w, new_w);
            else 
                printf("Sample: %d predicts incorrectly old sample weight: %f new sample weight %f\n", cls, w, new_w);
            */
        }
    }
}

//* verify that the GPU sim result aligns with CPU sim
void verifyResult(std::string phrase)
{
    for(int i = 0; i < (int)spiking_que.size(); i++){
        if(spiking_que[i]->m_name == "data")    continue;
        Spiking* layer = (Spiking*)Layers::instance()->get(spiking_que[i]->m_name);
        layer->verify(phrase);
    }
}


void predictTestRate(cuMatrixVector<bool>&x,
        cuMatrix<int>*y ,
        cuMatrixVector<bool>&testX,
        cuMatrix<int>* testY,
        int batch,
        int nclasses,
        cublasHandle_t handle) {
    Config::instance()->setTraining(false);

    DataLayerSpiking *dl = static_cast<DataLayerSpiking*>(Layers::instance()->get("data"));
    dl->getBatchSpikesWithStreams(testX, 0);

    cuSVote->gpuClear();
    for (int k = 0; k < ((int)testX.size() + batch - 1) / batch; k ++) {
        dl->synchronize();
        int start = k * batch;
        printf("test %2d%%", 100 * start / (((int)testX.size() + batch - 1)));

        if(start + batch <= (int)testX.size() - batch)
            dl->getBatchSpikesWithStreams(testX, start + batch);
        else{
            int tstart = testX.size() - batch;
            dl->getBatchSpikesWithStreams(testX, tstart);
        }

        if(start + batch > (int)testX.size()){
            start = (int)testX.size() - batch;
        }

        dl->testData();
        resultPredict(testY->getDev() + start, cuSVote->getDev() + start * nclasses, k * batch - start);
        printf("\b\b\b\b\b\b\b\b\b");
    }

    cuSCorrect->gpuClear();
    cuSPredictions->gpuClear();
    g_getSpikeVotingResult<<<dim3((testX.size() + batch - 1) / batch), dim3(batch)>>>(
            cuSVote->getDev(),
            testY->getDev(),
            cuSCorrect->getDev(),
            cuSPredictions->getDev(),
            testX.size(),
            nclasses);
    cudaStreamSynchronize(0);
    getLastCudaError("g_getSpikeVotingResult");
    cuSCorrect->toCpu();
    if (cuSCorrect->get(0, 0, 0) > cuSCurCorrect) {
        cuSCurCorrect = cuSCorrect->get(0, 0, 0);
        cuSaveSpikingNet();
    }
}


float getSpikingCost(){
    float cost = 0.0;
    for(int i = 0; i < (int)spiking_que.size(); i++){
        if(spiking_que[i]->m_name == "output" || spiking_que[i]->m_type == std::string("SOFTMAXSPIKING")){
            LayerBase* layer = (LayerBase*)Layers::instance()->get(spiking_que[i]->m_name);
            layer->calCost();
            cost += layer->getCost();
        }
    }
    return cost;
}

void cuTrainSpikingNetwork(cuMatrixVector<bool>&x,
        cuMatrix<int>*y,
        cuMatrixVector<bool>&testX,
        cuMatrix<int>* testY,
        int batch,
        int nclasses,
        std::vector<float>&nlrate,
        std::vector<float>&nMomentum,
        std::vector<int>&epoCount,
        cublasHandle_t handle)
{
    char logStr[1024];
    if(nlrate.size() != nMomentum.size() || nMomentum.size() != epoCount.size() || nlrate.size() != epoCount.size())
    {
        printf("nlrate, nMomentum, epoCount size not equal\n");
        exit(0);
    }

    if(Config::instance()->getIsGradientChecking())
        gradientChecking(x.m_devPoint, y->getDev(), batch, nclasses, handle);

    float my_start = (float)clock();
    predictTestRate(x, y, testX, testY, batch, nclasses, handle);
    float my_end = (float)clock();
    sprintf(logStr, "===================output fire counts================\n");
    LOG(logStr, "Result/log.txt");
    y->toCpu();
    printf("The last test sample has label: %d\n", testY->get(testY.size() - batch, 0, 0));
    for(int i = 0; i < (int)spiking_que.size(); i++){
        SpikingLayerBase* layer = (SpikingLayerBase*) Layers::instance()->get(spiking_que[i]->m_name);
        layer->printFireCount();
    }


    sprintf(logStr, "time spent on test : time=%.03lfs\n", (float) (my_end - my_start) / CLOCKS_PER_SEC);
    LOG(logStr, "Result/log.txt");

    if(Config::instance()->getIsGradientChecking())
        verifyResult(std::string("train"));

    sprintf(logStr, "correct is %d\n", cuSCorrect->get(0,0,0));
    LOG(logStr, "Result/log.txt");

    int epochs = Config::instance()->getTestEpoch();

    float lrate = 0.05f;
    float Momentum = 0.9f;
    int id = 0;
    cudaProfilerStart();
    for (int epo = 0; epo < epochs; epo++) {
        if (id >= (int)nlrate.size())
            break;
        lrate = nlrate[id];
        Momentum = nMomentum[id];
        Config::instance()->setLrate(lrate);
        Config::instance()->setMomentum(Momentum);

        float start, end;
        start = (float)clock();

        Config::instance()->setTraining(true);

        x.shuffle(5000, y, cuSampleWeight);

        DataLayerSpiking *dl = static_cast<DataLayerSpiking*>(Layers::instance()->get("data"));
        dl->getBatchSpikesWithStreams(x, 0);

        cuSTrVote->gpuClear();
        float cost = 0.0f;
        for (int k = 0; k < ((int)x.size() + batch - 1) / batch; k ++) {
            dl->synchronize();
            int start = k * batch;
            printf("train %2d%%", 100 * start / (((int)x.size() + batch - 1)));

            if(start + batch <= (int)x.size() - batch)
                dl->getBatchSpikesWithStreams(x, start + batch);
            else{
                int tstart = x.size() - batch;
                dl->getBatchSpikesWithStreams(x, tstart);
            }
            if(start + batch > (int)x.size()){
                start = (int)x.size() - batch;   
            }
 
            dl->trainData();
            getSpikingNetworkCost(
                y->getDev() + start, 
                cuSampleWeight->getDev() + start, 
                cuSTrVote->getDev() + start * nclasses, 
                k * batch - start);
            cost += getSpikingCost();
            printf("\b\b\b\b\b\b\b\b\b");
        }
        cost /= (float)x.size();

        end = (float)clock();
        sprintf(logStr, "epoch=%d time=%.03lfs cost=%f Momentum=%.06lf lrate=%.08lf\n",
                epo, (float) (end - start) / CLOCKS_PER_SEC,
                cost,
                Config::instance()->getMomentum(), Config::instance()->getLrate());
        LOG(logStr, "Result/log.txt");

        cuSTrCorrect->gpuClear();
        cuSTrPredictions->gpuClear();
        g_getSpikeVotingResult<<<dim3((x.size() + batch - 1) / batch), dim3(batch)>>>(
                cuSTrVote->getDev(),
                y->getDev(),
                cuSTrCorrect->getDev(),
                cuSTrPredictions->getDev(),
                x.size(),
                nclasses);
        cudaStreamSynchronize(0);
        getLastCudaError("g_getSpikeVotingResult");
        cuSTrCorrect->toCpu();

        sprintf(logStr, "train performance: %.2lf%%\n", 100.0 * cuSTrCorrect->get(0, 0, 0) / x.size());
        LOG(logStr, "Result/log.txt");
    
        if (Config::instance()->hasBoostWeightTrain()) {               
            g_boostWeightUpdate<<<dim3(1), dim3(nclasses), sizeof(float) * 2 * nclasses>>>(
                cuSampleWeight->getDev(), 
                cuSTrPredictions->getDev(), 
                y->getDev(), 
                x.size(), 
                nclasses);
            cudaStreamSynchronize(0);
            getLastCudaError("g_getSpikeVotingResult");
            cuSampleWeight->toCpu();
        }

        if (epo && epo % epoCount[id] == 0) {
            id++;
        }

        sprintf(logStr, "===================weight value================\n");
        LOG(logStr, "Result/log.txt");
        for(int i = 0; i < (int)spiking_que.size(); i++){
            LayerBase* layer = Layers::instance()->get(spiking_que[i]->m_name);
            layer->printParameter();
        }

        sprintf(logStr, "===================test Result================\n");
        LOG(logStr, "Result/log.txt");
        predictTestRate(x, y, testX, testY, batch, nclasses, handle);

        if(Config::instance()->getIsGradientChecking())
            verifyResult(std::string("test"));

        sprintf(logStr, "test %.2lf%%/%.2lf%%\n", 100.0 * cuSCorrect->get(0, 0, 0) / testX.size(),
                100.0 * cuSCurCorrect / testX.size());
        LOG(logStr, "Result/log.txt");

        sprintf(logStr, "===================output fire counts================\n");
        LOG(logStr, "Result/log.txt");
        testY->toCpu();
        printf("First test sample has label: %d\n", testY->get(0, 0, 0));
        for(int i = 0; i < (int)spiking_que.size(); i++){
            SpikingLayerBase* layer = (SpikingLayerBase*) Layers::instance()->get(spiking_que[i]->m_name);
            layer->printFireCount();
        }


        if(epo == 0){
            MemoryMonitor::instance()->printCpuMemory();
            MemoryMonitor::instance()->printGpuMemory();
        }
    }
    cudaProfilerStop();
}

