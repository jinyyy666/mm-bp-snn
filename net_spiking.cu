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
#include <queue>
#include <set>


int cuSCurCorrect;
cuMatrix<int>*cuSCorrect = NULL;
cuMatrix<int>*cuSVote = NULL;
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
	}

}

void cuFreeSNNMemory(
        int batch,
        cuMatrixVector<bool>&trainX, 
        cuMatrixVector<bool>&testX)
{
}

void getSpikingNetworkCost(int* y)
{
    /*feedforward*/
    for(int i = 0; i < (int)spiking_que.size(); i++){
        if(spiking_que[i]->m_name == std::string("output")){
            Spiking* output = (Spiking*)Layers::instance()->get(spiking_que[i]->m_name);
            output->setPredict(y);
        }
    }


    for(int i = 0; i < (int)spiking_que.size(); i++){
        LayerBase* layer = Layers::instance()->get(spiking_que[i]->m_name);
        layer->feedforward();
    }

    /*backpropagation*/
    for(int i = (int)spiking_que.size() - 1; i >=0; i--){
        ConfigBase* top = spiking_que[i];
        if(top->m_name == std::string("reservoir")) continue;

        LayerBase* layer = Layers::instance()->get(top->m_name);

        layer->backpropagation();
        layer->getGrad();
        layer->updateWeight();
    }
    cudaStreamSynchronize(Layers::instance()->get_stream());
    getLastCudaError("updateWB");
}

/*
 * Get the network prediction result
 * block = dim3(1)
 * thread = dim3(batch)
 */
__global__ void g_getPredict(int* batchfireCount, int cols,  int* vote)
{
    int batchid = threadIdx.x;
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

void resultPredict(int* vote)
{
    /*feedforward*/
    for(int i = 0; i < (int)spiking_que.size(); i++){
        LayerBase* layer = Layers::instance()->get(spiking_que[i]->m_name);
        layer->feedforward();
    }

    for(int i = 0; i < (int)spiking_que.size(); i++){
        if(spiking_que[i]->m_name == std::string("output")){
            g_getPredict<<<dim3(1), Config::instance()->getBatchSize()>>>(
                    Layers::instance()->get(spiking_que[i]->m_name)->getFireCount()->getDev(),
                    Layers::instance()->get(spiking_que[i]->m_name)->getFireCount()->cols,
                    vote);
            cudaStreamSynchronize(0);
            getLastCudaError("g_getPredict");
        }
    }
}

void gradientChecking(bool**x, int*y, int batch, int nclasses, cublasHandle_t handle)
{
}

/*
 * block = (testX.size() + batch - 1) / batch
 * thread = batch
 */
void __global__ g_getSpikeVotingResult(int* voting, int* y, int* correct, int len, int nclasses)
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
            }
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
        printf("train %2d%%", 100 * start / (((int)testX.size() + batch - 1)));

        if(start + batch <= (int)testX.size() - batch)
            dl->getBatchSpikesWithStreams(testX, start + batch);
        else{
            int tstart = testX.size() - batch;
            dl->getBatchSpikesWithStreams(testX, tstart);
        }

        dl->testData();
        resultPredict(cuSVote->getDev() + start * nclasses);
        printf("\b\b\b\b\b\b\b\b\b");
    }

    cuSCorrect->gpuClear();
    g_getSpikeVotingResult<<<dim3((testX.size() + batch - 1) / batch), dim3(batch)>>>(
            cuSVote->getDev(),
            testY->getDev(),
            cuSCorrect->getDev(),
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
        if(spiking_que[i]->m_name != "output")  continue;
        LayerBase* layer = (LayerBase*)Layers::instance()->get(spiking_que[i]->m_name);
        layer->calCost();
        layer->printCost();
        cost += layer->getCost();
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

        x.shuffle(5000, y);

        DataLayerSpiking *dl = static_cast<DataLayerSpiking*>(Layers::instance()->get("data"));
        dl->getBatchSpikesWithStreams(x, 0);

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
 
            dl->trainData();
            getSpikingNetworkCost(y->getDev() + start);
            printf("\b\b\b\b\b\b\b\b\b");
        }

        float cost = getSpikingCost();

        end = (float)clock();
        sprintf(logStr, "epoch=%d time=%.03lfs cost=%f Momentum=%.06lf lrate=%.08lf\n",
                epo, (float) (end - start) / CLOCKS_PER_SEC,
                cost,
                Config::instance()->getMomentum(), Config::instance()->getLrate());
        LOG(logStr, "Result/log.txt");

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

        if(epo == 0){
            MemoryMonitor::instance()->printCpuMemory();
            MemoryMonitor::instance()->printGpuMemory();
        }
    }
    cudaProfilerStop();
}

