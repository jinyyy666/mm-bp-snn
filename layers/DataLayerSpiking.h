/*
ref : ImageNet Classification with Deep Convolutional Neural Networks
*/
#ifndef __LAYERS_DATA_LAYER_SPIKING_H__
#define __LAYERS_DATA_LAYER_SPIKING_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include <map>
//#include <thread>
#include "../common/util.h"


class DataLayerSpiking: public SpikingLayerBase{
public:
	DataLayerSpiking(std::string name);

	void feedforward(); /*only copy the input to output*/
	void backpropagation(){};
    void verify(const std::string& phrase){};
	void getGrad(){};
	void updateWeight(){};
    void intrinsicPlasticity(){};
	void clearMomentum(){};

	void calCost(){};
	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};
    void setPredict(int * p){};
    void setSampleWeight(float * s_weights){};

	~DataLayerSpiking(){
		delete outputs;
		checkCudaErrors(cudaStreamDestroy(stream1));
	}

	cuMatrix<bool>* getSpikingOutputs(){return outputs;}
    cuMatrix<int>*  getSpikingTimeOutputs(){return outputs_time;}

    cuMatrix<float>* getOutputs(){return NULL;}
	cuMatrix<float>* getCurDelta(){return NULL;}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}
    
    cuMatrix<int>* getFireCount(){
        return fireCount;
    }

	void getBatchSpikesWithPreproc(cuMatrixVector<bool>& inputs, int start);
	void testData();

	void printParameter(){};
    void printFireCount(){};
	void synchronize();

    void generateRandom(unsigned long long seed);
	void getBatchSpikes(cuMatrixVector<bool>& inputs, int start);
	void loadBatchSpikes(cuMatrixVector<bool>& inputs, int start);
private:
	cuMatrix<bool>* outputs;
    cuMatrix<int>* outputs_time;
    cuMatrix<int>* fireCount;
    cuMatrix<float>* cu_randomNum;
	cuMatrixVector<bool> batchSpeeches[2]; // batch size speeches,one for current read, one for buffer
	cuMatrixVector<float> batchSamplesFloat[2]; //raw float samples, the original one before pre-processing
    cuMatrixVector<float> processOutputs; // the raw samples after pre-processing (distortion)
    int myId; // use to asynchoronously load the data
	int batch;
    int inputSize;
    int outputSize; 
	cudaStream_t stream1;
};
#endif
