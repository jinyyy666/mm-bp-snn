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
    void getDeltaVth(){};
	void updateWeight(){};
    void updateVth(){};
	void clearMomentum(){};

	void calCost(){};
	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};
    void setPredict(int * p){};

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

	void trainData();
	void testData();

	void printParameter(){};
    void printFireCount(){};
	void synchronize();

	void getBatchSpikesWithStreams(cuMatrixVector<bool>& inputs, int start);
private:
	cuMatrix<bool>* outputs;
    cuMatrix<int>* outputs_time;
    cuMatrix<int>* fireCount;
	cuMatrixVector<bool> batchSpeeches[2]; // batch size speeches,one for current read, one for buffer
    int myId; // use to asynchoronously load the data
	int batch;
	cudaStream_t stream1;
};
#endif
