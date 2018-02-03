#ifndef __SOFT_MAX_SPIKING_H__
#define __SOFT_MAX_SPIKING_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"

class SoftMaxSpiking: public SpikingLayerBase
{
public:
	SoftMaxSpiking(std::string name);
    ~SoftMaxSpiking(){
        delete inputs_float;
        delete outputs;
        delete curDelta;
        delete groundTruth;
    }

	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight() ;
	void clearMomentum();
	void calCost();

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);

    cuMatrix<float>* getOutputs(){return outputs;}
	cuMatrix<float>* getCurDelta(){return curDelta;}

    cuMatrix<bool>* getSpikingOutputs(){return NULL;}
    cuMatrix<int>* getSpikingTimeOutputs(){return NULL;}
    cuMatrix<int>* getFireCount(){return NULL;}
    
    void verify(const std::string& phrase){};

	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w->toCpu();
		sprintf(logStr, "weight:%f, %f;\n", w->get(0,0,0), w->get(0,1,0));
		LOG(logStr, "Result/log.txt");
		b->toCpu();
		sprintf(logStr, "bias  :%f\n", b->get(0,0,0));
		LOG(logStr, "Result/log.txt");
	}

	void setPredict(int* p){
		predict = p;
	}

    void setSampleWeight(float* s_weights){
        sample_weights = s_weights;
    }

    void printFireCount(){
        char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		outputs->toCpu();
		sprintf(logStr, "softmax output: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f;\n", outputs->get(0,0,0), outputs->get(0,1,0), outputs->get(0,2,0), outputs->get(0,3,0), outputs->get(0,4,0), outputs->get(0,5,0), outputs->get(0,6,0), outputs->get(0,7,0), outputs->get(0,8,0), outputs->get(0,9,0));
		LOG(logStr, "Result/log.txt");
    }

private:
	cuMatrix<int>*   inputs;
	cuMatrix<float>* outputs;
	cuMatrix<float>* inputs_float;//the normalized (by max count) fire counts
	cuMatrix<float>* curDelta;
	cuMatrix<float>* preDelta;

	cuMatrix<float>* w;
	cuMatrix<float>* wgrad;
	cuMatrix<float>* b;
	cuMatrix<float>* bgrad;
	cuMatrix<float>* groundTruth;

	cuMatrix<float>* momentum_w;
	cuMatrix<float>* momentum_b;
	int* predict;
    float* sample_weights;

	float lambda;
	int batch;
};
#endif
