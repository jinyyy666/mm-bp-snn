#ifndef __LAYERS_POOLING_H__
#define __LAYERS_POOLING_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"


class PoolingSpiking: public SpikingLayerBase{
public:
	PoolingSpiking(std::string name);
	~PoolingSpiking(){
		delete outputs;
        delete outputs_time;
        delete inputs_resp;
        delete curDelta;
        delete fireCount;
	}


	void feedforward();
	void backpropagation();
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};
	void calCost(){};
    void loadRef();
    void verify(const std::string& phrase);

	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

    cuMatrix<int>* getFireCount(){
        return fireCount;
    }

	cuMatrix<float>* getOutputs(){return NULL;}

    cuMatrix<bool>* getSpikingOutputs(){
        return outputs;
    }

    cuMatrix<int>*  getSpikingTimeOutputs(){
        return outputs_time;
    }

	cuMatrix<float>* getCurDelta(){
		return curDelta;
	}

	int getOutputAmount(){
		return outputAmount;
	}

	int getInputDim(){
		return inputDim;
	}

	int getOutputDim(){
		return outputDim;
	}

    void setPredict(int* p){}
    void setSampleWeight(float* s_weights){}

	void printParameter(){}
    virtual void printFireCount(){
        char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		fireCount->toCpu();
		sprintf(logStr, "fire count: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d;\n", fireCount->get(0,0,0), fireCount->get(0,1,0), fireCount->get(0,2,0), fireCount->get(0,3,0), fireCount->get(0,4,0), fireCount->get(0,5,0), fireCount->get(0,6,0), fireCount->get(0,7,0), fireCount->get(0,8,0), fireCount->get(0,9,0));
		LOG(logStr, "Result/log.txt");
    }

private:
	cuMatrix<bool>*  inputs;
	cuMatrix<float>* preDelta;
	cuMatrix<bool>*  outputs;
    cuMatrix<float>* inputs_resp;
	cuMatrix<float>* curDelta; // size(curDelta) == size(outputs)
    cuMatrix<int>*    inputs_time;
    cuMatrix<int>*    outputs_time;

    cuMatrix<int>*   fireCount;
    cuMatrix<int>*   preFireCount;

	int psize;
	int pskip;
	int batch;
    int T_REFRAC;
    float threshold;
    float TAU_M;
    float TAU_S;

    cuMatrix<bool>* output_train_ref;
    cuMatrix<bool>* output_test_ref;

};
#endif
