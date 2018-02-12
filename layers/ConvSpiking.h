#ifndef __CONV_COMBINE_FEATURE_MAP_CU_H__
#define __CONV_COMBINE_FEATURE_MAP_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class ConvSpiking: public SpikingLayerBase
{
public:
	ConvSpiking(std::string name);
	~ConvSpiking(){
		delete outputs;
        delete outputs_time;
        delete curDelta;
        delete fireCount;
        delete weightSqSum;
        delete b1_t;
        delete b2_t;
        delete output_train_ref;
        delete output_test_ref;
	}


	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void calCost();
    void loadRef();
    void verify(const std::string& phrase);    

	void initRandom();
	void initFromCheckpoint(FILE* file);
    void initFromDumpfile(const std::string& filename, cuMatrixVector<float>& cuW);
	void save(FILE* file);

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
    
	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w[0]->toCpu();
		sprintf(logStr, "weight:%f, %f, %f;\n", w[0]->get(0,0,0), w[0]->get(0,1,0), w[0]->get(0, 2, 0));
		LOG(logStr, "Result/log.txt");
		b[0]->toCpu();
		b[1]->toCpu();
		sprintf(logStr, "bias  :%f %f\n", b[0]->get(0,0,0), b[1]->get(0, 0, 0));
		LOG(logStr, "Result/log.txt");
	}
    
    virtual void printFireCount(){
        char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		fireCount->toCpu();
		sprintf(logStr, "fire count: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d;\n", fireCount->get(0,0,0), fireCount->get(0,1,0), fireCount->get(0,2,0), fireCount->get(0,3,0), fireCount->get(0,4,0), fireCount->get(0,5,0), fireCount->get(0,6,0), fireCount->get(0,7,0), fireCount->get(0,8,0), fireCount->get(0,9,0));
		LOG(logStr, "Result/log.txt");
    }

private:
	cuMatrix<bool>* inputs;
	cuMatrix<float>* preDelta;
	cuMatrix<bool>* outputs;
	cuMatrix<float>* curDelta; // size(curDelta) == size(outputs)
    cuMatrix<int>*    inputs_time;
    cuMatrix<int>*    outputs_time;

    cuMatrix<int>*   fireCount;
    cuMatrix<int>*   preFireCount;

    cuMatrix<float>* weightSqSum;
    cuMatrix<float>* b1_t;
    cuMatrix<float>* b2_t;

	int kernelSize;
	int padding;
	int batch;
    int T_REFRAC;
    float threshold;
    float TAU_M;
    float TAU_S;
	float lambda;
    float beta;
    float weightLimit;
private:
	cuMatrixVector<float> w;
	cuMatrixVector<float> wgrad;
	cuMatrixVector<float> wgradTmp;
	cuMatrixVector<float> b;
	cuMatrixVector<float> bgrad;
	cuMatrixVector<float> momentum_w;
	cuMatrixVector<float> momentum_b;
	cuMatrixVector<float> g1_w;
	cuMatrixVector<float> g1_b;
	cuMatrixVector<float> g2_w;
	cuMatrixVector<float> g2_b;
    
    cuMatrixVector<float> w_ref;
    cuMatrix<bool>* output_train_ref;
    cuMatrix<bool>* output_test_ref;
};

#endif 
