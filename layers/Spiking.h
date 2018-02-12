#ifndef __SPIKING_CU_H__
#define __SPIKING_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class Spiking: public SpikingLayerBase
{
public:
	Spiking(std::string name);
	~Spiking(){
        delete inputs_resp;
        delete inputs_float;
        delete inputs_resp_tmp;
        delete inputs_time_format;
		delete outputs;
        delete curDelta;
        delete preDelta_format;
        delete preFireCount_format;
        delete fireCount;
        delete accEffect;
        delete weightSqSum;
        delete lateralFactor;
        delete effectRatio;
        delete maxCount;
        delete groundTruth;
        delete w;
        delete b;
        delete wgrad;
        delete bgrad;
        delete wgradTmp;
        delete bgradTmp;
        delete w_laterial;
        delete momentum_w;
        delete momentum_b;
        delete g1_w;
        delete g1_b;
        delete g2_w;
        delete g2_b;
	}


	void feedforward();
	void backpropagation();
    void verify(const std::string& phrase);
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void calCost();
    void loadRef();

	void initRandom();
    void initReservoirConnection(const std::vector<int>& reservoirDim);
    void initLaterial();
    void initLocalInhibition(float strength);
	void initFromCheckpoint(FILE* file);
    void initBiasFromDumpfile(const std::string& filename, cuMatrix<float>*& cuW);
    void initFromDumpfile(const std::string& filename, cuMatrix<float>*& cuW);
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

	int getInputSize(){
		return inputSize;
	}

	int getOutputSize(){
		return outputSize;
	}
    
    void setPredict(int* p){
        predict = p;
    }
    
    void setSampleWeight(float* s_weights){
        sample_weights = s_weights;
    }

	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w->toCpu();
		sprintf(logStr, "weight:%f, %f, %f;\n", w->get(0,0,0), w->get(0,1,0), w->get(0, 2, 0));
		LOG(logStr, "Result/log.txt");
        
		b->toCpu();
		sprintf(logStr, "bias  :%f\n", b->get(0,0,0));
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
	cuMatrix<bool>*   inputs;
	cuMatrix<float>*  preDelta;
	cuMatrix<float>*  preDelta_format; //preDelta(batch, size, channel) --> (batch, size * channel)
	cuMatrix<bool>*   outputs;
	cuMatrix<float>*  curDelta; // size(curDelta) == size(fireCount)
    cuMatrix<int>*    inputs_time;
    cuMatrix<int>*    inputs_time_format;
    cuMatrix<int>*    outputs_time;
    cuMatrix<float>*  inputs_resp;
    cuMatrix<float>*  inputs_resp_tmp;
    cuMatrix<float>*  inputs_float;

    cuMatrix<int>*   fireCount;
    cuMatrix<int>*   maxCount;
    cuMatrix<float>* groundTruth;
    cuMatrix<int>*   preFireCount;
    cuMatrix<int>*   preFireCount_format; //preFireCount(batch, size, channel)->(batch, size*channel)
    cuMatrix<float>* accEffect;

    cuMatrix<float>* weightSqSum;
    cuMatrix<float>* lateralFactor;
    cuMatrix<float>* effectRatio;

    int * predict;
    float * sample_weights;
    int inputSize;
    int outputSize;
	int batch;
    int T_REFRAC;
    float threshold;
    float TAU_M;
    float TAU_S;
	float lambda;
    float beta;
    float weightLimit;
    float lateralW;
    float UNDESIRED_LEVEL;
    float DESIRED_LEVEL;
    float MARGIN;
private:
	cuMatrix<float>* w;
	cuMatrix<float>* wgrad;
	cuMatrix<float>* wgradTmp;
    cuMatrix<float>* w_laterial;
	cuMatrix<float>* b;
	cuMatrix<float>* bgrad;
    cuMatrix<float>* bgradTmp;
	cuMatrix<float>* momentum_w;
	cuMatrix<float>* momentum_b;
	cuMatrix<float>* g1_w;
	cuMatrix<float>* g1_b;
	cuMatrix<float>* g2_w;
	cuMatrix<float>* g2_b;
    float b1_t;
    float b2_t;

    cuMatrix<float>* w_ref;
    cuMatrix<float>* w_laterial_ref;
    cuMatrix<float>* b_ref;
    cuMatrixVector<bool>   output_train_ref;
    cuMatrixVector<bool>   output_test_ref;
};

#endif 
