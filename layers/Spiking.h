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
        delete inputs;
		delete outputs;
        delete curDelta;
        delete fireCount;
        delete maxCount;
        delete groundTruth;
	}


	void feedforward();
	void backpropagation();
    void verify(const std::string& phrase);
	void getGrad();
    void getDeltaVth();
	void updateWeight();
    void updateVth();
	void clearMomentum();
	void calCost();
    void loadRef();

	void initRandom();
    void initReservoirConnection(const std::vector<int>& reservoirDim);
    void initLaterial();
    void initLocalInhibition(float strength);
	void initFromCheckpoint(FILE* file);
    void initBiasFromDumpfile(const std::string& filename, cuMatrixVector<float>& cuW);
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
    
    void setPredict(int* p){
        predict = p;
    }

	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w[0]->toCpu();
		sprintf(logStr, "weight:%f, %f, %f;\n", w[0]->get(0,0,0), w[0]->get(0,1,0), w[0]->get(0, 2, 0));
		LOG(logStr, "Result/log.txt");
        
		b[0]->toCpu();
		sprintf(logStr, "bias  :%f\n", b[0]->get(0,0,0));
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
	cuMatrix<bool>*   outputs;
	cuMatrix<float>*  curDelta; // size(curDelta) == size(fireCount)
    cuMatrix<int>*    inputs_time;
    cuMatrix<int>*    outputs_time;

    cuMatrix<int>*   fireCount;
    cuMatrix<int>*   maxCount;
    cuMatrix<float>* groundTruth;
    cuMatrix<int>*   preFireCount;

    cuMatrix<float>* vth;
    cuMatrix<float>* vthDelta;
    cuMatrix<float>* vthDeltaTmp;

    int * predict;
	int batch;
    int T_REFRAC;
    float threshold;
    float TAU_M;
    float TAU_S;
	float lambda;
    float UNDESIRED_LEVEL;
    float DESIRED_LEVEL;
    float MARGIN;
private:
	cuMatrixVector<float> w;
	cuMatrixVector<float> wgrad;
	cuMatrixVector<float> wgradTmp;
    cuMatrixVector<float> w_laterial;
	cuMatrixVector<float> b;
	cuMatrixVector<float> bgrad;
    cuMatrixVector<float> bgradTmp;
	cuMatrixVector<float> momentum_w;
	cuMatrixVector<float> momentum_b;

    cuMatrixVector<float> w_ref;
    cuMatrixVector<float> w_laterial_ref;
    cuMatrixVector<float> b_ref;
    cuMatrixVector<bool>   output_train_ref;
    cuMatrixVector<bool>   output_test_ref;
};

#endif 
