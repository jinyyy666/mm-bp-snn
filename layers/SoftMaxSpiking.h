#ifndef __SOFTMAXSPIKING_CU_H__
#define __SOFTMAXSPIKING_CU_H__

#include "LayerBase.h"
#include "Spiking.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class SoftMaxSpiking: public Spiking
{
public:
	SoftMaxSpiking(std::string name);

	void feedforward();
	void backpropagation();
    void calCost();
    cuMatrix<float>* getOutputs(){return softMaxP;}
private:
    cuMatrix<float>* softMaxP; // the softmax output given the firing counts
};

#endif 
