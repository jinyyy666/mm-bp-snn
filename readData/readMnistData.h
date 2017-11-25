#ifndef _READ_MNIST_DATA_H_
#define _READ_MNIST_DATA_H_

#include "../common/cuMatrix.h"
#include "../common/cuMatrixVector.h"
#include "../common/util.h"
#include "../common/MemoryMonitor.h"
#include <string>
#include <vector>


/*read trainning data and lables*/
int readMnistData(cuMatrixVector<float> &x,
	cuMatrix<int>* &y, 
	std::string xpath,
	std::string ypath,
	int number_of_images,
	int flag);

/*read the MNIST and produce the poisson spike trains*/
int readSpikingMnistData(
        cuMatrixVector<bool>& x,
        cuMatrix<int>*& y, 
        std::string xpath,
        std::string ypath,
        int number_of_images,
        int input_neurons,
        int end_time);
#endif
