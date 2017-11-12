#ifndef _READ_NMNIST_DATA_H_
#define _READ_NMNIST_DATA_H_

#include "../common/cuMatrix.h"
#include "../common/cuMatrixVector.h"
#include "../common/util.h"
#include "../common/MemoryMonitor.h"
#include <string>
#include <vector>


//* read trainning data and lables
int readNMnistData(cuMatrixVector<bool> &x,
	cuMatrix<int>* &y, 
	std::string path,
	int number_of_images,
	int input_neurons,
    int end_time);

//* read the labels
int readNMnistLabel(const std::vector<int>& labels, cuMatrix<int>* &mat);

//* read the samples and label (encoded in the directory)
int readNMnist(std::string path, cuMatrixVector<bool>& x, std::vector<int>& labels, int num, int input_neurons, int end_time);

//* read each mnist file
void read_each_nmnist(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols);

//* read the given directory recursively
void file_finder(const std::string& path, cuMatrixVector<bool>& x, std::vector<int>& labels, int cur_label, int& sample_count, int num_of_samples, int end_time, int input_neurons);

#endif
