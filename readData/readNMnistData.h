#ifndef _READ_NMNIST_DATA_H_
#define _READ_NMNIST_DATA_H_

#include "../common/cuMatrix.h"
#include "../common/cuMatrixVector.h"
#include "../common/util.h"
#include "../common/MemoryMonitor.h"
#include <string>
#include <vector>
#include <utility>


//* read trainning data and lables
int readNMnistData(cuMatrixVector<bool> &x,
	cuMatrix<int>* &y, 
	std::string path,
	int samples_per_class,
	int input_neurons,
    int end_time,
    bool has_mark_test);

//* read the labels
int readNMnistLabel(const std::vector<int>& labels, cuMatrix<int>* &mat);

//* read the samples and label (encoded in the directory)
int readNMnist(std::string path, std::vector<std::pair<cuMatrix<bool>*, int> >& x, int num, int input_neurons, int end_time, bool need_shuffle);

//* read each nmnist file
void read_each_nmnist(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols);

//* read each nmnist file into the spike times
void read_each_nmnist_inside(const std::string& filename, std::vector<std::pair<cuMatrix<bool>* , int> >& x, int end_time, int input_neurons, int cur_label, std::vector<int>& quota);


//* read the data from the given directory
void file_finder(const std::string& path, std::vector<std::pair<cuMatrix<bool>*, int> >& x, int sample_per_class, int end_time, int input_neurons);

#endif
