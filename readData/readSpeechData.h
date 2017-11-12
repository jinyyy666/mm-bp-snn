#ifndef _READ_SPEECH_DATA_H_
#define _READ_SPEECH_DATA_H_

#include "../common/cuMatrix.h"
#include "../common/cuMatrixVector.h"
#include "../common/util.h"
#include "../common/MemoryMonitor.h"
#include <string>
#include <vector>

//* read each speech file
void read_each_speech(const std::string& filename, cuMatrixVector<int>& x, int nrows, int ncols);

//* read each speech from the dump file
void read_each_speech_dump(const std::string& filename, cuMatrixVector<int>& x, int nrows, int ncols);

//* read training data and lables
int readSpeechData(cuMatrixVector<bool> &x,
	cuMatrix<int>* &y, 
	std::string path,
	int number_of_speeches,
	int input_neurons,
    int max_spikes,
    int CLS,
    bool is_dump);

//* read the labels
int readSpeechLabel(const std::vector<int>& labels, cuMatrix<int>* &mat);

//* read the samples and label (encoded in the directory)
int readSpeech(std::string path, cuMatrixVector<int>& x, std::vector<int>& labels, int num, int input_neurons, int max_spikes, int CLS, bool is_dump);

#endif
