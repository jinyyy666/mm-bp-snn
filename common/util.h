#ifndef __UTIL_H__
#define __UTIL_H__

#include "cuMatrix.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include <memory>
#include <iostream>
#include <vector>
#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iterator>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define MAX_THREADS 1024

// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
#define NL_LRELU 3

bool checkSharedMemory(int cardId, size_t MemorySize);
void showImg(cuMatrix<float>* x, float scala);
void DebugPrintf(cuMatrix<float>*x);
void DebugPrintf(float* data, int len, int dim);
void LOG(const char* str, const char* file);
int  getCV_64();
void createGaussian(float* gaussian, float dElasticSigma1, float dElasticSigma2,
	int rows, int cols, int channels, float epsilon);

#define  cuAssert( X ) if ( !(X) ) {printf("tid %d: FILE=%s, LINE=%d\n", threadIdx.x, __FILE__, __LINE__); return; }
#define  Assert( X )   if ( !(X) ) {printf("FILE=%s, LINE=%d\n", __FILE__, __LINE__); int* p = (int*)-1; p[0] = 1; return; }

void dropDelta(cuMatrix<float>* M, float cuDropProb);
void dropScale(cuMatrix<float>* M, float cuDropProb);
void initMatrix(cuMatrix<float>* M, float iniw);
void checkMatrixIsSame(cuMatrix<float>*x, cuMatrix<float>*y, int channel = 0);
void checkMatrixIsSame(cuMatrix<bool>*x, cuMatrix<bool>*y, int n_outputs);
int extractNeuronIndex(const std::string& name);
void convertToSpikeTimes(cuMatrix<float>* preproc, std::vector<std::vector<int> >*& sp_times, int imgSize, int end_time);

//* reload the << for print the vector
template<class T> std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
{
    if(!v.empty()){
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

void print2DVectorToFile(std::vector<std::vector<int> >& v, std::string filename);
void readSpikesFromDumpfile(const std::string& filename, cuMatrix<bool>*& x);
#endif
