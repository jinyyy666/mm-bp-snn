#ifndef __CU_BASE_CU_H__
#define __CU_BASE_CU_H__

#include <helper_functions.h>
#include <helper_cuda.h>
#include "util.h"

__device__ float d_nonLinearity(float val, int NONLIN);
__device__ float d_dnonLinearity(float val,int NONLIN);
__device__ float d_Spiking_accumulate_effect(int* output_time, int* input_time, int n_ospikes, int n_ispikes, int o_idx, int i_idx, int outputDim, int inputDim, int endTime, int T_REFRAC, float TAU_M, float TAU_S);
__device__ void   swap(float& val1, float& val2);


__global__ void   g_dnonLinearity(float* delta, float*acti, int len, int NONLIN);
__global__ void   g_nonLinearity(float* inputs, int len, int NONLIN);

__global__ void g_vecAdd(float**_v_w, float** _wgrad,float** _w,
	float** _v_b, float** _bgrad, float** _b, 
	int lenw, int lenb,
	float momentum, float lratew, float lrateb);

__global__ void g_vecAdd(float*v_w, float*wgrad,float* w,
	float* v_b, float* bgrad, float* b, 
	int lenw, int lenb,
	float momentum, float lratew, float lrateb);

__global__ void g_getBgrad(float* softMaxDelta, float* bgrad, float* dropb, int batch);
__global__ void g_getBgrad(float* softMaxDelta, float* bgrad, int batch);

__global__ void g_sgd_vecAdd(float** v_m, float** wgrad, float** w, int lenw, float momentum, float lr);
__global__ void g_adam_vecAdd(float** g1_ws, float** g2_ws, float* b1_t, float* b2_t, float** _wgrad, float** _w, int lenw, float lr);

// Use these two functions when outputAmount = 1
__global__ void g_sgd_vecAdd(float* v_w, float* wgrad, float* w, int lenw, float momentum, float lr);
__global__ void g_adam_vecAdd(float* g1_w, float* g2_w, float b1t, float b2t, float* wgrad, float* w, int lenw, float lr);

__global__ void g_getCost_3(float* cost,
	float** weight,
	float lambda, int wlen);

__global__ void g_getCost_2(float* cost,
	float* weight,
	float lambda, int len);

__global__ void g_getCost_1(float* softMaxP,
	float* groundTruth, float* cost, int*y, int rows, int cols, int batch);

/*
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
* blocks  : dim3(batch)
* threads : dim3(min(512, cuPool[poolidx]->cols))
*/
__global__ void g_convert(float* cuPool, float*cuPoolToFlActi, int batch, int size, int channel);


/* function: cuMatrix<int>*(batch, inputDim2*endTime, amount) 
 *           to cuMatrix<int>*(batch, amount*inputDim2*endTime, 1)
 */
__global__ void g_convert_spiketimes(int* inputs_time, int endTime, int inputSize, int inputCols, int channels, int batch, int* inputs_tf);
__global__ void g_convert_firecounts(int* counts, int area, int inputSize, int inputDim2, int channels, int batch, int* counts_f);


/*
* function: cuMatrix<bool>*(batch, endTime*inputDim*inputDim, amount) 
*           to cuMatrix<float>*(inputSize, endTime*batch, 1)
* blocks  : dim3(batch, endTime)
* threads : dim3(min(1024, inputSize))
*/
__global__ void g_cast_bool_2_float(bool* inputs, int endTime, int inputSize, int inputCols, int channels, int batch, float* inputs_f);


/*
* function: cuMatrix<float>*(outputSize, endTime*batch) to cuMatrix<float>*(batch, outputSize*endTime)
* blocks  : dim3(batch, outputSize)
* threads : dim3(min(1024, endTime))
*/
__global__ void g_transform_2_batch(float* inputs_rt, int endTime, int outputSize, int batch, float* inputs_r);

/*
function: g_preDeltaFormat
threads : <<<dim3(batch), dim3(512)>>> 
*/
__global__ void g_preDeltaFormat(float* cuPoolFlDelta, 
	float* cuPoolDelta, int batch, int size, int channels);

/*
function: transform the binary response matrix to the spike times
threads : <<<dim3(batch), dim3(min(outputDim, 1024))>>>
*/
__global__ void g_response_2_spiketime(bool* outputs, int* outputs_time, int outputArea, int ouputDim, int endTime);

__global__ void g_divide_by_threshold(float * _delta, int area, int outputSize, float threshold);

__global__ void g_intrinsic_plasticity(int * batchFireCount, float* tauTmp, float* resTmp, float* _tau, float * _res, int endTime, int outputArea, int outputSize, int T_REFRAC, float vth, float u);

__global__ void g_intrinsic_plasticity_gradadd(float* taugradTmp, float* taugrad, float* resgradTmp, float* resgrad, int batch, int outputArea, int outputSize);

__global__ void g_intrinsic_plasticity_update(float* taugrad, float* resgrad, float* tau, float* res, int len, float lr);
/*
function: normalize the fire counts by the max count for SNN
threads : <<<dim3(batch), dim3(min(1024, inputDim))>>>
*/
__global__ void g_normalize_fireCount(int * inputs, float * inputs_float, int rows, int cols);

/*
function: compute the softmax prob of the softmax layer
threads : <<<dim3(batch), dim3(min(512, outputDim))>>>
*/
__global__ void g_getSoftMaxP(float* softMaxP, float* b, int cols);

__global__ void g_getSoftMaxDelta(float* softMaxDelta, float* softMaxP, float* groundTruth, int len);

__global__ void g_getSmrWgrad(float* wgrad, float* weight, float lambda, int len, int batch);
#endif


