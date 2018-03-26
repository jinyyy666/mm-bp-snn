#include "cuBase.h"

__device__ float d_nonLinearity(float val, int NONLIN){
	if(NONLIN == NL_RELU){
		if(val < 0.0) return 0.0;
		else return val;
	}else if(NONLIN == NL_LRELU){
        if(val < 0.0) return 0.1f * val;
        else return val;
    }else if(NONLIN == NL_TANH){
		return tanh(val * 2.0 / 3.0) * 1.7159;
	}
	else{
		return val;
	}
}

__device__ float d_dnonLinearity(float val,int NONLIN){
	if(NONLIN == NL_RELU){
		if(val > 0.0) return 1.0;
		else return 0.0;
	}else if (NONLIN == NL_LRELU){
        if(val > 0.0) return 1.0;
        else return 0.1;
    }
	else if(NONLIN == NL_TANH){
		float res = 1.7159;
		float temp = val * val / 1.7159;
		res = (res - temp) * 2.0 / 3.0;
		return res;
	}else {
		return val;
	}
}

/* given each input and output spike train of spike times, 
 * compute the accumulative synaptic effect
 * input: input spikes: endTime * inputDim
 * output: output spikes: endTime * outputDim
 */
__device__ float d_Spiking_accumulate_effect(
    int* output_time,
    int* input_time,
    int n_ospikes,
    int n_ispikes,
    int o_idx,
    int i_idx,
    int outputDim,
    int inputDim,
    int endTime,
    int T_REFRAC,
    float TAU_M,
    float TAU_S)
{
    float acc_response = 0.0f;
    int t_post_last = 1;
    for(int i = 0; i < n_ospikes; ++i){
        int t_post = output_time[o_idx * endTime + i];
        float sum = 0.0f;
        
        int ub = t_post;
        int lb = max(1, int(t_post - 4*TAU_M));
        for(int j = 0; j < n_ispikes; ++j){
            int t_pre = input_time[i_idx * endTime + j];
            if(t_pre < lb || t_pre >= ub)    continue;

            int pre_time = t_pre + T_REFRAC;
            if(pre_time > t_post)   continue;
            int s = t_post - t_post_last;
            int t = t_post - pre_time;
            float factor = __expf(-1*max(t - s, 0)/TAU_S)/(1 - TAU_S/TAU_M);
            sum += factor * (__expf(-1*min(s, t)/TAU_M) - __expf(-1*min(s, t)/TAU_S));
        }
        t_post_last = t_post + T_REFRAC;
        acc_response += sum;
    }
    if(n_ospikes == 0 && n_ispikes != 0)
        acc_response = 0.1;
    return acc_response;
}


__global__ void g_dnonLinearity(float* delta, float*acti, int len, int NONLIN)
{
	int skip = gridDim.x * blockDim.x;
	for(int i = 0; i < len; i += skip)
	{
		int id = blockDim.x * blockIdx.x + threadIdx.x + i;
		if(id < len)
		{	
			delta[id] *= d_dnonLinearity(acti[id], NONLIN);
		}
	}
}

__global__ void g_nonLinearity(float* inputs, int len, int NONLIN)
{
	for(int i = 0; i < len; i += gridDim.x * blockDim.x)
	{
		int id = blockDim.x * blockIdx.x + threadIdx.x + i;
		if(id < len)
		{	
			inputs[id] = d_nonLinearity(inputs[id], NONLIN);
		}
	}
}

__device__ void swap(float& val1, float& val2){
	float tmp = val1;
	val1 = val2;
	val2 = tmp;
}


__global__ void g_vecAdd(float**_v_w, float** _wgrad,float** _w,
	float** _v_b, float** _bgrad, float** _b, 
	int lenw, int lenb,
	float momentum, float lratew, float lrateb)
{
	float* v_w   = _v_w[blockIdx.x];
	float* wgrad = _wgrad[blockIdx.x];
	float* w     = _w[blockIdx.x];
	float* v_b   = _v_b[blockIdx.x];
	float* bgrad = _bgrad[blockIdx.x];
	float* b     = _b[blockIdx.x];

	int idx = threadIdx.x;
	for(int i = 0; i < lenw; i += blockDim.x)
	{
		int id = i + idx;
		if(id < lenw)
		{
			v_w[id] = v_w[id] * momentum + wgrad[id] * lratew;
			w[id] -= v_w[id];
		}
	}
	for(int i = 0; i < lenb; i += blockDim.x)
	{
		int id = i + idx;
		if(id < lenb)
		{
			v_b[id] = v_b[id] * momentum + bgrad[id] * lrateb;
			b[id] -= v_b[id];
		}
	}
}

__global__ void g_vecAdd(float*v_w, float*wgrad,float* w,
	float* v_b, float* bgrad, float* b, 
	int lenw, int lenb,
	float momentum, float lratew, float lrateb)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < lenw; i += blockDim.x * gridDim.x)
	{
		int id = i + idx;
		if(id < lenw)
		{
			v_w[id] = v_w[id] * momentum + wgrad[id] * lratew;
			w[id] -= v_w[id];
		}
	}
	for(int i = 0; i < lenb; i += blockDim.x * gridDim.x)
	{
		int id = i + idx;
		if(id < lenb)
		{
			v_b[id] = v_b[id] * momentum + bgrad[id] * lrateb;
			b[id] -= v_b[id];
		}
	}
}

/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_sgd_vecAdd(float** momentum_w, float** _wgrad, float** _w, int lenw, float momentum, float lr)
{
    int ok = blockIdx.x;
    float* v_w   = momentum_w[ok];
    float* w     = _w[ok];
    float* wgrad = _wgrad[ok];
    int idx = threadIdx.x;
    for(int i = 0; i < lenw; i += blockDim.x)
    {
        int id = i + idx;
        if(id < lenw)
        {
            v_w[id] = v_w[id] * momentum + wgrad[id] * lr;
            w[id]  -= v_w[id];
        }
    }
}


/*
 * block  = dim3(outputAmount)
 * thread = dim3(min(256, w[0]->getLen()))
 */
__global__ void g_adam_vecAdd(float** g1_ws, float** g2_ws, float* b1_t, float* b2_t, float** _wgrad, float** _w, int lenw, float lr)
{
    int ok = blockIdx.x;
    float* g1_w  = g1_ws[ok];
    float* g2_w  = g2_ws[ok];
    float* w     = _w[ok];
    float* wgrad = _wgrad[ok];
    int idx = threadIdx.x;
    float b1t = b1_t[ok];
    float b2t = b2_t[ok];
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float eps = 1.e-8f;
    __syncthreads();

    for(int i = 0; i < lenw; i += blockDim.x)
    {
        int id = i + idx;
        if(id < lenw)
        {
            float weight_grad = wgrad[id];
            float g1 = b1 * g1_w[id] + (1 - b1) * weight_grad;
            float g2 = b2 * g2_w[id] + (1 - b2) * weight_grad * weight_grad;
            w[id]  -= lr * (g1/(1.f - b1t)) / ((float)sqrtf(g2/(1. - b2t)) + eps);
            g1_w[id] = g1;
            g2_w[id] = g2;
        }
    }
    if(threadIdx.x == 0){
        b1_t[ok] *= b1;
        b2_t[ok] *= b2;
    }
}

/* Use this function when outputAmount = 1
 * block  = dim3(min((w->getLen() + 255)/256, 5120))
 * thread = dim3(256)
 */
__global__ void g_sgd_vecAdd(float* v_w, float* wgrad, float* w, int lenw, float momentum, float lr)
{
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   for(int i = 0; i < lenw; i += blockDim.x * gridDim.x)
   {
       int id = i + idx;
       if(id < lenw)
       {
           v_w[id] = v_w[id] * momentum + wgrad[id] * lr;
           w[id]  -= v_w[id];
       }
   }
}

/* Use this function when outputAmount = 1
 * block  = dim3(min((w->getLen() + 255)/256, 5120))
 * thread = dim3(256)
 */
__global__ void g_adam_vecAdd(float* g1_w, float* g2_w, float b1t, float b2t, float* wgrad, float* w, int lenw, float lr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float eps = 1.e-8f;
    
    for(int i = 0; i < lenw; i += blockDim.x * gridDim.x)
    {
        int id = i + idx;
        if(id < lenw)
        {
            float weight_grad = wgrad[id];
            float g1 = b1 * g1_w[id] + (1 - b1) * weight_grad;
            float g2 = b2 * g2_w[id] + (1 - b2) * weight_grad * weight_grad;
            w[id]  -= lr * (g1/(1.f - b1t)) / ((float)sqrtf(g2/(1. - b2t)) + eps);
            g1_w[id] = g1;
            g2_w[id] = g2;
        }
    }
}

__global__ void g_getCost_3(float* cost,
	float** weight,
	float lambda, int wlen)
{
	extern __shared__ float _sum[];
	_sum[threadIdx.x] = 0;
	__syncthreads();
	float* w = weight[blockIdx.x];

	for(int i = 0; i < wlen; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < wlen)
		{
			_sum[threadIdx.x] += w[id] * w[id];
		}
	}

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < skip && (threadIdx.x + skip) < len)
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = skip;
	}

	if(threadIdx.x == 0)
	{
		atomicAdd(cost, _sum[0] * lambda * 0.5);
	}
}


/*
*/
__global__ void g_getBgrad(float* softMaxDelta, float* bgrad, float* dropb, int batch)
{
	extern __shared__ float _sum[];
	_sum[threadIdx.x] = softMaxDelta[threadIdx.x * gridDim.x + blockIdx.x];

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < skip && (threadIdx.x + skip) < len)
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = skip;
	}
	if(threadIdx.x == 0)
	{
		bgrad[blockIdx.x] = _sum[0] / batch;
		bgrad[blockIdx.x] *= dropb[blockIdx.x];
	}
}


/*
dim3(curDelta->cols), dim3(curDelta->rows), 
sizeof(float) * curDelta->rows
*/
__global__ void g_getBgrad(float* softMaxDelta, float* bgrad, int batch)
{
	extern __shared__ float _sum[];
	_sum[threadIdx.x] = softMaxDelta[threadIdx.x * gridDim.x + blockIdx.x];

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < skip && (threadIdx.x + skip) < len)
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = skip;
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		bgrad[blockIdx.x] = _sum[0] / batch;
	}
}

/*
* function: getcost
*/
__global__ void g_getCost_1(float* softMaxP,
	float* groundTruth, float* cost, int*y, int rows, int cols, int batch)
{
	extern __shared__ float _sum[];
	int len = rows * cols;
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			groundTruth[id] = 0;
		}
	}
	__syncthreads();
	for(int i = 0; i < rows; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < rows)
		{
			int yy = y[id];
			groundTruth[id * cols + yy] = 1;
		}
	}
	_sum[threadIdx.x] = 0;
	__syncthreads();
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			_sum[threadIdx.x] += __logf(softMaxP[id] + 1.0e-10) * groundTruth[id];
		}
	}
	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < skip && (threadIdx.x + skip) < len)
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = skip;
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		cost[0] = -_sum[0] / batch;
	}
}


__global__ void g_getCost_2(float* cost,
	float* weight,
	float lambda, int len)
{
	extern __shared__ float _sum[];
	_sum[threadIdx.x] = 0;
	__syncthreads();
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			_sum[threadIdx.x] += 0.5 * weight[id] * weight[id];
		}
	}
	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < skip && (threadIdx.x + skip) < len)
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = skip;
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		cost[0] += _sum[0] * lambda;
	}
}


/*
* function: cuMatrix(batch, channel * size, 1) to cuMatrix(batch, size, channel)
* blocks  : dim3(batch)
* threads : dim3(min(512, size * channels))
*/
__global__ void g_preDeltaFormat(float* cuPoolFlDelta, 
	float* cuPoolDelta, int batch, int size, int channels){
	int b = blockIdx.x;
	int len = size * channels;
	for(int i = 0; i < len; i += blockDim.x){
		int id = i + threadIdx.x;
		if(id < len){
			int s = id / channels;
			int c = id % channels;
			cuPoolDelta[c * batch * size + b * size + s] = cuPoolFlDelta[b * size * channels + size * c + s];
		}
	}
}


/*
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, channel * size, 1)
* blocks  : dim3(batch)
* threads : dim3(min(512, cuPool[poolidx]->cols))
*/
__global__ void g_convert(float* cuPool, float*cuPoolToFlActi, int batch, int size, int channel){
	int b   = blockIdx.x;
	int len = size * channel;
	for(int i = 0; i < len; i+=blockDim.x){
		int id = i + threadIdx.x;
		if(id < len){
			int s = id / channel;
			int c = id % channel;
			cuPoolToFlActi[b * size * channel + size * c + s] = cuPool[c * batch * size + b * size + s];
		}
	}
}

/*
* 
* function: cuMatrix<int>*(batch, inputDim2*endTime, amount) 
*           to cuMatrix<int>*(batch, amount*inputDim2*endTime, 1)
* Notice that the inputDim is the one dim of the image if amount > 1 (CNN case for img)
*
*   inputSize = amount * inputDim*inputDim
*   inputCols = endTime * inputDim*inputDim
*    channels = amount
*
* blocks  : dim3(batch, endTime)
* threads : dim3(min(1024, inputSize))
*/
__global__ void g_convert_spiketimes(int* inputs_time, int endTime, int inputSize, int inputCols, int channels, int batch, int* inputs_tf){
    int b = blockIdx.x;
    int t = blockIdx.y;
    for(int i = 0; i < inputSize; i += blockDim.x){
        int i_idx = i + threadIdx.x;
        if(i_idx < inputSize){
            int s = i_idx / channels;
            int c = i_idx % channels;
            int index = c * batch * inputCols + b * inputCols + s*endTime + t;
            inputs_tf[b * inputCols * channels + c*inputCols + s*endTime+ t] = inputs_time[index];
        }
    }
}

/*
* 
* function: cuMatrix<int>*(batch, inputDim2, amount) 
*           to cuMatrix<int>*(batch, amount*inputDim2, 1)
* Notice that the inputDim is the one dim of the image if amount > 1 (CNN case for img)
*
*   inputSize = amount * inputDim*inputDim
*   inputDim2 = inputDim*inputDim
*    channels = amount
*
* blocks  : dim3(batch)
* threads : dim3(min(1024, inputSize))
*/
__global__ void g_convert_firecounts(int* counts, int area, int inputSize, int inputDim2, int channels, int batch, int* counts_f){
    int b = blockIdx.y;
    for(int i = 0; i < inputSize; i += blockDim.x){
        int i_idx = i + threadIdx.x;
        if(i_idx < inputSize){
            int s = i_idx / channels;
            int c = i_idx % channels;
            counts_f[b*inputDim2*channels + c*inputDim2 + s] = counts[c*area + b*inputDim2 + s];
        }
    } 
}
/*
* 
* function: cuMatrix<bool>*(batch, endTime*inputDim*inputDim, amount) 
*           to cuMatrix<bool>*(inputSize, endTime*batch, 1)
* Notice that the inputDim is the one dim of the image if amount > 1 (CNN case for img)
*
*   inputSize = amount * inputDim*inputDim
*   inputCols = endTime * inputDim*inputDim
*    channels = amount
*
* blocks  : dim3(batch, endTime)
* threads : dim3(min(1024, inputSize))
*/
__global__ void g_cast_bool_2_float(bool* inputs, int endTime, int inputSize, int inputCols, int channels, int batch, float* inputs_f){
	int b   = blockIdx.x;
    int t   = blockIdx.y;
    int inputDim2 = inputCols / endTime;
	for(int i = 0; i < inputSize; i += blockDim.x){
		int i_idx = i + threadIdx.x;
		if(i_idx < inputSize){
            int s = i_idx / channels; // the index for inputDim2, within the same channel
            int c = i_idx % channels;
            int index = c * batch * inputCols + b * inputCols + t * inputDim2 + s;
            inputs_f[(c * inputDim2 + s) * endTime * batch + t * batch + b] = inputs[index];
		}
	}
}


/*
* 
* function: cuMatrix<float>*(outputSize, endTime*batch) to cuMatrix<float>*(batch, outputSize*endTime)
* blocks  : dim3(batch, outputSize)
* threads : dim3(min(1024, endTime))
*/
__global__ void g_transform_2_batch(float* inputs_rt, int endTime, int outputSize, int batch, float* inputs_r){
	int b     = blockIdx.x;
    int o_idx = blockIdx.y;
    int size2 = outputSize * endTime;
    float* input_r = inputs_r + b * size2;
	for(int t = 0; t < endTime; t += blockDim.x){
		int time = t + threadIdx.x;
		if(time < endTime){
            input_r[o_idx * endTime + time] = inputs_rt[o_idx * endTime * batch + time * batch + b];
		}
	}
}

/*
* function: transform the binary response matrix (batch, outputSize * endTime, outputAmount) 
* to spike times matrix (batch, outputSize*"num of spikes", outputAmount),directly store the spike times. 
* blocks  : dim3(batch, outputAmount)
* threads : dim3(min(1024, outputSize))
*/
__global__ void g_response_2_spiketime(bool* outputs, int* outputs_time, int outputArea, int outputSize, int endTime)
{
    int batchId = blockIdx.x;
    int ok = blockIdx.y;
    bool* output = outputs + ok * outputArea + batchId * endTime * outputSize;
    int* output_time = outputs_time + ok * outputArea + batchId * endTime * outputSize;

    for(int i = 0; i < outputSize; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputSize)
        {
            int col_idx = 0;
            for(int time = 0; time < endTime; ++time)
            {
                if(output[o_idx + time * outputSize])
                {
                    output_time[o_idx * endTime + col_idx] = time;
                    col_idx++;
                }
            }
        }
    }
}

/*
* function: divide the curDelta(batch, outputSize, outputAmount) by vth
* blocks  : dim3(batch, outputAmount)
* threads : dim3(min(1024, outputSize))
*/
__global__ void g_divide_by_threshold(float * _delta, int area, int outputSize, float threshold)
{
    int batchId = blockIdx.x;
    int ok = blockIdx.y;
    float * delta = _delta + ok * area + batchId * outputSize;
    for (int tidx = 0; tidx < outputSize; tidx += blockDim.x) {
        int o_idx = tidx + threadIdx.x;
        if (o_idx < outputSize) {
            delta[o_idx] /= threshold;
        }
    }
}


/*
* blocks : cuSoftMaxP->rows
* threads: cuSoftMaxP->cols
* shared : sizeof(float) * cuSoftMaxP->cols * 2
*/
__global__ void g_getSoftMaxP(float* softMaxP, float* b, int cols)
{
	int bid = blockIdx.x;
	extern __shared__ float _share[];
	float * _max = _share;
	float * _sum = _share + blockDim.x;
	float* sp = softMaxP + bid * cols;
	_sum[threadIdx.x] = 0.0;
	_max[threadIdx.x] = -100000000.0;
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] += b[id];
			_max[threadIdx.x] = max(_max[threadIdx.x], sp[id]);
		}
	}
	__syncthreads();
	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			if(_max[threadIdx.x] < _max[threadIdx.x + skip])
			{
				_max[threadIdx.x] = _max[threadIdx.x + skip];
			}
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] -= _max[0];
			sp[id] = __expf(sp[id]);
			_sum[threadIdx.x] += sp[id];
		}
	}
	__syncthreads();
	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] /= _sum[0];
		}
	}
}

__global__ void g_getSoftMaxDelta(float* softMaxDelta, float* softMaxP, float* groundTruth, int len)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			softMaxDelta[id] = softMaxP[id] - groundTruth[id];
		}
	}
}

__global__ void g_getSmrWgrad(float* wgrad, float* weight, float lambda, int len, int batch)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			wgrad[id] = lambda * weight[id] + wgrad[id] / batch;
		}
	}
}
