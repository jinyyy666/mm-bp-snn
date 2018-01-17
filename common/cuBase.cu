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
function: g_preDeltaFormat
threads : <<<dim3(batch), dim3(512)>>> 
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
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
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
* function: cast cuMatrix<bool>*(batch, inputDim * endTime) to cuMatrix<float>*(endTime, inputDim)
*           only use this function when batch = 1 !
* blocks  : dim3(endTime) endTime --> nrows
* threads : dim3(min(1024, inputDim)) inputDim --> ncols
*/
__global__ void g_cast_bool_2_float(bool* inputs, int endTime, int inputDim, float* input_f){
	int t   = blockIdx.x;
	for(int i = 0; i < inputDim; i += blockDim.x){
		int i_idx = i + threadIdx.x;
		if(i_idx < inputDim){
            input_f[i_idx + t * inputDim] = inputs[i_idx + t * inputDim] == true ? 1 : 0;
		}
	}
}



/*
* function: transform the binary response matrix (batch, outputDim * endTime) to spike times matrix 
* (batch, outputDim * "num of spikes"), directly store the spike times. 
* blocks  : dim3(batch)
* threads : dim3(min(1024, outputDim))
*/
__global__ void g_response_2_spiketime(bool* outputs, int* outputs_time, int outputDim, int endTime)
{
    int batchId = blockIdx.x;
    bool* output = outputs + batchId * endTime * outputDim;
    int* output_time = outputs_time +  batchId * endTime * outputDim;

    for(int i = 0; i < outputDim; i += blockDim.x)
    {
        int o_idx = i + threadIdx.x;
        if(o_idx < outputDim)
        {
            int col_idx = 0;
            for(int time = 0; time < endTime; ++time)
            {
                if(output[o_idx + time * outputDim])
                {
                    output_time[o_idx * endTime + col_idx] = time;
                    col_idx++;
                }
            }
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
