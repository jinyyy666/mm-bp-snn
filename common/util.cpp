#include "util.h"
#include <opencv2/opencv.hpp>
#include "Config.h"
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"

#define CONST_SPIKING_SCALE (5.5 * 255.0f)

using namespace cv;
using namespace std;

int getSharedMemory(vector<unsigned int>& vec) {
    int dev_num = 0;
    if(!vec.empty())vec.clear();

    cudaError_t error_id = cudaGetDeviceCount(&dev_num);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) error_id,
                cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    for (int dev = 0; dev < dev_num; dev++) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        vec.push_back((unsigned int)deviceProp.sharedMemPerBlock);
    }
    return dev_num;
}

bool checkSharedMemory(int id, size_t MemorySize){
    static std::vector<unsigned int>ret;
    if(ret.size() == 0)
    {
        getSharedMemory(ret);
    }
    if(ret.size() > (size_t)id){
        if(ret[id] >= MemorySize){
            return false;
            return true;
        }
        else{
            return false;
            LOG("getSharedMemory error", "result/log.txt");
            exit(0);
        }
    }else{
        return false;
        LOG("getSharedMemory error", "result/log.txt");
        exit(0);
    }
}

int getCV_32()
{
    int cv_32;
    if(Config::instance()->getChannels() == 1){
        cv_32 = CV_32FC1;
    }
    else if(Config::instance()->getChannels() == 3){
        cv_32 = CV_32FC3;
    }
    else if(Config::instance()->getChannels() == 4){
        cv_32 = CV_32FC4;
    }
    return cv_32;
}

void showImg(cuMatrix<float>* x, float scala)
{
    x->toCpu();

    int CV_32;
    if(x->channels == 1){
        CV_32 = CV_32FC1;
    }
    else if(x->channels == 3){
        CV_32 = CV_32FC3;
    }
    else if(x->channels == 4){
        CV_32 = CV_32FC4;
    }
    Mat src(x->rows, x->cols, CV_32);;


    for(int i = 0; i < x->rows; i++)
    {
        for(int j = 0; j < x->cols; j++)
        {
            if(x->channels == 1){
                src.at<float>(i, j) = x->get(i, j, 0);
            }
            else if(x->channels == 3){
                src.at<Vec3f>(i, j) = 
                    Vec3f(
                            x->get(i, j, 0),
                            x->get(i, j, 1), 
                            x->get(i, j, 2));
            }else if(x->channels == 4){
                src.at<Vec4f>(i, j) = 
                    Vec4f(
                            x->get(i, j, 0),
                            x->get(i, j, 1),
                            x->get(i, j, 2),
                            x->get(i, j, 3));
            }
        }
    }

    Size size;
    size.width  = int(1.0f * src.cols * scala);
    size.height = int(1.0f * src.rows * scala);


    Mat dst(size.height, size.width, CV_32);

    cv::resize(src, dst, size);

    static int id = 0;
    id++;
    char ch[10];
    sprintf(ch, "%d", id);
    namedWindow(ch, WINDOW_AUTOSIZE);
    cv::imshow(ch, dst);
}

void DebugPrintf(cuMatrix<float>*x)
{
    FILE *file = fopen("DEBUG.txt", "w+");
    x->toCpu();
    for(int c = 0; c < x->channels; c++)
    {
        for(int i = 0; i < x->rows; i++)
        {
            for(int j = 0; j < x->cols; j++)
            {
                fprintf(file, "%f ", x->get(i, j, c));
            }fprintf(file, "\n");
        }
    }
}

void DebugPrintf(float* data, int len, int dim)
{
    for(int id = 0; id < len; id += dim*dim)
    {
        float* img = data + id;
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                printf("%f ", img[i * dim + j]);
            }printf("\n");
        }
    }
}

void LOG(const char* str, const char* file)
{
    FILE* f = fopen(file,"a");
    printf("%s", str);
    fprintf(f,"%s",str);
    fclose(f);
}


void createGaussian(float* gaussian, float dElasticSigma1, float dElasticSigma2,
        int rows, int cols, int channels, float epsilon)
{
    int iiMidr = rows >> 1;
    int iiMidc = cols >> 1;

    float _sum = 0.0;
    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col < cols; col++)
        {
            float val1 = 1.0f / (dElasticSigma1 * dElasticSigma2 * 2.0f * 3.1415926535897932384626433832795f);
            float val2 = 1.0f * (row-iiMidr)*(row-iiMidr) / (dElasticSigma1 * dElasticSigma1) + 1.0f * (col-iiMidc)*(col-iiMidc) / (dElasticSigma2 * dElasticSigma2) 
                + 2.0f * (row - iiMidr) * (col - iiMidc) / (dElasticSigma1 * dElasticSigma2);
            gaussian[row * cols + col] = val1 * exp(-1.0f * val2);
            //gaussian[row * cols + col] = exp(gaussian[row * cols + col]);
            _sum += gaussian[row * cols + col];
            // 			if(_max < fabs(gaussian[row * cols + col]))
            // 			{
            // 				_max = fabs(gaussian[row * cols + col]);
            // 			}
        }
    }
    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col < cols; col++)
        {
            float val = gaussian[row * cols + col] / _sum;
            //val = val * 2.0 - 0.5;
            //val = val * epsilon;
            gaussian[row * cols + col] = val * epsilon;
            //printf("%f ", val * epsilon);
        }//printf("\n");
    }
    //printf("\n\n");
}


void dropDelta(cuMatrix<float>* M, float cuDropProb)
{
    //srand(clock());
    for(int c = 0; c < M->channels; c++){
        //cv::Mat ran = cv::Mat::zeros(M->rows, M->cols, CV_64FC1);
        //cv::theRNG().state = clock();
        //randu(ran, cv::Scalar(0), cv::Scalar(1.0));
        for(int i = 0; i < M->rows; i++){
            for(int j = 0; j < M->cols; j++){
                float r = 1.0f * rand() / RAND_MAX;
                if(r < cuDropProb)
                    M->set(i, j, c, 0.0);
                else 
                    M->set(i, j, c, 1.0);
            }
        }
    }
    M->toGpu();
}


void dropScale(cuMatrix<float>* M, float cuDropProb)
{
    for(int c = 0; c < M->channels; c++){
        for(int i = 0; i < M->rows; i++){
            for(int j = 0; j < M->cols; j++){
                M->set(i, j, c, 1.0 - cuDropProb);
            }
        }
    }
    M->toGpu();
}


void initMatrix(cuMatrix<float>* M, float initW)
{
    for(int c = 0; c < M->channels; c++){
        srand(clock());
        Mat matrix2xN = Mat::zeros(M->rows,M->cols,CV_64FC1);
        randn(matrix2xN, 0, initW); 
        for(int i = 0; i < matrix2xN.rows; i++){
            for(int j = 0; j < matrix2xN.cols; j++){
                M->set(i,j,c, matrix2xN.at<float>(i, j));
                printf("%f ", matrix2xN.at<float>(i, j));
            }printf("\n");
        }
        printf("\n\n");
    }

    M->toGpu();
}

void checkMatrixIsSame(cuMatrix<float>*x, cuMatrix<float>*y, int channel)
{
    assert(x->rows == y->rows);
    assert(x->cols == y->cols);
    assert(x->channels == y->channels);
    for(int i = 0; i < x->rows; i++){
        for(int j = 0; j < x->cols; j++){
            for(int k = 0; k < x->channels; k++){
                float diff = x->get(i, j, k) - y->get(i, j, k);
                if(fabs(diff) > 0.0001){
                    printf("\n for %d_th weight matrix (or filter)", channel);
                    printf("\n%d %d %d %f %f %f\n", i, j, k, x->get(i,j, k), y->get(i,j,k), diff);
                }
                assert(fabs(diff) < 0.0001);
            }
        }
    }
}

//* this is only used to check the spike outputs
//* the matrix: 1 x (endTime * outputSize) x outputAmount
void checkMatrixIsSame(cuMatrix<bool>*x, cuMatrix<bool>*y, int n_outputs)
{
    assert(x->rows == y->rows);
    assert(x->cols == y->cols);
    assert(x->channels == y->channels);
    int nrows = x->rows, ncols = x->cols;
    for(int c = 0; c < x->channels; c++){
        for(int i = 0; i < nrows; i++){
            for(int j = 0; j < ncols; j++){
                int t = j / n_outputs;
                int idx = j % n_outputs;
                float diff = int(x->get(i, j, c)) - int(y->get(i, j, c));
                if(fabs(diff) > 0.001){
                    printf("\n%d %d %d %d %d %f\n", i, j, c, x->get(i,j, c), y->get(i,j,c), diff);
                    std::cout<<"For neuron: "<<idx<<" in output channel: "<<c<<" at time: "<<t<<"\n"
                            <<(x->get(i,j,c) ? "Expect to fire" : "Expect not fire")<<"\n"
                            <<(y->get(i,j,c) ? "But fire" : "But not fire")<<std::endl;
 
                }
                assert(fabs(diff) < 0.001);
            }
        }
    }
}

int extractNeuronIndex(const string& name)
{
    size_t pos = name.find_last_of("_");
    if(pos == string::npos)
    {
        cout<<"Invalid neuron name: "<<name<<endl;
        exit(EXIT_FAILURE);
    }
    string index = name.substr(pos + 1);
    return atoi(index.c_str());
}

void convertToSpikeTimes(cuMatrix<float>* preproc_img, vector<vector<int> >*& sp_times, int imgSize, int end_time){
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    assert(sp_times->size() == preproc_img->getLen());
    assert(imgSize == preproc_img->rows);
    assert(imgSize == preproc_img->cols);
    assert(imgSize > 0);

    int index = -1;
    for(int i = 0; i < sp_times->size(); ++i){
        int r = i / imgSize;
        int c = i % imgSize;
        float distorted = preproc_img->get(r, c, 0); // this value is in (-1, 1);
        float freq = ((distorted + 1) * 255.0f / 2) / CONST_SPIKING_SCALE; // map back to freq range
        index++;
        (*sp_times)[index].clear();
        if(freq < 0 || fabs(freq - 0.0f) < 1e-5)    continue;
        for(int time = 1; time < end_time; ++time){
            if(dist(e2) < freq) (*sp_times)[index].push_back(time);
        }
    }
}

void print2DVectorToFile(vector<vector<int> >& v, string filename){
    ofstream f_out(filename.c_str());
    if(!f_out.is_open()){
       cout<<"print2DVectorToFile::Cannot open the file: "<<filename<<endl;
        exit(EXIT_FAILURE); 
    }
    for(int i = 0; i < v.size(); i++){
        f_out<<v[i].size()<<endl;
    }
}

//* read the dumped spikes from matlab code, only used for spiking CNN
void readSpikesFromDumpfile(const std::string& filename, cuMatrix<bool>*& x){
    FILE *file = fopen(filename.c_str(), "r");

    char logStr[256];
    if(file == NULL){
        sprintf(logStr, "Cannot open file: %s", filename.c_str());  LOG(logStr, "Result/log.txt");
        assert(0);
    }
 
    float val = 0;
    for(int c = 0; c < x->channels; c++){
        for(int i = 0; i < x->rows; i++){
            for(int j = 0; j < x->cols; j++){
                if(fscanf(file, "%f", &val) == EOF)
                {
                    sprintf(logStr, "Reading dumped spikes failed for %s @row: %d\t@col: %d\t@channel: %d\n", filename.c_str(), i, j, c);
                    LOG(logStr, "Result/log.txt");
                    assert(0);
                }
                if(val < 1)
                    x->set(i, j, c, false);
                else
                    x->set(i, j, c, true);
            }
        }
    }
}
