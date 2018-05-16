#pragma warning (disable: 4819)
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include "net.cuh"
#include "net_spiking.cuh"
#include "common/cuMatrix.h"
#include "common/util.h"
#include "dataAugmentation/cuTransformation.cuh"
#include "common/Config.h"
#include "common/cuMatrixVector.h"

#include "common/Config.h"
#include "dataAugmentation/dataPretreatment.cuh"

#include "readData/readMnistData.h"
#include "readData/readCIFAR10Data.h"
#include "readData/readChineseData.h"
#include "readData/readCIFAR100Data.h"
#include "readData/readNMnistData.h"
#include "readData/readSpeechData.h"
#include "common/track.h"
#include "layers/Pooling.h"

#define VERIFY
//#define VERIFY_RESERVOIR
//#define SPIKING_CNN
//#define VERIFY_SOFTMAX_SPIKING_CNN
//#define VERIFY_SPIKING_CNN

void runMnist();
void runCifar10();
void runCifar100();
void runChinese();
void runNMnist();
void runSpikingMnist();
void runSpeech();
void cuVoteMnist();
bool init(cublasHandle_t& handle);

std::vector<int> g_argv;


int main (int argc, char** argv)
{
    //cudaDeviceReset();
    cudaSetDevice(0);
#ifdef linux
    signal(SIGSEGV, handler);   // install our handler
#endif
	srand(clock());
	if(argc >= 3){
		g_argv.push_back(atoi(argv[1]));
		g_argv.push_back(atoi(argv[2]));
	}
	printf("1. MNIST\n2. CIFAR-10\n3. CHINESE\n4. CIFAR-100\n5. VOTE MNIST\n6. NMnist\n7. Spiking Mnist\n8. Spoken English Letter\nChoose the dataSet to run:");
	int cmd = 8;
    /*
	if(g_argv.size() >= 2)
		cmd = g_argv[0];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    */
	if(cmd == 1)
		runMnist();
	else if(cmd == 2)
		runCifar10();
	else if(cmd == 3)
		runChinese();
	else if(cmd == 4)
		runCifar100();
	else if(cmd == 5)
		cuVoteMnist();
    else if(cmd == 6)
        runNMnist();
    else if(cmd == 7)
        runSpikingMnist();
    else if(cmd == 8)
        runSpeech();
	return EXIT_SUCCESS;
}

#ifdef linux
void runChinese() {
	const int nclasses = 10;
	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<float> trainX;
	cuMatrixVector<float> testX;
	cuMatrix<int>* trainY, *testY;
	readChineseData(trainX, testX, trainY, testY);

	Config::instance()->initPath("Config/ChineseConfig.txt");

	/*build CNN net*/
	int ImgSize = trainX[0]->rows;
	
	int crop = Config::instance()->getCrop();

	int nsamples = trainX.size();
	int batch = Config::instance()->getBatchSize();
	float start, end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf(
			"1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else {
        if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    }

	if (cmd == 2)
		cuReadConvNet(ImgSize - crop,
				"Result/checkPoint.txt", nclasses);
    
    buildNetWork(trainX.size(), testX.size());

	/*learning rate*/
	std::vector<float>nlrate;
	std::vector<float>nMomentum;
	std::vector<int>epoCount;
	float r = 0.05;
	float m = 0.90;
	int e = 50;
	for(int i = 0; i < 20; i++){
		nlrate.push_back(r);
		nMomentum.push_back(m);
		epoCount.push_back(e);
		r = r * 0.90;
		m = m + 0.005;
		if(m >= 1.0) m = 0.99;
	}
	start = clock();
	cuTrainNetwork(trainX, trainY, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}
#else
void runChinese(){

}
#endif


void runCifar100(){
	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<float>trainX;
	cuMatrixVector<float>testX;
	cuMatrix<int>* trainY, *testY;

	Config::instance()->initPath("Config/Cifar100Config.txt");
	read_CIFAR100_Data(trainX, testX, trainY, testY);
	preProcessing(trainX, testX);

	const int nclasses = Config::instance()->getClasses();

	/*build CNN net*/
	int ImgSize = trainX[0]->rows;
	Config::instance()->setImageSize(ImgSize - Config::instance()->getCrop());
	int crop = Config::instance()->getCrop();

	int nsamples = trainX.size();

	int batch = Config::instance()->getBatchSize();
	float start,end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf("1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }

	buildNetWork(trainX.size(), testX.size());


	if(cmd == 2)
		cuReadConvNet(ImgSize - crop, "Result/checkPoint.txt", nclasses);

	/*learning rate*/
	std::vector<float>nlrate;
	std::vector<float>nMomentum;
	std::vector<int>epoCount;
	float r = 0.05f;
	float m = 0.90f;
	int e = 10;
	for(int i = 0; i < 100; i++){
		nlrate.push_back(r);
		nMomentum.push_back(m);
		epoCount.push_back(e);
		r = r * 0.90f;
		m = m + 0.001f;
		if(m >= 1.0) break;
	}
	start = clock();
	cuTrainNetwork(trainX, trainY, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();
	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}

void runCifar10()
{
	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<float>trainX;
	cuMatrixVector<float>testX;
	cuMatrix<int>* trainY, *testY;

	Config::instance()->initPath("Config/Cifar10Config.txt");
	read_CIFAR10_Data(trainX, testX, trainY, testY);
	preProcessing(trainX, testX);

	const int nclasses = Config::instance()->getClasses();

	/*build CNN net*/
	int ImgSize = trainX[0]->rows;
	Config::instance()->setImageSize(ImgSize - Config::instance()->getCrop());
	int crop = Config::instance()->getCrop();

	int nsamples = trainX.size();
	int batch = Config::instance()->getBatchSize();
	float start,end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf("1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }

	buildNetWork(trainX.size(), testX.size());

	if(cmd == 2)
		cuReadConvNet(ImgSize - crop, "Result/checkPoint.txt", nclasses);
	
	/*learning rate*/
	std::vector<float>nlrate;
	std::vector<float>nMomentum;
	std::vector<int>epoCount;

	nlrate.push_back(0.005f);    nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.004f);    nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.003f);    nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.002f);    nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.001f);    nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0009f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0008f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0007f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0006f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0005f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0004f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0003f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0002f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.0001f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.00001f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	nlrate.push_back(0.000001f);   nMomentum.push_back(0.90f);  epoCount.push_back(50);
	start = clock();
	cuTrainNetwork(trainX, trainY, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}

/*init cublas Handle*/
bool init(cublasHandle_t& handle)
{
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf ("init: CUBLAS initialization failed\n");
		exit(0);
	}
	return true;
}

void runMnist(){
	const int nclasses = 10;

 	/*state and cublas handle*/
 	cublasHandle_t handle;
	init(handle);
	
 	/*read the data from disk*/
	cuMatrixVector<float>trainX;
	cuMatrixVector<float>testX;
 	cuMatrix<int>* trainY, *testY;
	Config::instance()->initPath("Config/MnistConfig.txt");
 	readMnistData(trainX, trainY, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", 60000, 1);
 	readMnistData(testX , testY,  "mnist/t10k-images-idx3-ubyte",  "mnist/t10k-labels-idx1-ubyte",  10000, 1);
	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();
 	/*build CNN net*/
 	int ImgSize = trainX[0]->rows;
	Config::instance()->setImageSize(ImgSize - Config::instance()->getCrop());
	int crop = Config::instance()->getCrop();

 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf("1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }

	buildNetWork(trainX.size(), testX.size());

	if(cmd == 2)
		cuReadConvNet(ImgSize - crop, "Result/checkPoint.txt", nclasses);

	/*learning rate*/
	std::vector<float>nlrate;
	std::vector<float>nMomentum;
	std::vector<int>epoCount;
	nlrate.push_back(0.05f);   nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.04f);   nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.03f);   nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.02f);   nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.01f);   nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.009f);  nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.008f);  nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.007f);  nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.006f);  nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.005f);  nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.004f);  nMomentum.push_back(0.90f);  epoCount.push_back(15);
	nlrate.push_back(0.003f);  nMomentum.push_back(0.90f);  epoCount.push_back(15);

	start = clock();
	cuTrainNetwork(trainX, trainY, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}

void runNMnist(){
	const int nclasses = 10;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
#if   defined(VERIFY)
    config->initPath("Config/NMnistConfig_test_complete.txt");
#elif defined(VERIFY_RESERVOIR)
    config->initPath("Config/NMnistConfig_test_reservoir.txt");
#else
    config->initPath("Config/NMnistConfig.txt");
#endif

    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_per_class = config->getTrainPerClass();
    int test_per_class = config->getTestPerClass();
 	readNMnistData(trainX, trainY, config->getTrainPath(), train_per_class, input_neurons, end_time, false);
 	readNMnistData(testX , testY, config->getTestPath(), test_per_class, input_neurons, end_time, config->hasMarkTest());

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net 
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	cuInitDistortionMemery(batch, 28);
    /*
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    */
	buildSpikingNetwork(trainX.size(), testX.size());

    
	//if(cmd == 2)
	//cuReadSpikingNet("Result/checkPoint_current_best.txt");
    

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;
#if defined(VERIFY) || defined(VERIFY_RESERVOIR)
	nlrate.push_back(0.00001f);   nMomentum.push_back(0.00f);  epoCount.push_back(1);
#else
    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.001f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }
#endif

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}


void runSpikingMnist(){
	const int nclasses = 10;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
#if   defined(VERIFY)
    config->initPath("Config/SpikingMnistConfig_test.txt");
#elif     defined(VERIFY_SPIKING_CNN)
    config->initPath("Config/SpikingCNNMnistConfig_test.txt");
#elif   defined(VERIFY_SOFTMAX_SPIKING_CNN)
    config->initPath("Config/SpikingCNNMnistConfig_test_softmax.txt");
#elif   defined(SPIKING_CNN)
    config->initPath("Config/SpikingCNNMnistConfig.txt");
#else
    config->initPath("Config/SpikingMnistConfig.txt");
#endif

    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_samples = config->getTrainSamples();
    int test_samples = config->getTestSamples();
#if defined(VERIFY) || defined(VERIFY_SPIKING_CNN) || defined(VERIFY_SOFTMAX_SPIKING_CNN)
    read_dumped_input_inside(config->getTrainPath(), trainX, end_time, input_neurons);
    std::vector<int> my_label(1, 5);
    trainY = new cuMatrix<int>(1, 1, 1);
    readSpeechLabel(my_label, trainY);
    read_dumped_input_inside(config->getTestPath(), testX, end_time, input_neurons);
    testY = new cuMatrix<int>(1, 1, 1);
    readSpeechLabel(my_label, testY);
#else
 	readSpikingMnistData(trainX, trainY, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", train_samples, input_neurons, end_time);
 	readSpikingMnistData(testX , testY, "mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte",  test_samples, input_neurons, end_time);
#endif
	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net
 	int ImgSize = 28; // fixed here for spiking mnist
	Config::instance()->setImageSize(ImgSize);
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    
	int cmd;
	cuInitDistortionMemery(batch, ImgSize);
    /*
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    */
	buildSpikingNetwork(trainX.size(), testX.size());

    /*
	if(cmd == 2)
		cuReadSpikingNet("Result/checkPoint.txt");
    */

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;
#if defined(VERIFY)
	nlrate.push_back(0.001f);   nMomentum.push_back(0.00f);  epoCount.push_back(1);
#elif defined(VERIFY_SPIKING_CNN)
	nlrate.push_back(0.001f);   nMomentum.push_back(0.00f);  epoCount.push_back(1);
#else
    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.001f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }
#endif

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}



void runSpeech(){
	const int nclasses = 26;

 	//*state and cublas handle
 	cublasHandle_t handle;
	init(handle);
	
 	//* read the data from disk
	cuMatrixVector<bool> trainX;
	cuMatrixVector<bool> testX;
 	cuMatrix<int>* trainY, *testY;
    
    //* initialize the configuration
	Config * config = Config::instance();
#if   defined(VERIFY)
    config->initPath("Config/SpokenLetterConfig_test.txt");
#else
    config->initPath("Config/SpokenLetterConfig.txt");
#endif
    ConfigDataSpiking * ds_config = (ConfigDataSpiking*)config->getLayerByName("data");
    int input_neurons = ds_config->m_inputNeurons;
    int end_time = config->getEndTime();
    int train_samples = config->getTrainSamples();
    int test_samples = config->getTestSamples();
 	readSpeechData(trainX, trainY, config->getTrainPath(), train_samples, input_neurons, end_time, nclasses);
 	readSpeechData(testX , testY,  config->getTestPath(), test_samples, input_neurons, end_time, nclasses);

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

 	//* build SNN net 
 	int nsamples = trainX.size();

 	int batch = Config::instance()->getBatchSize();
	float start,end;
    /*
	int cmd;
	printf("1. random init weight\n2. Read weight from the checkpoint\nChoose the way to init weight:");

	if(g_argv.size() >= 2)
		cmd = g_argv[1];
	else 
		if(1 != scanf("%d", &cmd)){
            LOG("scanf fail", "result/log.txt");
        }
    */
	buildSpikingNetwork(trainX.size(), testX.size());

    /*
	if(cmd == 2)
	    cuReadSpikingNet("Result/checkPoint.txt");
    */

	//* learning rate
	std::vector<float> nlrate;
	std::vector<float> nMomentum;
	std::vector<int> epoCount;
#ifdef VERIFY
	nlrate.push_back(0.00001f);   nMomentum.push_back(0.00f);  epoCount.push_back(1);
#else
    int epochs = Config::instance()->getTestEpoch();
    for(int i = 0; i < epochs; ++i){
        nlrate.push_back(0.001f/sqrt(i+1)); nMomentum.push_back(0.90f);  epoCount.push_back(1);
    }
#endif

	start = clock();
	cuTrainSpikingNetwork(trainX, trainY, testX, testY, batch, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	char logStr[1024];
	sprintf(logStr, "training time hours = %f\n", 
		(end - start) / CLOCKS_PER_SEC / 3600);
	LOG(logStr, "Result/log.txt");
}

void cuVoteMnist()
{
	return;
	const int nclasses = 10;

	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<float>trainX;
	cuMatrixVector<float>testX;
	cuMatrix<int>* trainY, *testY;

	readMnistData(trainX, trainY, "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", 60000, 1);
 	readMnistData(testX , testY,  "mnist/t10k-images.idx3-ubyte",  "mnist/t10k-labels.idx1-ubyte",  10000, 1);

	int ImgSize = trainX[0]->rows;
	Config::instance()->setImageSize(ImgSize - Config::instance()->getCrop());

	const char* path[] = {"mnist_result_cfm_5/1/checkPoint.txt",
		"mnist_result_cfm_5/2/checkPoint.txt",
		"mnist_result_cfm_5/3/checkPoint.txt",
		"mnist_result_cfm_5/4/checkPoint.txt",
		"mnist_result_cfm_5/5/checkPoint.txt",
		"mnist_result_cfm_5/6/checkPoint.txt",
		"mnist_result_cfm_5/7/checkPoint.txt",
		"mnist_result_cfm_5/8/checkPoint.txt",
		"mnist_result_cfm_5/9/checkPoint.txt"};

	const char* initPath[] = {"mnist_result_cfm_4/1/MnistConfig.txt",
		"mnist_result_cfm_5/2/MnistConfig.txt",
		"mnist_result_cfm_5/3/MnistConfig.txt",
		"mnist_result_cfm_5/4/MnistConfig.txt",
		"mnist_result_cfm_5/5/MnistConfig.txt",
		"mnist_result_cfm_5/6/MnistConfig.txt",
		"mnist_result_cfm_5/7/MnistConfig.txt",
		"mnist_result_cfm_5/8/MnistConfig.txt",
		"mnist_result_cfm_5/9/MnistConfig.txt"};

	/*build CNN net*/
	/*
	char* path[] = {"mnist_result_cfm_2/1_9967/checkPoint.txt",
		"mnist_result_cfm_2/2_9964/checkPoint.txt",
		"mnist_result_cfm_2/3_9966/checkPoint.txt",
		"mnist_result_cfm_2/4_9970/checkPoint.txt",
		"mnist_result_cfm_2/5_9967/checkPoint.txt",
		"mnist_result_cfm_2/6_9963/checkPoint.txt",
		"mnist_result_cfm_2/7_9965/checkPoint.txt",
		"mnist_result_cfm_2/8_9963/checkPoint.txt",
		"mnist_result_cfm_2/9_9964/checkPoint.txt"};

	char* initPath[] = {"mnist_result_cfm_2/1_9967/MnistConfig.txt",
		"mnist_result_cfm_2/2_9964/MnistConfig.txt",
		"mnist_result_cfm_2/3_9966/MnistConfig.txt",
		"mnist_result_cfm_2/4_9970/MnistConfig.txt",
		"mnist_result_cfm_2/5_9967/MnistConfig.txt",
		"mnist_result_cfm_2/6_9963/MnistConfig.txt",
		"mnist_result_cfm_2/7_9965/MnistConfig.txt",
		"mnist_result_cfm_2/8_9963/MnistConfig.txt",
		"mnist_result_cfm_2/9_9964/MnistConfig.txt"};
		*/
	/*
	ncfm
	char* path[] = {"mnist_result/1_9973/checkPoint.txt",
	"mnist_result/2_9972/checkPoint.txt",
	"mnist_result/3_9969/checkPoint.txt",
	"mnist_result/4_9968/checkPoint.txt",
	"mnist_result/5_9974/checkPoint.txt",
	"mnist_result/6_9973/checkPoint.txt",
	"mnist_result/7_9968/checkPoint.txt",
	"mnist_result/8_9968/checkPoint.txt",
	"mnist_result/9_9968/checkPoint.txt",
	"mnist_result/10_9976/checkPoint.txt",
	"mnist_result/11_9969/checkPoint.txt",
	"mnist_result/12_9968/checkPoint.txt",
	"mnist_result/13_9973/checkPoint.txt"};

	char* initPath[] = {"mnist_result/1_9973/MnistConfig.txt",
	"mnist_result/2_9972/MnistConfig.txt",
	"mnist_result/3_9969/MnistConfig.txt",
	"mnist_result/4_9968/MnistConfig.txt",
	"mnist_result/5_9974/MnistConfig.txt",
	"mnist_result/6_9973/MnistConfig.txt",
	"mnist_result/7_9968/MnistConfig.txt",
	"mnist_result/8_9968/MnistConfig.txt",
	"mnist_result/9_9968/MnistConfig.txt",
	"mnist_result/10_9976/MnistConfig.txt",
	"mnist_result/11_9969/MnistConfig.txt",
	"mnist_result/12_9968/MnistConfig.txt",
	"mnist_result/13_9973/MnistConfig.txt"};
	*/

	int numPath = sizeof(path) / sizeof(char*);
	int * mark  = new int[1 << numPath];
	memset(mark, 0, sizeof(int) * (1 << numPath));
	std::vector<int>vCorrect;
	std::vector<cuMatrix<int>*>vPredict;

	for(int i = 0; i < numPath; i++)
	{
		Config::instance()->initPath(initPath[i]);

		int ImgSize = trainX[0]->rows;
		int crop    = Config::instance()->getCrop();

		int nsamples = trainX.size();
		int batch = Config::instance()->getBatchSize();

		vPredict.push_back(new cuMatrix<int>(testY->getLen(), nclasses, 1));
		
		buildNetWork(trainX.size(), testX.size());
		cuReadConvNet(ImgSize - crop, path[i], nclasses);
		int cur;
// 		int cur = voteTestDate(
// 			testX,
// 			testY,
// 			vPredict[i],
// 			batch,
// 			ImgSize,
// 			nclasses,
// 			handle);
		cuFreeCNNMemory(batch, trainX, testX);
		cuFreeConvNet();
		vCorrect.push_back(cur);
		Config::instance()->clear();
		printf("%d %s\n", cur, path[i]);
	}
	
	int max = -1;
	int val = 1;
	int cur;
	cuMatrix<int>* voteSum = new cuMatrix<int>(testY->getLen(), nclasses, 1);
	cuMatrix<int>* correct = new cuMatrix<int>(1, 1, 1);

	for(int m = (1 << numPath) - 1; m >= 1; m--)
	{
		int t = 0;
		for(int i = 0; i < numPath; i++){
			if(m & (1 << i)){
				t++;
			}
		}
		if(t != 5) continue;
		voteSum->gpuClear();
		if(mark[m] != 0)
		{
			cur = mark[m];
			for(int i = 0; i < numPath; i++)
			{
				if(!(m & (1 << i)))continue;
				printf("%d %s\n", vCorrect[i], path[i]);
			}
		}
		else
		{
			int v = 0;
			for(int i = numPath - 1; i >= 0; i--)
			{
				if(!(m & (1 << i)))continue;
				v = v | (1 << i);
				correct->gpuClear();
				cur = cuVoteAdd(voteSum, vPredict[i], testY, correct, nclasses);
				mark[v] = cur;
				printf("%d %d %s\n", vCorrect[i], cur, path[i]);
			}
		}
		if(cur >= max)
		{
			max = cur;
			val = m;
		}

		printf("m = %d val = %d max = %d \n\n",m, val, max);
	}

	voteSum->gpuClear();
	int m = val;
	for(int i = numPath - 1; i >= 0; i--)
	{
		if(!(m & (1 << i)))continue;
		correct->gpuClear();
		cur = cuVoteAdd(voteSum, vPredict[i], testY, correct, nclasses);
		printf("%d %d %s\n", vCorrect[i], cur, path[i]);
	}
}
