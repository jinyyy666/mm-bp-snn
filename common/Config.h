#ifndef __CONFIG_H__
#define __CONFIG_H__


#include <string>
#include <vector>
#include "util.h"
#include <time.h>

class ConfigGradient
{
public:
	ConfigGradient(bool isGradientChecking)
	{
		m_IsGradientChecking = isGradientChecking;
	}
	bool getValue(){return m_IsGradientChecking;};
private:
	bool m_IsGradientChecking;
};

class ConfigRecordWrong
{
public:
    ConfigRecordWrong(bool hasRecordWrong)
    {
        m_hasRecordWrong = hasRecordWrong;
    }
    bool getValue(){return m_hasRecordWrong;}
private:
    bool m_hasRecordWrong;
};

class ConfigMarkTest
{
public:
    ConfigMarkTest(bool hasMarker)
    {
        m_hasMarkTest = hasMarker;
    }
    bool getValue(){return m_hasMarkTest;}
private:
    bool m_hasMarkTest;
};

class ConfigBoostWeight
{
public:
    ConfigBoostWeight(bool hasBoostWeight)
    {
        m_hasBoostWeight = hasBoostWeight;
    }
    bool getValue(){return m_hasBoostWeight;}
private:
    bool m_hasBoostWeight;
};


class ConfigEffectRatio
{
public:
    ConfigEffectRatio(bool useEffect)
    {
        m_useEffect = useEffect;
    }
    bool getValue(){return m_useEffect;}
private:
    bool m_useEffect;
};

class ConfigOptimizer
{
public:
    ConfigOptimizer(std::string opt_t):m_optType(opt_t){}
    std::string getType(){return m_optType;}
private:
    std::string m_optType;
};

class ConfigWeightReg
{
public:
    ConfigWeightReg(float lambda, float beta):m_lambda(lambda), m_beta(beta){}
    float getLambda(){return m_lambda;}
    float getBeta(){return m_beta;}
private:
    float m_lambda;
    float m_beta;
};

class ConfigWeightLimit
{
public:
    ConfigWeightLimit(float limit):m_limit(limit){}
    float getLimit(){return m_limit;}
private:
    float m_limit;
};

class ConfigImageShow
{
public:
	ConfigImageShow(bool imageShow)
	{
		m_imageShow = imageShow;
	}
	bool getValue(){return m_imageShow;};
private:
	bool m_imageShow;
};

class ConfigCrop
{
public:
	ConfigCrop(int crop)
	{
		m_crop = crop;
	}
	int getValue(){return m_crop;};
private:
	int m_crop;
};

class ConfigScale
{
public:
	ConfigScale(float scale):m_scale(scale){}
	float getValue(){return m_scale;}
private:
	float m_scale;
};

class ConfigWhiteNoise
{
public:
	ConfigWhiteNoise(float stdev):m_stdev(stdev){}
	float getValue(){return m_stdev;}
private:
	float m_stdev;
};

class ConfigDataset
{
public:
    ConfigDataset(std::string train_path, std::string test_path, int train_samples, int test_samples, int train_per_class, int test_per_class)
    {
        m_trainPath = train_path;
        m_testPath = test_path;
        m_trainSamples = train_samples;
        m_testSamples = test_samples;
        m_trainPerClass = train_per_class;
        m_testPerClass = test_per_class;    
    }
    std::string getTrainPath(){return m_trainPath;}
    std::string getTestPath(){return m_testPath;}
    int getTrainSamples(){return m_trainSamples;}
    int getTestSamples(){return m_testSamples;}
    int getTrainPerClass(){return m_trainPerClass;}
    int getTestPerClass(){return m_testPerClass;}
private:
    std::string m_trainPath;
    std::string m_testPath;
    int m_trainSamples;
    int m_testSamples;
    int m_trainPerClass;
    int m_testPerClass;
};

class ConfigTestEpoch
{
public:
	ConfigTestEpoch(int testEpoch)
	{
		m_testEpoch = testEpoch;
	}
	int getValue(){return m_testEpoch;}
private:
	int m_testEpoch;
};

class ConfigRotation
{
public:
	ConfigRotation(float rotation)
	{
		m_rotation = rotation;
	}
	float getValue()
	{
		return m_rotation;
	}
private:
	float m_rotation;

};

class ConfigDistortion
{
public:
	ConfigDistortion(float distortion)
	{
		m_distortion = distortion;
	}
	float getValue()
	{
		return m_distortion;
	}
private:
	float m_distortion;
};

class ConfigBatchSize
{
public:
	ConfigBatchSize(int batchSize)
	{
		m_batchSize = batchSize;
	}
	int getValue(){
		return m_batchSize;
	}
private:
	int m_batchSize;
};

class ConfigNonLinearity
{
public:
	ConfigNonLinearity(std::string poolMethod)
	{
		if(poolMethod == std::string("NL_SIGMOID")){
			m_nonLinearity = NL_SIGMOID;
		}else if(poolMethod == std::string("NL_TANH")){
			m_nonLinearity = NL_TANH;
		}else if(poolMethod == std::string("NL_RELU")){
			m_nonLinearity = NL_RELU;
		}else if(poolMethod == std::string("NL_LRELU")){
            m_nonLinearity = NL_LRELU;
        }else{
			m_nonLinearity = -1;
		}
	}
	int getValue(){return m_nonLinearity;}
private:
	int m_nonLinearity;
};


class ConfigChannels
{
public:
	ConfigChannels(int channels):m_channels(channels){};
	int m_channels;

	int getValue(){return m_channels;};
};

class ConfigBase
{
public:
	std::string m_name;
	std::string m_input;
	std::string m_subInput;
	std::vector<ConfigBase*> m_next;
	std::string m_type;
	float m_initW;
	int m_nonLinearity; 
	std::string m_initType;
	bool isGaussian(){
		return m_initType == std::string("Gaussian");
	}
    bool isBernoulli(){
        return m_initType == std::string("Bernoulli");
    }
    bool isFixed(){
        return m_initType == std::string("Fixed");
    }
    bool isExternal(){
        return m_initType == std::string("External");   
    }
	bool hasSubInput(){
		return m_subInput != std::string("NULL");
	}
	bool isBranchLayer(){
		return m_type == std::string("BRANCHLAYER");
	}
};

class ConfigLRN : public ConfigBase{
public:
	ConfigLRN(std::string name, 
		std::string input,
		std::string subInput,
		std::string type,
		float k, int n, 
		float alpha, float belta, 
		int nonLinearity):m_k(k), m_n(n), m_alpha(alpha), m_belta(belta){
		  m_nonLinearity = nonLinearity;
		  m_name  = name;
		  m_input = input; 
		  m_subInput = subInput;
		  m_type = type;
	  }
	float m_k;
	float m_alpha;
	float m_belta;
	int m_n;
};


class ConfigBranchLayer: public ConfigBase{
public:
	ConfigBranchLayer(std::string name, 
		std::string input,
		std::string subInput,
		std::string type,
		std::vector<std::string>&outputs){
			m_name = name;
			m_input = input;
			m_subInput = subInput;
			m_type = type;
			m_outputs = outputs;
	}
	std::vector<std::string>m_outputs;
};


class ConfigCombineLayer:public ConfigBase{
public:
	ConfigCombineLayer(std::string name, 
		std::vector<std::string> inputs,
		std::string subInput,
		std::string type){
			m_name = name;
			m_inputs = inputs;
			m_subInput = subInput;
			m_type = type;
			m_input = std::string("NULL");
	}
	std::vector<std::string>m_inputs;
};


class ConfigData:public ConfigBase{
public:
	ConfigData(std::string name,
		std::string type){
		m_name = name;
		m_type = type;
		m_input = std::string("NULL");
	}
};

class ConfigDataSpiking : public ConfigBase{
public:
    ConfigDataSpiking(std::string name,
        std::string type,
        int input_neurons){
        m_name = name;
		m_type = type;
		m_input = std::string("NULL");
        m_inputNeurons = input_neurons;	
    }
    int m_inputNeurons;
};


class ConfigConv : public ConfigBase
{
public:
	ConfigConv(std::string name, std::string input,
		std::string subInput, std::string type,
		int kernelSize, int padding, int amount,
		float weightDecay, float initW, std::string initType,
		int non_linearity){
		m_kernelSize = kernelSize;
		m_padding = padding;
		m_amount = amount;
		m_weightDecay = weightDecay;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_initW = initW;
		m_nonLinearity = non_linearity;
		m_initType = initType;
	}
	int m_kernelSize;
	int m_padding;
	int m_amount;
	float m_weightDecay;
};

class ConfigConvCFM : public ConfigBase
{
public:
	ConfigConvCFM(std::string name, std::string input,
		std::string subInput, std::string type,
		int kernelSize, int padding, int amount,
		float weightDecay, float initW, std::string initType,
		int non_linearity){
		m_kernelSize = kernelSize;
		m_padding = padding;
		m_amount = amount;
		m_weightDecay = weightDecay;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_initW = initW;
		m_nonLinearity = non_linearity;
		m_initType = initType;
	}
	int m_kernelSize;
    int m_padding;
    int m_amount;
    float m_weightDecay;
};

class ConfigLocal : public ConfigBase
{
public:
	ConfigLocal(std::string name,
		std::string input,
		std::string subInput,
		std::string type,
		int kernelSize,
		float weightDecay, float initW, std::string initType,
		int non_linearity){
			m_kernelSize = kernelSize;
			m_weightDecay = weightDecay;
			m_name = name;
			m_input = input;
			m_subInput = subInput;
			m_type = type;
			m_initW = initW;
			m_nonLinearity = non_linearity;
			m_initType = initType;
	}
	int m_kernelSize;
	float m_weightDecay;
};

/*
[
	LAYER = ONE;
	NAME  = one1;
	INPUT = conv3;
	WEIGHT_DECAY = 1e-6;
]
*/
class ConfigONE : public ConfigBase{
public:
	ConfigONE(std::string name, 
		std::string input, 
		std::string subInput,
		std::string type,
		float weightDecay){
		m_name = name;
		m_input= input;
		m_subInput = subInput;
		m_type = type;
		m_weightDecay = weightDecay;
	}
	float m_weightDecay;
};


/*
[
	LAYER = NIN;
	NAME  = nin1;
	INPUT = xxxx;
	WEIGHT_DECAY = 1e-6;
]
*/
class ConfigNIN : public ConfigBase{
public:
	ConfigNIN(std::string name, 
		std::string input, 
		std::string subInput,
		std::string type,
		float weightDecay,
		float initW,
		int non_linearity,
        int amount){
		m_name = name;
		m_input= input;
		m_subInput = subInput;
		m_type = type;
		m_weightDecay = weightDecay;
		m_initW = initW;
		m_nonLinearity = non_linearity;
        m_amount = amount;
        printf("~~~~%d\n", m_amount);
	}
	float m_weightDecay;
    int m_amount;
};

class ConfigPooling : public ConfigBase
{
public:
	ConfigPooling(std::string name,
		std::string input,
		std::string subInput,
		std::string type,
		std::string poolingType,
		int size, int skip,
		int non_linearity){
		m_size = size;
		m_skip = skip;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_nonLinearity = non_linearity;
		m_poolingType = poolingType;
	}
	int m_size;
	int m_skip;
	std::string m_poolingType;
};

class ConfigFC : public ConfigBase
{
public:
	ConfigFC(std::string name,
		std::string input,
		std::string subInput,
		std::string type,
		int numFullConnectNeurons,
		float weightDecay,
		float dropoutRate,
		float initW, 
		std::string initType, 
		int non_linearity)
	{
		m_numFullConnectNeurons = numFullConnectNeurons;
		m_weightDecay = weightDecay;
		m_dropoutRate = dropoutRate;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_initW = initW;
		m_nonLinearity = non_linearity;
		m_initType = initType;
	}
	int m_numFullConnectNeurons;
	float m_weightDecay;
	float m_dropoutRate;
};

class ConfigSoftMax : public ConfigBase
{
public:
	int m_numClasses;
	float m_weightDecay;
	ConfigSoftMax(std::string name,
		std::string input,
		std::string subInput,
		std::string type, 
		int numClasses, 
		float weightDecay, 
		float initW, 
		std::string initType,
		int non_linearity)
	{
		m_numClasses = numClasses;
		m_weightDecay = weightDecay;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_initW = initW;
		m_nonLinearity = non_linearity;
		m_initType = initType;
	}
};

class ConfigConvSpiking : public ConfigBase
{
public:
	ConfigConvSpiking(std::string name, std::string type, std::string input, int kernelSize, 
              int amount, int padding, float initW, std::string initType, float vth,
              float t_refrac, float tau_m, float tau_s, std::string weight_path,
              std::map<std::string, std::string> ref_paths)
    {
        m_kernelSize = kernelSize;
        m_padding = padding;
        m_amount = amount;
        m_name = name;
        m_input = input;
        m_type = type;
        m_initW = initW;
        m_initType = initType;
        m_vth = vth;
        m_t_ref = t_refrac;
        m_tau_m = tau_m;
        m_tau_s = tau_s;
        m_weightPath = weight_path;
        m_ref_weight_path = ref_paths[std::string("refWeightPath")];
        m_ref_output_train_path = ref_paths[std::string("refOutputTrainPath")];
        m_ref_output_test_path = ref_paths[std::string("refOutputTestPath")];
	}
    int m_kernelSize;
    int m_padding;
    int m_amount;
    float m_vth;
    int m_t_ref;
    float m_tau_m;
    float m_tau_s;
    std::string m_weightPath;
    std::string m_ref_weight_path;
    std::string m_ref_output_train_path;
    std::string m_ref_output_test_path;
};

class ConfigPoolingSpiking : public ConfigBase
{
public:
    ConfigPoolingSpiking(std::string name, std::string type, std::string input, int size, 
            int skip, float vth, float t_refrac, float tau_m, float tau_s, std::map<std::string, std::string> ref_paths)
    {
        m_name = name;
        m_input = input;
        m_type = type;
        m_size = size;
        m_skip = skip;
        m_vth = vth;
        m_t_ref = t_refrac;
        m_tau_m = tau_m;
        m_tau_s = tau_s;
        m_ref_output_train_path = ref_paths[std::string("refOutputTrainPath")];
        m_ref_output_test_path = ref_paths[std::string("refOutputTestPath")];
    }
    int m_size;
    int m_skip;
    float m_vth;
    int m_t_ref;
    float m_tau_m;
    float m_tau_s;
    std::string m_ref_output_train_path;
    std::string m_ref_output_test_path;
};


class ConfigSpiking : public ConfigBase
{
public:
	ConfigSpiking(std::string name, std::string type, std::string input, int num_classes,
        int num_neurons, float vth, float t_refrac, float tau_m, float tau_s,
        float initW, int weight_connect, std::string initType, std::string weight_path,
        std::string lweight_path, std::string laterial_type, std::string r_dim, 
        float local_inb_strength, float undesired_level, float desired_level, float margin, 
        std::map<std::string, std::string> ref_paths, bool has_bias, int dummy_freq)
    {
        m_name = name;
        m_type = type;
        m_input = input;
        m_numNeurons = num_neurons;
        m_classes = num_classes;
        m_vth = vth;
        m_t_ref = t_refrac;
        m_tau_m = tau_m;
        m_tau_s = tau_s;
        m_initW = initW;
        m_weightConnect = weight_connect;
        m_initType = initType;
        m_weightPath = weight_path;
        m_lweightPath = lweight_path;
        m_laterialType = laterial_type;
        m_reservoirDim = this->parseDim(r_dim);
        m_localInbStrength = local_inb_strength;
        m_undesired_level = undesired_level;
        m_desired_level = desired_level;
        m_margin = margin;
        m_ref_weight_path = ref_paths[std::string("refWeightPath")];
        m_ref_lweight_path = ref_paths[std::string("refLWeightPath")];
        m_ref_output_train_path = ref_paths[std::string("refOutputTrainPath")];
        m_ref_output_test_path = ref_paths[std::string("refOutputTestPath")];
        m_hasBias = has_bias;
        m_dummyFreq = dummy_freq;
	}
    bool hasLaterialWeight(){return m_laterialType != std::string("NULL");}
    bool hasLaterialInh(){return m_laterialType == std::string("LOCAL_INHIBITION");}
    std::vector<int> parseDim(std::string s){
        std::vector<int> dim;
        size_t pos = 0;
        std::string token;
        while((pos = s.find("x")) != std::string::npos) {
            token = s.substr(0, pos);
            dim.push_back(atoi(token.c_str()));
            s.erase(0, pos + 1);
        }
        dim.push_back(atoi(s.c_str()));
        return dim;
    }
    bool hasBias(){return m_hasBias;}
    int getBiasFreq(){return m_dummyFreq;}
    int m_numNeurons;
    int m_classes;
    float m_vth;
    int m_t_ref;
    float m_tau_m;
    float m_tau_s;
    int m_weightConnect;
    std::string m_weightPath;
    std::string m_lweightPath;
    std::string m_laterialType;
    std::vector<int> m_reservoirDim;
    float m_localInbStrength;
    float m_undesired_level;
    float m_desired_level;
    float m_margin;
    std::string m_ref_weight_path;
    std::string m_ref_lweight_path;
    std::string m_ref_output_train_path;
    std::string m_ref_output_test_path;
    bool m_hasBias;
    int m_dummyFreq;
};


class ConfigHorizontal
{
public:
	ConfigHorizontal(int horizontal)
	{
		m_horizontal = horizontal;
	}
	int getValue(){
		return m_horizontal;
	}
private:
	int m_horizontal;
};


class Config
{
public:
	void setMomentum(float _momentum){
		momentum = _momentum;
	}
	float getMomentum(){
		return momentum;
	}
	void setLrate(float _lrate){lrate = _lrate;}
	float getLrate(){return lrate;}

	void initPath(std::string path){
		m_path = path;
		init(m_path);
	}
	static Config* instance(){
		static Config* config = new Config();
		return config;
	}

	void clear(){

		delete  m_nonLinearity;
		delete  m_isGradientChecking;
        delete  m_hasBoostWeightTrain;
        delete  m_hasRecordWrong;
        delete  m_hasMarkTest;
        delete  m_useEffectRatio;
        delete  m_optimizer;
		delete  m_batchSize;
		delete  m_channels;

		delete m_crop;
		delete m_scale;
		delete m_rotation;
		delete m_distortion;
		delete m_imageShow;
		delete m_horizontal;
        delete m_dataset;
	}

	bool getImageShow(){
		return m_imageShow->getValue();}

	bool getIsGradientChecking(){
		return m_isGradientChecking->getValue();}

    bool hasRecordWrong(){
        return m_hasRecordWrong->getValue();
    }

    bool hasMarkTest(){
        return m_hasMarkTest->getValue();
    }
    bool hasBoostWeightTrain(){
        return m_hasBoostWeightTrain->getValue();
    }
    bool useEffectRatio(){
        return m_useEffectRatio->getValue();
    }
    float getLambda(){
        return m_weightReg->getLambda();
    }
    float getBeta(){
        return m_weightReg->getBeta();
    }
    float getWeightLimit(){
        return m_weightLimit->getLimit();
    }
    std::string getOptimizerType(){
        return m_optimizer->getType();
    }

	int getBatchSize(){
		return m_batchSize->getValue();}

	int getChannels(){
		return m_channels->getValue();
	}

	int getCrop(){
		return m_crop->getValue();
	}

	int getHorizontal(){
		return m_horizontal->getValue();
	}

	float getScale(){
		return m_scale->getValue();
	}

	float getRotation(){
		return m_rotation->getValue();
	}

	float getDistortion(){
		return m_distortion->getValue();
	}

    bool applyPreproc(){
        return (fabs(getDistortion() >= 0.1) || fabs(getScale() >= 1) || fabs(getRotation() >= 1));
    }

	const std::vector<ConfigBase*> getFirstLayers(){
		return m_firstLayers;
	}

	ConfigBase* getLayerByName(std::string name){
		if(m_layerMaps.find(name) != m_layerMaps.end()){
			return m_layerMaps[name];
		}
		else{
			char logStr[1024];
			sprintf(logStr, "layer %s does not exist\n", name.c_str());
			LOG(logStr, "Result/log.txt");
			exit(0);
		}
	}

	void insertLayerByName(std::string name, ConfigBase* layer){
		if(m_layerMaps.find(name) == m_layerMaps.end()){
			m_layerMaps[name] = layer;
		}
		else {
			char logStr[1024];
			sprintf(logStr, "layer %s exist\n", name.c_str());LOG(logStr, "Result/log.txt");
			exit(0);
		}
	}

	void setImageSize(int imageSize){
		m_imageSize = imageSize;
	}

	int getImageSize(){
		return m_imageSize;
	}

	int getTestEpoch(){
		return m_test_epoch->getValue();
	}

	float getWhiteNoise(){
		return m_white_noise->getValue();
	}

	int getClasses(){
		return m_classes;
	}
    
    int getEndTime(){
        return m_endTime;
    }
    
    std::string getTrainPath(){
        return m_dataset->getTrainPath();
    }

    std::string getTestPath(){
        return m_dataset->getTestPath();
    }

    int getTrainSamples(){
        return m_dataset->getTrainSamples();
    }

    int getTestSamples(){
        return m_dataset->getTestSamples();
    }

    int getTrainPerClass(){
        return m_dataset->getTrainPerClass();
    }

    int getTestPerClass(){
        return m_dataset->getTestPerClass();
    }

    void setEndTime(int end_time){
        m_endTime = end_time;
    }
	void setTraining(bool isTrainning){training = isTrainning;}
	bool isTraining(){return training;}
    bool hasDynamicThreshold(){return m_allowDynamicThreshold;}

private:
	void deleteComment();
	void deleteSpace();
	bool get_word_bool(std::string &str, std::string name);
	std::string get_word_type(std::string &str, std::string name);
	float get_word_float(std::string &str, std::string name);
	int get_word_int(std::string &str, std::string name);
	std::string read_2_string(std::string File_name);
	void get_layers_config(std::string &str);
	std::vector<std::string> get_name_vector(std::string &str, std::string name);

	void init(std::string path);
	std::string m_configStr;
	std::string m_path;

	std::map<std::string, ConfigBase*>m_layerMaps;
	std::vector<ConfigBase*>m_firstLayers;



	ConfigNonLinearity       *m_nonLinearity;
	ConfigGradient           *m_isGradientChecking;
    ConfigBoostWeight        *m_hasBoostWeightTrain;
    ConfigRecordWrong        *m_hasRecordWrong;
    ConfigMarkTest           *m_hasMarkTest;
    ConfigEffectRatio        *m_useEffectRatio;
    ConfigOptimizer          *m_optimizer;
    ConfigWeightReg          *m_weightReg;
    ConfigWeightLimit        *m_weightLimit;
	ConfigBatchSize          *m_batchSize;
	ConfigChannels           *m_channels;

	ConfigCrop               *m_crop;
	ConfigScale              *m_scale;
	ConfigRotation           *m_rotation;
	ConfigDistortion         *m_distortion;
	ConfigImageShow          *m_imageShow;
	ConfigHorizontal         *m_horizontal;
	ConfigTestEpoch          *m_test_epoch;
	ConfigWhiteNoise         *m_white_noise;
    ConfigDataset            *m_dataset;

	float momentum;
	float lrate;
	int m_imageSize;
	int m_classes;
    int m_endTime;
	bool training;
    bool m_allowDynamicThreshold;
};

#endif
