#include "readSpeechData.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cuda_runtime_api.h>


//* the BSA algorithm for encoding the continuous value into the spike trains
void BSA(const std::vector<float>& analog, int input_channel, int step_analog, int step_spikeT, vector<vector<int> >* & mat, int end_time)
{
    int length_kernel = 24;
    int length_signal = (analog.size()*step_analog)/step_spikeT + 24;
    double threshold = 0.6;
    double error1, error2, temp;
    double * kernel = new double[length_kernel];
    double * signal = new double[length_signal];

    for(int i = 0; i < length_kernel; i++) kernel[i] = exp(-(i-double(length_kernel)/2)*(i-double(length_kernel)/2)/25);
    temp = 0;
    for(int i = 0; i < length_kernel; i++) temp += kernel[i];
    for(int i = 0; i < length_kernel; i++) kernel[i] /= temp;

    int index = 0;
    for(int i = 0; i < analog.size(); i++)
        for(; index < (i+1)*step_analog/step_spikeT; index++)
            signal[index] = analog[i]*2e3;

    for(; index < length_signal; index++) signal[index] = 0;

    int j;
    for(int i = 0; i < length_signal-24; i++){
        error1 = 0;
        error2 = 0;
        for(j = 0; j < length_kernel; j++){
            temp = signal[i+j] - kernel[j];
            error1 += (temp<0) ? -temp : temp;
            temp = signal[i+j];
            error2 += (temp<0) ? -temp : temp;
        }
        if(error1 < (error2-threshold)){
            // set the spike time matrix:
            int time = i + 1;
            if(time < end_time)  (*mat)[input_channel].push_back(time);

            for(j = 0; j < length_kernel; j++) signal[i+j] -= kernel[j];
        }
    }

    delete [] kernel;
    delete [] signal;
}

//* recursively find the files
void file_finder(const std::string& path, cuMatrixVector<bool>& x, std::vector<int>& labels, int cur_label, int& sample_count, int num_of_samples, int end_time, int input_neurons, int CLS)
{
    DIR *dir;
    struct dirent *ent; 
    struct stat st;

    dir = opendir(path.c_str());
    while((ent = readdir(dir)) != NULL){
        if(sample_count >= num_of_samples)  return;

        std::string file_name = ent->d_name;
        std::string full_file_name = path + "/" + file_name;
        if(file_name[0] == '.') continue;
    
        if(stat(full_file_name.c_str(), &st) == -1) continue;

        bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if(file_name.length() <= 2){
            cur_label = atoi(file_name.c_str());
            assert(cur_label >= 0 && cur_label < CLS);
        }
        
        if(is_directory){
            assert(cur_label >= 0 && cur_label < CLS);
            file_finder(full_file_name, x, labels, cur_label, sample_count, num_of_samples, end_time, input_neurons, CLS); 
        }
        else{
            // this is indeed the data file:
            string suffix = ".dat";
            assert(file_name.length() >= suffix.length() && file_name.substr(file_name.length() - suffix.length()) == suffix);
            read_each_speech(full_file_name, x, end_time, input_neurons);

            labels.push_back(cur_label);
            sample_count++;
            printf("read %2d%%", 100 * sample_count / num_of_samples);
        }
        printf("\b\b\b\b\b\b\b\b");
    }

}


//* read the train data and label of speeches at the same time
int readSpeech(
        std::string path, 
        cuMatrixVector<bool>& x,
        std::vector<int>& labels,
        int num,
        int input_neurons,
        int end_time,
        int CLS)
{
    //* read the data from the path
    struct stat sb;
    if(stat(path.c_str(), &sb) != 0){
        std::cout<<"The given path: "<<path<<" does not exist!"<<std::endl;
        exit(EXIT_FAILURE);
    } 

    if(path[path.length() - 1] == '/')  path = path.substr(0, path.length() - 1);
    //* recursively read the samples in the directory
    int sample_count = 0;
    file_finder(path, x, labels, -1, sample_count, num, end_time, input_neurons, CLS);
    assert(x.size() == num);
    assert(x.size() == labels.size());

    return x.size();
}

//* read each sample of Speech dataset
void read_each_speech(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    // get all the analog values of the speeches
    std::vector<std::vector<float> > spectrum;
    std::string analogs;
    while(getline(f_in, analogs)){
        std::istringstream iss(analogs);
        float analog;
        vector<float> tmp;
        while(iss>>analog)  tmp.push_back(analog);
        spectrum.push_back(tmp);
    }
    f_in.close();
    if(spectrum.size() != ncols){
        std::cout<<"The number of channels in the raw speech file: "<<spectrum.size()
                 <<" does not match the number of input neuron: "<<ncols<<std::endl;
        exit(EXIT_FAILURE);
    }

    // perform the BSA algorithm for each channel of the spectrum
    vector<vector<int> > * sp_time = new vector<vector<int> >(ncols, vector<int>());
    int end_time = nrows;
    for(int c = 0; c < spectrum.size(); ++c){
        BSA(spectrum[c], c, 10, 1, sp_time, end_time);
    }
    cuMatrix<bool>* tpmat = new cuMatrix<bool>(nrows, ncols, 1, sp_time);
    tpmat->freeCudaMem(); 
    x.push_back(tpmat);
}


//* read each speech from the dump file of the CPU simulator
void read_each_speech_dump(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    cuMatrix<bool>* tpmat = new cuMatrix<bool>(nrows, ncols, 1);
    tpmat->freeCudaMem();
    
    int index, spike_time;
    f_in>>index>>spike_time; // get rid of -1   -1 at the beginning
    while(f_in>>index>>spike_time){
        if(index == -1 && spike_time == -1) break; // only read one iteration of speech
        assert(index < ncols);
        if(spike_time >= nrows) continue;
        tpmat->set(spike_time, index, 0, true);
    }   
    f_in.close();
    x.push_back(tpmat);
}

//* read the dumped input of CPU as a spike time matrix
void read_dumped_input_inside(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    vector<vector<int> > * sp_time = new vector<vector<int> >(ncols, vector<int>()); 
    int index, spike_time;
    f_in>>index>>spike_time; // get rid of -1   -1 at the beginning
    while(f_in>>index>>spike_time){
        if(index == -1 && spike_time == -1) break; // only read one iteration of speech
        assert(index < ncols);
        if(spike_time >= nrows) continue;
        assert(spike_time > ((*sp_time)[index].empty() ? -1 : (*sp_time)[index].back()));

        (*sp_time)[index].push_back(spike_time);
    }   
    f_in.close();
    cuMatrix<bool>* tpmat = new cuMatrix<bool>(nrows, ncols, 1, sp_time);
    tpmat->freeCudaMem();
    x.push_back(tpmat);
}

//* read the label
int readSpeechLabel(const std::vector<int>& labels, cuMatrix<int>* &mat){
    for(int i = 0; i < labels.size(); ++i){
        mat->set(i, 0, 0, labels[i]);
    }
    mat->toGpu(); 
    return labels.size();
}

//* read trainning data and lables
int readSpeechData(
        cuMatrixVector<bool>& x,
        cuMatrix<int>*& y, 
        std::string path,
        int number_of_images,
        int input_neurons,
        int end_time,
        int CLS)
{

    std::vector<int> labels;
    int len = readSpeech(path, x, labels, number_of_images, input_neurons, end_time, CLS);
    //* read speech label into cuMatrix
    y = new cuMatrix<int>(len, 1, 1);
    int t = readSpeechLabel(labels, y);
    return t;
}
