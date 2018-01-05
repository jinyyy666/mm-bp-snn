#include "readNMnistData.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cuda_runtime_api.h>


//* find the data samples in the directory
void file_finder(const std::string& path, std::vector<std::pair<cuMatrix<bool>*, int> >& x, int sample_per_class, int end_time, int input_neurons)
{
    DIR *dir;
    struct dirent *ent; 
    struct stat st;

    dir = opendir(path.c_str());
    std::vector<int> quota(10, sample_per_class);
    
    while((ent = readdir(dir)) != NULL){
        std::string file_name = ent->d_name;
        std::string full_file_name = path + "/" + file_name;
        if(file_name[0] == '.') continue;
    
        if(stat(full_file_name.c_str(), &st) == -1) continue;

        bool is_directory = (st.st_mode & S_IFDIR) != 0;
        int num_of_samples = file_name.find("Train") != std::string::npos ? 60000 : 10000;
        if(sample_per_class != -1)  num_of_samples = sample_per_class * 10;
      
        if(is_directory){
            printf("Do not support recursively read the directories of directory anymore!");
            assert(0);
        }
        else{
            // this is indeed the data file:
            std::string suffix = ".dat";
            assert(file_name.length() >= suffix.length() && file_name.substr(file_name.length() - suffix.length()) == suffix);
            // handle the new data format
            size_t pos = file_name.find('_');
            assert(pos != string::npos);
            int cur_label = atoi(file_name.substr(pos+1).c_str());
            assert(cur_label >= 0 && cur_label < quota.size());
            
            read_each_nmnist_inside(full_file_name, x, end_time, input_neurons, cur_label, quota);
            printf("read %2d%%", 100 * int(x.size()) / num_of_samples);
        }
        printf("\b\b\b\b\b\b\b\b");
    }

}


//* read each sample of NMnist dataset
void read_each_nmnist(const std::string& filename, cuMatrixVector<bool>& x, int nrows, int ncols)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    cuMatrix<bool>* tpmat = new cuMatrix<bool>(nrows, ncols, 1);
    tpmat->freeCudaMem();
    int index = 0;
    std::string times;
    while(getline(f_in, times)){
        std::istringstream iss(times);
        int time;
        while(iss>>time){
            // tricky! the nmnist start from time = 0 but to match with our
            // CPU simulation, in CPU, we have shift 1 by one. We do the same here for GPU
            if(time + 1 >= nrows || index >= ncols) continue;
            tpmat->set(time + 1, index, 0, true);
        }
        index++;
    }
    x.push_back(tpmat); 
    f_in.close();
}


//* read each sample of NMnist dataset in terms of spikes time, do not translate them into binary
//* tricky: do this to avoid the issue that the data cannot be loaded into the main memory
void read_each_nmnist_inside(const std::string& filename, std::vector<std::pair<cuMatrix<bool>*, int > >& x, int end_time, int input_neurons, int cur_label, std::vector<int>& quota)
{
    std::ifstream f_in(filename.c_str());
    if(!f_in.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<std::endl;
        exit(EXIT_FAILURE);
    }
    std::string times;
    int index = 0;
    bool new_file = true;
    vector<vector<int> > * sp_time = NULL;
    while(getline(f_in, times)){
        if(times[0] == '#'){
            new_file = true;
            continue;
        }
        if(new_file){
            if(quota[cur_label] == 0)   break;
            if(quota[cur_label] > 0)    quota[cur_label]--;

            // prepare to read a new sample
            sp_time = new vector<vector<int> >(input_neurons, vector<int>());
            cuMatrix<bool>* tpmat = new cuMatrix<bool>(end_time, input_neurons, 1, sp_time);
            tpmat->freeCudaMem();
            x.push_back({tpmat, cur_label}); 
            
            index = 0;
            new_file = false;
        }
        std::istringstream iss(times);
        int time;
        // each line start with the input neuron index (1 based)
        iss>>index;
        assert(index> 0 && index <= input_neurons);
        index--;
        while(iss>>time){
            // tricky! the nmnist start from time = 0 but to match with our
            // CPU simulation, in CPU, we have shift 1 by one. We do the same here for GPU
            if(time + 1 >= end_time || index >= input_neurons) continue;
            (*sp_time)[index].push_back(time+1);
        }
    }

    f_in.close();
}




//* read the train data and label of the NMnist at the same time
int readNMnist(
        std::string path, 
        std::vector<std::pair<cuMatrix<bool>*, int> > & x,
        int sample_per_class,
        int input_neurons,
        int end_time)
{
    //* read the data from the path
    struct stat sb;
    if(stat(path.c_str(), &sb) != 0){
        std::cout<<"The given path: "<<path<<" does not exist!"<<std::endl;
        exit(EXIT_FAILURE);
    } 

    if(path[path.length() - 1] == '/')  path = path.substr(0, path.length() - 1);
    //* read the samples in the directory
    file_finder(path, x, sample_per_class, end_time, input_neurons);

    //* random shuffle the train data, tricky! 
    //* this is very important because the datafile are stored in ordered, but we want it
    //* to be random when training
    random_shuffle(x.begin(), x.end());
    return x.size();
}

//* read the label
int readNMnistLabel(const std::vector<std::pair<cuMatrix<bool>*, int> >& collect, cuMatrix<int>* &mat){
    for(int i = 0; i < collect.size(); ++i){
        mat->set(i, 0, 0, collect[i].second);
    }
    mat->toGpu();
    return collect.size();
}


//* read trainning data and lables
int readNMnistData(
        cuMatrixVector<bool>& x,
        cuMatrix<int>*& y, 
        std::string path,
        int samples_per_class,
        int input_neurons,
        int end_time)
{

    std::vector<std::pair<cuMatrix<bool>*, int> > collect;
    int len = readNMnist(path, collect, samples_per_class, input_neurons, end_time);
    //* read MNIST sample into cuMatrixVector
    for(int i = 0; i < collect.size(); ++i) x.push_back(collect[i].first);

    //* read MNIST label into cuMatrix
    y = new cuMatrix<int>(len, 1, 1);
    int t = readNMnistLabel(collect, y);
    return t;
}
