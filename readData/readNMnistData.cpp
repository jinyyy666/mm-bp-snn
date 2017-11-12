#include "readNMnistData.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cuda_runtime_api.h>


//* recursively find the files
void file_finder(const std::string& path, cuMatrixVector<bool>& x, std::vector<int>& labels, int cur_label, int& sample_count, int num_of_samples, int end_time, int input_neurons)
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
        if(file_name.length() == 1){
            cur_label = atoi(file_name.c_str());
            assert(cur_label >= 0 && cur_label <= 9);
        }
        
        if(is_directory){
            assert(cur_label >= 0 && cur_label <= 9);
            file_finder(full_file_name, x, labels, cur_label, sample_count, num_of_samples, end_time, input_neurons); 
        }
        else{
            // this is indeed the data file:
            std::string suffix = ".dat";
            assert(file_name.length() >= suffix.length() && file_name.substr(file_name.length() - suffix.length()) == suffix);
            read_each_nmnist(full_file_name, x, end_time, input_neurons);
            labels.push_back(cur_label);
            sample_count++;
            printf("read %2d%%", 100 * sample_count / num_of_samples);
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


//* read the train data and label of the NMnist at the same time
int readNMnist(
        std::string path, 
        cuMatrixVector<bool>& x,
        std::vector<int>& labels,
        int num,
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
    //* recursively read the samples in the directory
    int sample_count = 0;
    file_finder(path, x, labels, -1, sample_count, num, end_time, input_neurons);
    assert(x.size() == num);
    assert(x.size() == labels.size());

    return x.size();
}


//* read the label
int readNMnistLabel(const std::vector<int>& labels, cuMatrix<int>* &mat){
    for(int i = 0; i < labels.size(); ++i){
        mat->set(i, 0, 0, labels[i]);
    }
    mat->toGpu();
    return labels.size();
}


//* read trainning data and lables
int readNMnistData(
        cuMatrixVector<bool>& x,
        cuMatrix<int>*& y, 
        std::string path,
        int number_of_images,
        int input_neurons,
        int end_time)
{

    std::vector<int> labels;
    int len = readNMnist(path, x, labels, number_of_images, input_neurons, end_time);
    //* read MNIST label into cuMatrix
    y = new cuMatrix<int>(len, 1, 1);
    int t = readNMnistLabel(labels, y);
    return t;
}
