#include "readMnistData.h"
#include <string>
#include <fstream>
#include <vector>
#include <cuda_runtime_api.h>
#include <vector>
#include <random>



int checkError(int x)
{
	int mark[] = {2426,3532,4129,4476,7080,8086,9075,10048,10800,10994,
		12000,12132,12830,14542,15766,16130,20652,21130,23660,23886,
		23911,25562,26504,26560,30792,33506,35310,36104,38700,39354,
		40144,41284,42616,43109,43454,45352,49960,51280,51442,53396,
		53806,53930,54264,56596,57744,58022,59915};
	for(int i = 0; i < 47; i++)
	{
		if(mark[i] == x)
			return true;
	}
	return false;
}

/*reverse the int*/
int ReverseInt (int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int) ch1 << 24) | ((int)ch2 << 16) | ((int)ch3 << 8) | ch4;
}

/*read the number of images, n_rows, n_cols from the mnist file*/
void readFileInform(std::ifstream& file, int& number_of_images, int& n_rows, int& n_cols, int num)
{
    int magic_number = 0;
    file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*) &number_of_images,sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    if(number_of_images >= num){
        number_of_images = num;
    }
    else{
        printf("readFileInform::number of images is overflow\n");
        exit(0);
    }
    file.read((char*) &n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    file.read((char*) &n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols); 
}


/*read the train data*/
int read_Mnist(std::string filename, 
	cuMatrixVector<float>& vec,
	int num,
	int flag){
		/*read the data from file*/
		std::ifstream file(filename.c_str(), std::ios::binary);
		int id = 0;
		if (file.is_open()){
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
            readFileInform(file, number_of_images, n_rows, n_cols, num);

			for(int i = 0; i < number_of_images; ++i){
				cuMatrix<float>* tpmat = new cuMatrix<float>(n_rows, n_cols, 1);
				tpmat->freeCudaMem();
				for(int r = 0; r < n_rows; ++r){
					for(int c = 0; c < n_cols; ++c){
						unsigned char temp = 0;
						file.read((char*) &temp, sizeof(temp));
						tpmat->set(r, c, 0, (float)temp * 2.0f / 255.0f - 1.0f);
					}
				}
				//tpmat->toGpu();
				if(!flag){
					if(!checkError(id)){
						vec.push_back(tpmat);
					}
					else {
						printf("train data %d\n", id);
					}
				}
				else {
					vec.push_back(tpmat);
				}
				id++;
			}
		}
		//vec.toGpu();
		return vec.size();
}

/*read the lable*/
int read_Mnist_Label(std::string filename, 
	cuMatrix<int>* &mat,
    int num,
	int flag){
		std::ifstream file(filename.c_str(), std::ios::binary);
		if (file.is_open()){
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
            readFileInform(file, number_of_images, n_rows, n_cols, num);

			int id = 0;
			for(int i = 0; i < number_of_images; ++i){
				unsigned char temp = 0;
				file.read((char*) &temp, sizeof(temp));
				if(!flag){
					if(!checkError(i)){
						mat->set(id, 0, 0, temp);
						id++;
					}
					else {
						printf("train label %d\n", i);
					}
				}
				else {
					mat->set(i, 0, 0, temp);
					id++;
				}
			}
			mat->toGpu();
            file.close();

			if(!flag) return id;
			else return number_of_images;
			return id;
		}
		return 0;
}

/*read trainning data and lables*/
int readMnistData(cuMatrixVector<float>& x,
	cuMatrix<int>*& y, 
	std::string xpath,
	std::string ypath, 
	int number_of_images,
	int flag)
{
    /*read MNIST iamge into cuMatrix*/
    int len = read_Mnist(xpath, x, number_of_images, flag);
    /*read MNIST label into cuMatrix*/
    y = new cuMatrix<int>(len, 1, 1);
    int t = read_Mnist_Label(ypath, y, number_of_images, flag);
    return t;
}


void readMnistImg(const std::string& filename, std::vector<std::vector<std::vector<float> > >& data, int num)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
	int id = 0;
    if(!file.is_open()){
        std::cout<<"Cannot open the file: "<<filename<<endl;
        assert(0);
    }
	int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    readFileInform(file, number_of_images, n_rows, n_cols, num);

    for(int i = 0; i < number_of_images; ++i){
        std::vector<std::vector<float> > tpmat(std::vector<std::vector<float> >(n_rows, std::vector<float>(n_cols, 1)));
        for(int r = 0; r < n_rows; ++r){
            for(int c = 0; c < n_cols; ++c){
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                tpmat[r][c] = (float)temp / (5.5 * 255.0f);
            }
        }
        data.push_back(tpmat);
    }
    file.close();
}

void generatePoissonSpikes(
    cuMatrixVector<bool>& x, 
    const std::vector<std::vector<std::vector<float> > >& data,
    int input_neurons,
    int end_time)
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);
    for(int i = 0; i < data.size(); ++i){
        std::vector<std::vector<int> > * sp_time = new std::vector<std::vector<int> >(input_neurons, std::vector<int>());
        int index = 0;
        for(int j = 0; j < data[i].size(); ++j){
            for(int k = 0; k < data[i][j].size(); ++k){
                float freq = data[i][j][k];
                if(fabsf(freq - 0.0f) < 1e-5)   continue;
                for(int time = 1; time < end_time; ++time){
                    if(dist(e2) < freq)    (*sp_time)[index].push_back(time);
                }
                index++;
            }
        }
        cuMatrix<bool>* tpmat = new cuMatrix<bool>(end_time, input_neurons, 1, sp_time);
        tpmat->freeCudaMem();
        x.push_back(tpmat);
    } 
} 

int readSpikingMnist(
    const std::string& filename, 
    cuMatrixVector<bool>& x,
    int num, 
    int input_neurons,
    int end_time)
{
    vector<vector<vector<float> > > data;
    readMnistImg(filename, data, num);
    assert(!data.empty() && data[0].size() * data[0].size() == input_neurons);

    generatePoissonSpikes(x, data, input_neurons, end_time);
    return x.size(); 
}


/*read the MNIST and produce the poisson spike trains*/
int readSpikingMnistData(
        cuMatrixVector<bool>& x,
        cuMatrix<int>*& y, 
        std::string xpath,
        std::string ypath,
        int number_of_images,
        int input_neurons,
        int end_time)
{
    int len = readSpikingMnist(xpath, x, number_of_images, input_neurons, end_time);
    //* read MNIST label into cuMatrix
    y = new cuMatrix<int>(len, 1, 1);
    int t = read_Mnist_Label(ypath, y, number_of_images, 1);
    assert(len == t);
    return t;
}
