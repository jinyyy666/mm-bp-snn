# Hybrid Macro/Micro Level Backpropagation for SNNs
This repo is the CUDA implementation of SNNs trained the hybrid macro/micro level backpropagation. I modified based original ANN version: <a href="https://github.com/zhxfl/CUDA-CNN">zhxfl</a> for spiking neuron networks.

# Dependencies and Libraries
* opencv
* cuda (suggest cuda 8.0)

You can compile the code on windows or linux.   
##### SDK include path(-I)   
* linux: /usr/local/cuda/samples/common/inc/ (For include file "helper_cuda"); /usr/local/include/opencv/ (Depend on situation)        
* windows: X:/Program Files (x86) /NVIDIA Corporation/CUDA Samples/v6.5/common/inc (For include file "helper_cuda"); X:/Program Files/opencv/vs2010/install/include (Depend on situation)

##### Library search path(-L)   
>* linux: /usr/local/lib/   
>* windows: X:/Program Files/opencv/vs2010/install/x86/cv10/lib (Depend on situation)    
>
##### libraries(-l)      
>* opencv_core   
>* opencv_highgui   
>* opencv_imgproc   
>* opencv_imgcodecs (need for opencv3.0)   
>* ***cublas***   
>* ***curand***   
>* ***cudadevrt***  

# Installation

The repo requires [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 8.0+ to run.
Please install the opencv and cuda before hand.

Install CMake and OpenCV
```sh
$ sudo apt-get install cmake libopencv-dev 
```

Checkout and compile the code:
```sh
$ git clone https://github.com/jinyyy666/mm-bp-snn.git
$ cd mm-bp-snn
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```

## Get Dataset
Get the MNIST dataset:
```sh
$ cd mm-bp-snn/mnist/
$ ./get_mnist.sh
```
Get the N-MNIST dataset by [the link](http://www.garrickorchard.com/datasets/n-mnist). Then unzip the ''Test.zip'' and ''Train.zip''. 
Run the matlab code: [NMNIST_Converter.m](https://github.com/jinyyy666/mm-bp-snn/tree/master/nmnist) in nmnist/
