# Hybrid Macro/Micro Level Backpropagation for SNNs
This repo is the CUDA implementation of SNNs trained the hybrid macro/micro level backpropagation, modified based on <a href="https://github.com/zhxfl/CUDA-CNN">zhxfl</a> for spiking neuron networks.

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
##### GPU compute compatibility
* capability 6.0 for Titan XP, which is used for the authors. 


## Get Dataset
Get the MNIST dataset:
```sh
$ cd mm-bp-snn/mnist/
$ ./get_mnist.sh
```
Get the N-MNIST dataset by [the link](http://www.garrickorchard.com/datasets/n-mnist). Then unzip the ''Test.zip'' and ''Train.zip''. 

Run the matlab code: [NMNIST_Converter.m](https://github.com/jinyyy666/mm-bp-snn/tree/master/nmnist) in nmnist/

## Run the code 
* MNIST 
```sh
$ cd mm-bp-snn
$ ./build/CUDA-SNN 6 1
```
* N-MNIST 
```sh
$ cd mm-bp-snn
$ ./build/CUDA-SNN 7 1
```
* For Spiking-CNN, you need to enable the #define SPIKING_CNN in main.cpp, and recompile.
```sh
$ cd mm-bp-snn
$ ./build/CUDA-SNN 6 1
```

##### For Window user
Do the following to set up compilation environment.
* Install [Visual Stidio](https://www.visualstudio.com/downloads/) and [OpenCV](https://opencv.org/releases.html).
* When you create a new project using VS, You can find NVIDIA-CUDA project template, create a cuda-project.
* View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Device-> Code Generation-> compute_60,sm_60   
* View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Common-> Generate Relocatable Device Code-> Yes(-rdc=true) 
* View-> Property Pages-> Configuration Properties-> Linker-> Input-> Additional Dependencies-> libraries(-l)   
* View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Library search path(-L)  
* View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Include Directories(-I)  

# Notes
* The SNNs are implemented in terms of layers. User can config the SNNs by using configuration files in Config/
* The program will save the best test result and save the network weight in the file "Result/checkPoint.txt", If the program exit accidentally, you can continue the program form this checkpoint.
* The log for the reported performance of the three datasets and the correspoding checkout point files can be found in [Result](https://github.com/jinyyy666/mm-bp-snn/tree/master/Result) folder.
