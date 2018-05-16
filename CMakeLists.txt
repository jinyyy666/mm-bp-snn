cmake_minimum_required (VERSION 3.0)
project (CUDA-SNN)

# The version number.
set (CUDA-SNN_VERSION_MAJOR 2)
set (CUDA-SNN_VERSION_MINOR 0)

find_package (CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
MESSAGE("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
include_directories ("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
set(CUDA_SEPARABLE_COMPILATION ON)

# opencv
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )
set(OpenCV_LIBRARIES ${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -rdynamic")


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_60,code=sm_60
  -O3
  --use_fast_math
#  -G
#  -g
  -rdc=true
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )

MESSAGE(${CUDA_NVCC_FLAGS})

file(GLOB CUDA_SNN_CU_SOURCES  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    "readData/*" 
    "layers/*"
    "dataAugmentation/*" 
    "common/*" 
    "main.cpp"
    "net.*"
    "net_spiking.*")

cuda_add_executable(CUDA-SNN ${CUDA_SNN_CU_SOURCES})
target_link_libraries(CUDA-SNN
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_LIBRARIES}
  )
CUDA_ADD_CUBLAS_TO_TARGET(CUDA-SNN)

