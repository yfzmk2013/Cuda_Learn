#CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#PROJECT(Cuda_Learn)
#
##FIND_PACKAGE(CUDA REQUIRED)
#find_package(CUDA QUIET REQUIRED)
##FIND_PACKAGE(MPI REQUIRED)
#
#INCLUDE(FindCUDA)
#
#INCLUDE_DIRECTORIES(/usr/local/cuda/include)
#
#FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")
##CUDA_ADD_EXECUTABLE(lbmslv )
#
#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
#
#LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo  -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
#
#
#CUDA_ADD_EXECUTABLE(lbmslv has_gpu.cpp)
#CUDA_ADD_EXECUTABLE(testGPU GPU_mx.cpp gg.cu)
#
#
#TARGET_LINK_LIBRARIES(lbmslv /usr/local/cuda/lib64/libcudart.so)
#TARGET_LINK_LIBRARIES(testGPU ${CUDA_LIBRARIES})




# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)




# Specify binary name and source file to build it from
#add_library(utils utils.cpp)
cuda_add_executable(
        squaresum
        test.cpp squaresum.cu utils.cu)
#target_link_libraries(squaresum utils)

