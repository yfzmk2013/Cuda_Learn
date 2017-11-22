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
find_package(OpenCV)
#link_directories(/home/yanhao/lib/FaceApi/Lib/OpenBlas/lib)
#link_directories(/home/yanhao/tmp/FaceRecognitionAPI/Lib)
#
#
#option(USE_SSE "Set to ON to build use SSE" ON)
## Use SSE
#if (USE_SSE)
#    add_defInitions(-DUSE_SSE)
#    message(STATUS "Use SSE")
#    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
#    set(CUDA_NVCC_FLAGS "${CMAKE_CXX_FLAGS} -mavx -msse -msse2 -msse3 -mssse3 -msse4.1")
#endif ()


# Specify binary name and source file to build it from
#add_library(utils utils.cpp)


LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo  -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

link_libraries(/usr/local/cuda/lib64)

cuda_add_executable(
        mul_cublas
        test.cpp mul_cublas.cu)
#cuda_add_executable(
#        mul_cublas2
#        test.cpp )
##target_link_libraries(matrixmul ${OpenCV_LIBS})
target_link_libraries(mul_cublas cublas ${OpenCV_LIBS})

#target_link_libraries(mul_cublas2 /usr/local/cuda-8.0/lib64/libcublas.so ${OpenCV_LIBS})
cuda_add_executable(
        mul_cublas222
        cublasSgemm.cu mul_cublas.cu)
target_link_libraries(mul_cublas222 cublas ${OpenCV_LIBS})

#cuda_add_executable(cublas_test cublas_test.cpp)
#target_link_libraries(cublas_test /usr/local/cuda-8.0/lib64/libcublas.so)