cmake_minimum_required (VERSION 3.5)

set(libname_rt "yolo_rt_lib")
project(${libname_rt})

#cuda
find_package(CUDA REQUIRED)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_HOST_COMPILATION_CPP ON)
    set(CUDA_NVCC_FLAGS -std=c++11 -g -Xcompiler -fexceptions -Xcompiler -fPIC)
    set(CUDA_SEPARABLE_COMPILATION ON)
elseif(MSVC)
#    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_HOST_COMPILATION_CPP ON)
#    set(CUDA_NVCC_FLAGS -std=c++11 -g -Xcompiler -fexceptions -Xcompiler -fPIC)
    set(CUDA_SEPARABLE_COMPILATION ON)
else()
#    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_HOST_COMPILATION_CPP ON)
    set(CUDA_NVCC_FLAGS -std=c++11 -g -Xcompiler -fexceptions -Xcompiler -fPIC)
    set(CUDA_SEPARABLE_COMPILATION ON)
endif()

set(CUDA_WARNING "cross-execution-space-call")
# new flags introduced in CUDA 9 set(CUDA_WARNING "reorder,cross-execution-
# space-call,deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror ${CUDA_WARNING} -restrict")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=compute_75")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(CUDNN REQUIRED)
find_package(TensorRT REQUIRED)
find_package(gflags REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${GFLAGS_INCLUDE_DIR})

file(GLOB TENSORRT_SOURCE_FILES *.cpp)
file(GLOB TENSORRT_HEADER_FILES *.h)

file(GLOB TENSORRT_CUDA_FILES *.cu)

cuda_add_library(${libname_rt} SHARED
    ${TENSORRT_CUDA_FILES}
    ${TENSORRT_SOURCE_FILES}
    ${TENSORRT_HEADER_FILES}
)

set(TENSORRT_LIBS
    ${OpenCV_LIBS}
    #${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    #${CUDA_CUDA_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
    # ${LIB_PTHREAD}
    ${TensorRT_LIBRARIES}
    ${GFLAGS_LIBRARIES}
)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(TENSORRT_LIBS ${TENSORRT_LIBS} stdc++fs nvinfer_plugin)
endif(CMAKE_COMPILER_IS_GNUCXX)

target_link_libraries(${libname_rt} ${TENSORRT_LIBS})
