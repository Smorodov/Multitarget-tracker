cmake_minimum_required (VERSION 3.10)

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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_72,code=sm_72")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=compute_75")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(CUDNN REQUIRED)
find_package(TensorRT REQUIRED)

message("TensorRT major version: " ${TensorRT_VERSION_MAJOR})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${TensorRT_INCLUDE_DIRS})

file(GLOB TENSORRT_SOURCE_FILES *.cpp common/*.cpp)
file(GLOB TENSORRT_HEADER_FILES *.h* common/*.h*)

file(GLOB TENSORRT_CUDA_FILES *.cu)

cuda_add_library(${libname_rt} SHARED
    ${TENSORRT_CUDA_FILES}
    ${TENSORRT_SOURCE_FILES}
    ${TENSORRT_HEADER_FILES}
)

#message("TensorRT OpenCV libraries:")
#message("${OpenCV_LIBS}")
#message(${OpenCV_DIR})

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
)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(TENSORRT_LIBS ${TENSORRT_LIBS} stdc++fs nvinfer_plugin nvonnxparser)
endif(CMAKE_COMPILER_IS_GNUCXX)

target_link_libraries(${libname_rt} ${TENSORRT_LIBS})

install(TARGETS ${libname_rt}
        EXPORT MTTrackingExports
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")