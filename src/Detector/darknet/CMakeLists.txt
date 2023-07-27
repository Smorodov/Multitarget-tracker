cmake_minimum_required (VERSION 3.10)

set(libname "yolo_lib")
project(${libname})

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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_72,code=sm_72")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=compute_75")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(CUDNN REQUIRED)

# target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

# target_compile_definitions(${PROJECT_NAME} PUBLIC DKGPU)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/stb/include)

if (MSVC)
    if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
        set(BIT_SYSTEM x32)
    else()
        set(BIT_SYSTEM x64)
    endif()

    include_directories(${PROJECT_SOURCE_DIR}/3rdparty/include)
    set(LINK_DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/lib/${BIT_SYSTEM}) 
    link_directories(${LINK_DIRECTORY})

    add_definitions(-DHAVE_STRUCT_TIMESPEC)
    add_definitions(-DWIN32)
    set(LIB_PTHREAD pthreadVC2)
else()
    set(LIB_PTHREAD pthread)
endif()

file(GLOB DARKNET_SOURCE_FILES src/*.c)
set(DARKNET_SOURCE_FILES ${DARKNET_SOURCE_FILES} src/yolo_v2_class.cpp src/http_stream.cpp src/image_opencv.cpp)

file(GLOB DARKNET_HEADER_FILES src/*.h)
set(DARKNET_HEADER_FILES ${DARKNET_HEADER_FILES} include/darknet.h include/yolo_v2_class.hpp)

file(GLOB DARKNET_CUDA_FILES src/*.cu)

add_definitions(-DYOLO_DETECTOR_EXPORTS)
add_definitions(-DLIB_EXPORTS)
add_definitions(-DGPU)
add_definitions(-DCUDNN)
# add_definitions(-DOPENCV)

#if(NOT CMAKE_DEBUG_POSTFIX)
#  set(CMAKE_DEBUG_POSTFIX d)
#endif()

cuda_add_library(${libname} SHARED
    ${DARKNET_CUDA_FILES}
    ${DARKNET_SOURCE_FILES}
    ${DARKNET_HEADER_FILES}
)

set(YOLO_LIBS
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
    ${LIB_PTHREAD}
)

# message("${YOLO_LIBS}")

target_link_libraries(${libname} ${YOLO_LIBS})

install(TARGETS ${libname}
        EXPORT MTTrackingExports
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

if (MSVC)
    install(FILES "${LINK_DIRECTORY}/${LIB_PTHREAD}.lib"           
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")