cmake_minimum_required(VERSION 3.5)

project(MTTracking)

unset(CMAKE_CXX_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS_RELEASE CACHE)
unset(CMAKE_CXX_FLAGS_DEBUG CACHE)

find_package(OpenMP)
if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors -std=c++14" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native -mtune=native -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /Ox /Ob2 /Oi /Ot /arch:AVX /fp:fast /DNDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /Od /Ob0 /DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
endif()

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

FIND_PACKAGE(OpenCV REQUIRED)

option(SILENT_WORK "Remove all imshow and waitKey functions?" OFF)
if (SILENT_WORK)
    add_definitions(-DSILENT_WORK)
endif(SILENT_WORK)

option(BUILD_EXAMPLES "Should compiled examples (motion detection, pedestrians, faces, DNNs etc)?" ON)
if (BUILD_EXAMPLES)
    add_subdirectory(example)
endif(BUILD_EXAMPLES)

option(BUILD_CARS_COUNTING "Should compiled Cars counting example?" OFF)
if (BUILD_CARS_COUNTING)
    add_subdirectory(cars_counting)
endif(BUILD_CARS_COUNTING)

option(BUILD_ASYNC_DETECTOR "Should compiled async example with low fps Detector?" OFF)
if (BUILD_ASYNC_DETECTOR)
    add_subdirectory(async_detector)
endif(BUILD_ASYNC_DETECTOR)

option(BUILD_YOLO_LIB "Should compiled standalone yolo_lib with original darknet?" OFF)
if (BUILD_YOLO_LIB)
    add_subdirectory(src/Detector/darknet)
    add_definitions(-DBUILD_YOLO_LIB)

if (MSVC)
    if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
        set(BIT_SYSTEM x32)
    else()
        set(BIT_SYSTEM x64)
    endif()

    set(LIB_PTHREAD pthreadVC2)
else()
    set(LIB_PTHREAD pthread)
endif()

if (MSVC)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Detector/darknet/3rdparty/dll/${BIT_SYSTEM}/pthreadVC2.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Detector/darknet/3rdparty/dll/${BIT_SYSTEM}/pthreadVC2.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)

    file(GLOB CUDNN_DLL ${CUDNN_DIR}/bin/*.dll)
    file(COPY ${CUDNN_DLL} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    file(COPY ${CUDNN_DLL} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
endif()

endif(BUILD_YOLO_LIB)

option(BUILD_YOLO_TENSORRT "Should compiled TensorRT binding for YOLO?" OFF)
if (BUILD_YOLO_TENSORRT)
    add_subdirectory(src/Detector/tensorrt_yolo)
    add_definitions(-DBUILD_YOLO_TENSORRT)
endif(BUILD_YOLO_TENSORRT)


add_subdirectory(src)
