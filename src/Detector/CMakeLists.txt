cmake_minimum_required(VERSION 3.16)

project(mdetection)

set(detector_sources
             BaseDetector.cpp
             MotionDetector.cpp
             BackgroundSubtract.cpp
             vibe_src/vibe.cpp)

set(detector_headers
             BaseDetector.h
             MotionDetector.h
             BackgroundSubtract.h
             vibe_src/vibe.hpp)

if (BUILD_ONNX_TENSORRT)
    set(detector_sources ${detector_sources} ONNXTensorRTDetector.cpp)
    set(detector_headers ${detector_headers} ONNXTensorRTDetector.h)
endif()

option(USE_OCV_DNN "Use OpenCV DNN module?" ON)
if (USE_OCV_DNN)
    set(detector_sources ${detector_sources} OCVDNNDetector.cpp)
    set(detector_headers ${detector_headers} OCVDNNDetector.h)
    add_definitions(-DUSE_OCV_DNN)
endif()

SOURCE_GROUP("Detector" FILES ${detector_sources} ${detector_headers})

include(CheckIncludeFileCXX)
check_include_file_cxx(opencv2/bgsegm.hpp HAVE_OPENCV_CONTRIB)
if(HAVE_OPENCV_CONTRIB)
    add_definitions(-DHAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" ON)

else(HAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" OFF)

endif(HAVE_OPENCV_CONTRIB)

if(USE_OCV_BGFG)
    add_definitions(-DUSE_OCV_BGFG)
else()
    remove_definitions(-DUSE_OCV_BGFG)
endif(USE_OCV_BGFG)


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../src)
include_directories(${PROJECT_SOURCE_DIR}/..)

add_library(${PROJECT_NAME} ${MT_LIBRARY_TYPE}
	${detector_sources})

# OpenCV types appear in the public headers, so it is a PUBLIC dependency;
# the threading/OpenMP runtimes and yolo_rt_lib stay private (roadmap M21, M22).
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${OpenCV_LIBS}
        mtracking_core
    PRIVATE
        Threads::Threads)

if (BUILD_ONNX_TENSORRT)
    target_link_libraries(${PROJECT_NAME} PRIVATE yolo_rt_lib)
endif(BUILD_ONNX_TENSORRT)

if (OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Headers are installed flat into include/<project>; expose that root to
# consumers of the exported CMake package (roadmap 4.4).
target_include_directories(${PROJECT_NAME} INTERFACE
    $<INSTALL_INTERFACE:include>)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${detector_headers}")
install(TARGETS ${PROJECT_NAME}
        EXPORT MTTrackingExports
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")
