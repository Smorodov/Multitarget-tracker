cmake_minimum_required(VERSION 3.9)

project(mdetection)

set(detector_sources
             BaseDetector.cpp
             MotionDetector.cpp
             BackgroundSubtract.cpp
             vibe_src/vibe.cpp
             Subsense/BackgroundSubtractorLBSP.cpp
             Subsense/BackgroundSubtractorLOBSTER.cpp
             Subsense/BackgroundSubtractorSuBSENSE.cpp
             Subsense/LBSP.cpp
             FaceDetector.cpp
             PedestrianDetector.cpp
             pedestrians/c4-pedestrian-detector.cpp
)

set(detector_headers
             BaseDetector.h
             MotionDetector.h
             BackgroundSubtract.h
             vibe_src/vibe.hpp
             Subsense/BackgroundSubtractorLBSP.h
             Subsense/BackgroundSubtractorLOBSTER.h
             Subsense/BackgroundSubtractorSuBSENSE.h
             Subsense/DistanceUtils.h
             Subsense/LBSP.h
             Subsense/RandUtils.h
             FaceDetector.h
             PedestrianDetector.h
             pedestrians/c4-pedestrian-detector.h
)

if (BUILD_YOLO_LIB)
    set(detector_sources ${detector_sources} YoloDarknetDetector.cpp)
    set(detector_headers ${detector_headers} YoloDarknetDetector.h)
endif()

if (BUILD_YOLO_TENSORRT)
    set(detector_sources ${detector_sources} YoloTensorRTDetector.cpp)
    set(detector_headers ${detector_headers} YoloTensorRTDetector.h)
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
include_directories(${PROJECT_SOURCE_DIR}/../common)

if (CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME} SHARED
	${detector_sources})
else(CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME}
	${detector_sources})
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
set(LIBS
    ${OpenCV_LIBS}
#    iconv
)
else(CMAKE_COMPILER_IS_GNUCXX)
set(LIBS
    ${OpenCV_LIBS}
)
endif()

if (BUILD_YOLO_LIB)
    set(LIBS ${LIBS} yolo_lib)
endif(BUILD_YOLO_LIB)

if (BUILD_YOLO_TENSORRT)
    set(LIBS ${LIBS} yolo_rt_lib)
endif(BUILD_YOLO_TENSORRT)


target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

#set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${detector_headers};../common/defines.h;../common/object_types.h")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${detector_headers}")
install(TARGETS ${PROJECT_NAME}
        EXPORT MTTrackingExports
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")