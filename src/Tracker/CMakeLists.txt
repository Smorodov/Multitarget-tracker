cmake_minimum_required(VERSION 3.9)

project(mtracking)

set(main_sources
             ../mtracking/nms.h
             ../mtracking/defines.h
             ../mtracking/object_types.h
             ../mtracking/object_types.cpp)

set(tracker_sources
             BaseTracker.cpp
             track.cpp
             Kalman.cpp
             TrackerSettings.cpp
             ShortPathCalculator.cpp
             HungarianAlg/HungarianAlg.cpp
             LAPJV_algorithm/lap.cpp
             byte_track/BYTETracker.cpp
             byte_track/KalmanFilter.cpp
             byte_track/lapjv.cpp
             byte_track/STrack.cpp)

set(tracker_headers
             BaseTracker.h
             ShortPathCalculator.h
             track.h
             trajectory.h
             Kalman.h
             TrackerSettings.h
             HungarianAlg/HungarianAlg.h
             LAPJV_algorithm/lap.h
             EmbeddingsCalculator.hpp
             byte_track/BYTETracker.h
             byte_track/KalmanFilter.h
             byte_track/lapjv.h
             byte_track/STrack.h)

SOURCE_GROUP("Src" FILES ${main_sources})
SOURCE_GROUP("Tracker" FILES ${tracker_sources} ${tracker_headers})

include(CheckIncludeFileCXX)
check_include_file_cxx(opencv2/bgsegm.hpp HAVE_OPENCV_CONTRIB)
if(HAVE_OPENCV_CONTRIB)
    add_definitions(-DHAVE_OPENCV_CONTRIB)

    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" ON)

else(HAVE_OPENCV_CONTRIB)

    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" OFF)

endif(HAVE_OPENCV_CONTRIB)

option(USE_OCV_UKF "Should use the Unscented Kalman Filter from opencv_contrib?" OFF)

if(USE_OCV_UKF)
    add_definitions(-DUSE_OCV_UKF)
else()
    remove_definitions(-DUSE_OCV_UKF)
endif(USE_OCV_UKF)

if(USE_OCV_KCF)
    add_definitions(-DUSE_OCV_KCF)
else()
    remove_definitions(-DUSE_OCV_KCF)
endif(USE_OCV_KCF)

option(USE_OCV_EMBEDDINGS "Should use the embeddings from opencv_dnn + OpenVINO?" ON)
if(USE_OCV_EMBEDDINGS)
    add_definitions(-DUSE_OCV_EMBEDDINGS)
else()
    remove_definitions(-DUSE_OCV_EMBEDDINGS)
endif(USE_OCV_EMBEDDINGS)


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../src)
include_directories(${PROJECT_SOURCE_DIR}/../mtracking)
include_directories(${PROJECT_SOURCE_DIR}/../../thirdparty)

if (CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME} SHARED
        ${main_sources}
	${tracker_sources})
    set(LIBS
        ${OpenCV_LIBS}
        inih)
else(CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME}
        ${main_sources}
        ${tracker_sources})
    set(LIBS
        # ${OpenCV_LIBS}
        inih)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${tracker_headers};../mtracking/defines.h;../mtracking/object_types.h")
install(TARGETS ${PROJECT_NAME}
        EXPORT MTTrackingExports
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")
