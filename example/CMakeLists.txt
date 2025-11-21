cmake_minimum_required (VERSION 3.5)

project(MultitargetTracker)

set(SOURCES main.cpp
            VideoExample.cpp)

set(HEADERS MouseExample.h
            VideoExample.h
            examples.h
            MotionDetectorExample.h
            FileLogger.h)

if (BUILD_CARS_COUNTING)
    set(SOURCES ${SOURCES} CarsCounting.cpp)
    set(HEADERS ${HEADERS} CarsCounting.h)
endif(BUILD_CARS_COUNTING)

# ----------------------------------------------------------------------------
# добавляем include директории
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../src
                    ${PROJECT_SOURCE_DIR}/../src/mtracking
                    ${PROJECT_SOURCE_DIR}/../src/Detector
                    ${PROJECT_SOURCE_DIR}/../src/Detector/vibe_src
                    ${PROJECT_SOURCE_DIR}/../src/Detector/Subsense
                    ${PROJECT_SOURCE_DIR}/../src/Tracker
                    ${PROJECT_SOURCE_DIR}/../src/Tracker/HungarianAlg
                    ${PROJECT_SOURCE_DIR}/../thirdparty
                    ${PROJECT_SOURCE_DIR}/../thirdparty/spdlog/include)

set(LIBS ${OpenCV_LIBS}
         mtracking
         mdetection
         inih)

if (BUILD_ONNX_TENSORRT)
    add_definitions(-DBUILD_ONNX_TENSORRT)
endif(BUILD_ONNX_TENSORRT)

if (BUILD_CARS_COUNTING)
    add_definitions(-DBUILD_CARS_COUNTING)
endif(BUILD_CARS_COUNTING)

if (USE_CLIP)
    add_definitions(-DUSE_CLIP)
    set(LIBS ${LIBS} ruclip)
endif(USE_CLIP)

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES} ${HEADERS})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "apps")