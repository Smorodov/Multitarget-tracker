cmake_minimum_required (VERSION 3.5)

project(AsyncDetector)

set(SOURCES main.cpp
            AsyncDetector.cpp)

set(HEADERS AsyncDetector.h
            Queue.h)

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
                    ${PROJECT_SOURCE_DIR}/../thirdparty/spdlog/include)

set(LIBS ${OpenCV_LIBS}
         mtracking
         mdetection)
                     
ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES} ${HEADERS})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "apps")