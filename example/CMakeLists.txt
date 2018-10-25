cmake_minimum_required (VERSION 3.5)

project(MultitargetTracker)

set(SOURCES
             main.cpp
             VideoExample.cpp
)

set(HEADERS
             MouseExample.h
             VideoExample.h
)

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# ----------------------------------------------------------------------------
# добавляем include директории
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(
                    ${PROJECT_SOURCE_DIR}/../src
                    ${PROJECT_SOURCE_DIR}/../src/Detector
                    ${PROJECT_SOURCE_DIR}/../src/Detector/vibe_src
                    ${PROJECT_SOURCE_DIR}/../src/Detector/Subsense
                    ${PROJECT_SOURCE_DIR}/../src/Tracker
                    ${PROJECT_SOURCE_DIR}/../src/Tracker/HungarianAlg
)

set(LIBS
    ${OpenCV_LIBS}
    mtracking
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LIBS})
