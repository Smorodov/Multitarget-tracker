cmake_minimum_required (VERSION 2.8)

project(CarsCounting)

set(SOURCES
             main.cpp
             CarsCounting.cpp
)

set(HEADERS
             CarsCounting.h
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
