cmake_minimum_required (VERSION 3.16)

project(mot_eval)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../src
                    ${PROJECT_SOURCE_DIR}/../src/mtracking
                    ${PROJECT_SOURCE_DIR}/../src/Tracker
                    ${PROJECT_SOURCE_DIR}/../src/Tracker/HungarianAlg
                    ${PROJECT_SOURCE_DIR}/../thirdparty
                    ${PROJECT_SOURCE_DIR}/../thirdparty/spdlog/include)

set(LIBS ${OpenCV_LIBS}
         mtracking
         Threads::Threads)

if (OpenMP_CXX_FOUND)
    set(LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif()

ADD_EXECUTABLE(${PROJECT_NAME}
               mot_eval/mot_metrics.cpp
               mot_eval/main.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${LIBS})
