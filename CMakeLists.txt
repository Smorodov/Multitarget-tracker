# ----------------------------------------------------------------------------
# минимальная версия CMake, способная переварить этот файл
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
project(MultitargetTracker)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors --std=gnu++14" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native -mtune=native --fast-math -ffast-math -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4" CACHE STRING COMPILE_FLAGS FORCE)
endif()


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------------- 
# Предполагаем, что FindOpenCV.cmake расположен по адресу CMAKE_MODULE_PATH.
# ----------------------------------------------------------------------------
  FIND_PACKAGE(OpenCV)
# ----------------------------------------------------------------------------  
# ищем все .cpp и .h файлы и добавляем их в наш проект 
# ----------------------------------------------------------------------------
  set(folder_source  main.cpp
             Detector/Detector.cpp
             HungarianAlg/HungarianAlg.cpp
             vibe_src/BackgroundSubtract.cpp
             vibe_src/vibe.cpp
             Tracker/Ctracker.cpp
             Tracker/LocalTracker.cpp
             Tracker/Kalman.cpp)

  set(folder_header  Detector/Detector.h
             HungarianAlg/HungarianAlg.h
             vibe_src/BackgroundSubtract.h
             vibe_src/vibe.hpp
             Tracker/track.h
             Tracker/Ctracker.h
             Tracker/LocalTracker.h
             Tracker/Kalman.h
             Tracker/defines.h)

  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})
# ----------------------------------------------------------------------------  
# создаем проект
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(MultitargetTracker ${folder_source} ${folder_header})
# ----------------------------------------------------------------------------   
# добавляем include директории
# ----------------------------------------------------------------------------
  INCLUDE_DIRECTORIES(
		      ${PROJECT_SOURCE_DIR}/Detector
		      ${PROJECT_SOURCE_DIR}/HungarianAlg
              ${PROJECT_SOURCE_DIR}/Tracker
		      ${PROJECT_SOURCE_DIR}/vibe_src)

  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
# ----------------------------------------------------------------------------
# и Lib-ы opencv 
# ----------------------------------------------------------------------------
  TARGET_LINK_LIBRARIES(MultitargetTracker ${OpenCV_LIBS})
