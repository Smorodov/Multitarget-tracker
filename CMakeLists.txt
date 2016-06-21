# ----------------------------------------------------------------------------
# минимальна€ верси€ CMake, способна€ переварить этот файл
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
project(MultitargetTracker)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------------- 
# ѕредполагаем, что FindOpenCV.cmake расположен по адресу CMAKE_MODULE_PATH.
# ----------------------------------------------------------------------------
  FIND_PACKAGE(OpenCV)
# ----------------------------------------------------------------------------  
# ищем все .cpp и .h файлы и добавл€ем их в наш проект 
# ----------------------------------------------------------------------------
  set(folder_source  main.cpp
		     Detector/Detector.cpp
             HungarianAlg/HungarianAlg.cpp
             vibe_src/BackgroundSubtract.cpp
             vibe_src/VIBE.cpp
             Tracker/Ctracker.cpp
             Tracker/Kalman.cpp)

  set(folder_header  Detector/Detector.h
		     HungarianAlg/HungarianAlg.h
             vibe_src/BackgroundSubtract.h
             vibe_src/VIBE.h
             Tracker/Ctracker.h
             Tracker/Kalman.h
             Tracker/defines.h)

  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})
# ----------------------------------------------------------------------------  
# создаем проект
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(MultitargetTracker ${folder_source} ${folder_header})
# ----------------------------------------------------------------------------   
# добавл€ем include директории
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
