# ----------------------------------------------------------------------------
# минимальная версия CMake, способная переварить этот файл
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
project(MultitargetTracker)

unset(CMAKE_CXX_FLAGS CACHE)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors --std=gnu++14" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native -mtune=native --fast-math -ffast-math -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
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
             Detector/BackgroundSubtract.cpp
             Detector/vibe_src/vibe.cpp
             Tracker/Ctracker.cpp
             Tracker/HungarianAlg/HungarianAlg.cpp
             Tracker/LocalTracker.cpp
             Tracker/Kalman.cpp)

  set(folder_header
             Detector/Detector.h
             Detector/BackgroundSubtract.h
             Detector/vibe_src/vibe.hpp
             pedestrians/c4-pedestrian-detector.h
             Tracker/Ctracker.h
             Tracker/track.h
             Tracker/HungarianAlg/HungarianAlg.h
             Tracker/LocalTracker.h
             Tracker/Kalman.h
             Tracker/defines.h)

  set(graph_source
             Tracker/graph/tokenise.cpp
#             Tracker/graph/script.cpp
#             Tracker/graph/rings.cpp
             Tracker/graph/mytree.cpp
             Tracker/graph/mygraph.cpp
             Tracker/graph/mwbmatching.cpp
             Tracker/graph/mincut.cpp
             Tracker/graph/gport.cpp
#             Tracker/graph/gml2nestedsql.cpp
#             Tracker/graph/gml2dot.cpp
#             Tracker/graph/components.cpp
             Tracker/graph/fheap.c
             )

  set(graph_header
             Tracker/graph/tokenise.h
             Tracker/graph/mytree.h
             Tracker/graph/mygraph.h
             Tracker/graph/mwbmatching.h
             Tracker/graph/mincut.h
             Tracker/graph/gport.h
             Tracker/graph/gdefs.h
             Tracker/graph/fheap.h
             )

  set(gtl_source
             Tracker/graph/GTL/src/bellman_ford.cpp
             Tracker/graph/GTL/src/bfs.cpp
             Tracker/graph/GTL/src/biconnectivity.cpp
             Tracker/graph/GTL/src/bid_dijkstra.cpp
             Tracker/graph/GTL/src/components.cpp
             Tracker/graph/GTL/src/debug.cpp
             Tracker/graph/GTL/src/dfs.cpp
             Tracker/graph/GTL/src/dijkstra.cpp
             Tracker/graph/GTL/src/edge.cpp
             Tracker/graph/GTL/src/embedding.cpp
             Tracker/graph/GTL/src/fm_partition.cpp
             Tracker/graph/GTL/src/gml_parser.cpp
             Tracker/graph/GTL/src/gml_scanner.cpp
             Tracker/graph/GTL/src/graph.cpp
             Tracker/graph/GTL/src/maxflow_ff.cpp
             Tracker/graph/GTL/src/maxflow_pp.cpp
             Tracker/graph/GTL/src/maxflow_sap.cpp
             Tracker/graph/GTL/src/min_tree.cpp
             Tracker/graph/GTL/src/node.cpp
             Tracker/graph/GTL/src/planarity.cpp
             Tracker/graph/GTL/src/pq_node.cpp
             Tracker/graph/GTL/src/pq_tree.cpp
             Tracker/graph/GTL/src/ratio_cut_partition.cpp
             Tracker/graph/GTL/src/st_number.cpp
             Tracker/graph/GTL/src/topsort.cpp
             )

  set(gtl_header
             Tracker/graph/GTL/include/GTL/algorithm.h
             Tracker/graph/GTL/include/GTL/bellman_ford.h
             Tracker/graph/GTL/include/GTL/bfs.h
             Tracker/graph/GTL/include/GTL/biconnectivity.h
             Tracker/graph/GTL/include/GTL/bid_dijkstra.h
             Tracker/graph/GTL/include/GTL/bin_heap.h
             Tracker/graph/GTL/include/GTL/components.h
             Tracker/graph/GTL/include/GTL/debug.h
             Tracker/graph/GTL/include/GTL/dfs.h
             Tracker/graph/GTL/include/GTL/dijkstra.h
             Tracker/graph/GTL/include/GTL/edge.h
             Tracker/graph/GTL/include/GTL/edge_data.h
             Tracker/graph/GTL/include/GTL/edge_map.h
             Tracker/graph/GTL/include/GTL/embedding.h
             Tracker/graph/GTL/include/GTL/fm_partition.h
             Tracker/graph/GTL/include/GTL/gml_parser.h
             Tracker/graph/GTL/include/GTL/gml_scanner.h
             Tracker/graph/GTL/include/GTL/graph.h
             Tracker/graph/GTL/include/GTL/maxflow_ff.h
             Tracker/graph/GTL/include/GTL/maxflow_pp.h
             Tracker/graph/GTL/include/GTL/maxflow_sap.h
             Tracker/graph/GTL/include/GTL/min_tree.h
             Tracker/graph/GTL/include/GTL/ne_map.h
             Tracker/graph/GTL/include/GTL/node.h
             Tracker/graph/GTL/include/GTL/node_data.h
             Tracker/graph/GTL/include/GTL/node_map.h
             Tracker/graph/GTL/include/GTL/planarity.h
             Tracker/graph/GTL/include/GTL/pq_node.h
             Tracker/graph/GTL/include/GTL/pq_tree.h
             Tracker/graph/GTL/include/GTL/ratio_cut_partition.h
             Tracker/graph/GTL/include/GTL/st_number.h
             Tracker/graph/GTL/include/GTL/symlist.h
             Tracker/graph/GTL/include/GTL/topsort.h
             Tracker/graph/GTL/include/GTL/version.h
             Tracker/graph/GTL/include/GTL/GTL.h
             )

  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})

  SOURCE_GROUP("graph" FILES ${graph_source} ${graph_header})
  SOURCE_GROUP("GTL" FILES ${gtl_source} ${gtl_header})

option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" ON)
if(USE_OCV_BGFG)
    add_definitions(-DUSE_OCV_BGFG)
endif(USE_OCV_BGFG)

option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" ON)
if(USE_OCV_KCF)
    add_definitions(-DUSE_OCV_KCF)
endif(USE_OCV_KCF)

option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" ON)
if(USE_OCV_UKF)
    add_definitions(-DUSE_OCV_UKF)
endif(USE_OCV_UKF)

# ----------------------------------------------------------------------------  
# создаем проект
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(MultitargetTracker ${folder_source} ${folder_header} ${graph_source} ${graph_header} ${gtl_source} ${gtl_header})
# ----------------------------------------------------------------------------   
# добавляем include директории
# ----------------------------------------------------------------------------
  INCLUDE_DIRECTORIES(
		      ${PROJECT_SOURCE_DIR}/Detector
                      ${PROJECT_SOURCE_DIR}/vibe_src
                      ${PROJECT_SOURCE_DIR}/Tracker
                      ${PROJECT_SOURCE_DIR}/Tracker/HungarianAlg
                      ${PROJECT_SOURCE_DIR}/Tracker/graph
                      ${PROJECT_SOURCE_DIR}/Tracker/graph/GTL/include)

  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
# ----------------------------------------------------------------------------
# и Lib-ы opencv 
# ----------------------------------------------------------------------------


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
TARGET_LINK_LIBRARIES(MultitargetTracker ${LIBS})
