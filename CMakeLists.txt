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

  set(graph_source
             graph/tokenise.cpp
#             graph/script.cpp
#             graph/rings.cpp
             graph/mytree.cpp
             graph/mygraph.cpp
             graph/mwbmatching.cpp
             graph/mincut.cpp
             graph/gport.cpp
#             graph/gml2nestedsql.cpp
#             graph/gml2dot.cpp
#             graph/components.cpp
             graph/fheap.c
             
             graph/GTL/src/bellman_ford.cpp
             graph/GTL/src/bfs.cpp
             graph/GTL/src/biconnectivity.cpp
             graph/GTL/src/bid_dijkstra.cpp
             graph/GTL/src/components.cpp
             graph/GTL/src/debug.cpp
             graph/GTL/src/dfs.cpp
             graph/GTL/src/dijkstra.cpp
             graph/GTL/src/edge.cpp
             graph/GTL/src/embedding.cpp
             graph/GTL/src/fm_partition.cpp
             graph/GTL/src/gml_parser.cpp
             graph/GTL/src/gml_scanner.cpp
             graph/GTL/src/graph.cpp
             graph/GTL/src/maxflow_ff.cpp
             graph/GTL/src/maxflow_pp.cpp
             graph/GTL/src/maxflow_sap.cpp
             graph/GTL/src/min_tree.cpp
             graph/GTL/src/node.cpp
             graph/GTL/src/planarity.cpp
             graph/GTL/src/pq_node.cpp
             graph/GTL/src/pq_tree.cpp
             graph/GTL/src/ratio_cut_partition.cpp
             graph/GTL/src/st_number.cpp
             graph/GTL/src/topsort.cpp
             )

  set(graph_header
             graph/tokenise.h
             graph/mytree.h
             graph/mygraph.h
             graph/mwbmatching.h
             graph/mincut.h
             graph/gport.h
             graph/gdefs.h
             graph/fheap.h
             
             graph/GTL/include/GTL/algorithm.h
             graph/GTL/include/GTL/bellman_ford.h
             graph/GTL/include/GTL/bfs.h
             graph/GTL/include/GTL/biconnectivity.h
             graph/GTL/include/GTL/bid_dijkstra.h
             graph/GTL/include/GTL/bin_heap.h
             graph/GTL/include/GTL/components.h
             graph/GTL/include/GTL/debug.h
             graph/GTL/include/GTL/dfs.h
             graph/GTL/include/GTL/dijkstra.h
             graph/GTL/include/GTL/edge.h
             graph/GTL/include/GTL/edge_data.h
             graph/GTL/include/GTL/edge_map.h
             graph/GTL/include/GTL/embedding.h
             graph/GTL/include/GTL/fm_partition.h
             graph/GTL/include/GTL/gml_parser.h
             graph/GTL/include/GTL/gml_scanner.h
             graph/GTL/include/GTL/graph.h
             graph/GTL/include/GTL/maxflow_ff.h
             graph/GTL/include/GTL/maxflow_pp.h
             graph/GTL/include/GTL/maxflow_sap.h
             graph/GTL/include/GTL/min_tree.h
             graph/GTL/include/GTL/ne_map.h
             graph/GTL/include/GTL/node.h
             graph/GTL/include/GTL/node_data.h
             graph/GTL/include/GTL/node_map.h
             graph/GTL/include/GTL/planarity.h
             graph/GTL/include/GTL/pq_node.h
             graph/GTL/include/GTL/pq_tree.h
             graph/GTL/include/GTL/ratio_cut_partition.h
             graph/GTL/include/GTL/st_number.h
             graph/GTL/include/GTL/symlist.h
             graph/GTL/include/GTL/topsort.h
             graph/GTL/include/GTL/version.h
             graph/GTL/include/GTL/GTL.h
             )

  SOURCE_GROUP("Source Files" FILES ${folder_source})
  SOURCE_GROUP("Header Files" FILES ${folder_header})

  SOURCE_GROUP("graph" FILES ${graph_source} ${graph_source})

option(USE_OCV_BGFG "Should the bgfg algorithms from opencv_contrib?" ON)
if(USE_OCV_BGFG)
    add_definitions(-DUSE_OCV_BGFG)
endif(USE_OCV_BGFG)

# ----------------------------------------------------------------------------  
# создаем проект
# ----------------------------------------------------------------------------
  ADD_EXECUTABLE(MultitargetTracker ${folder_source} ${folder_header} ${graph_source} ${graph_source})
# ----------------------------------------------------------------------------   
# добавляем include директории
# ----------------------------------------------------------------------------
  INCLUDE_DIRECTORIES(
		      ${PROJECT_SOURCE_DIR}/Detector
		      ${PROJECT_SOURCE_DIR}/HungarianAlg
              ${PROJECT_SOURCE_DIR}/Tracker
		      ${PROJECT_SOURCE_DIR}/vibe_src
		      ${PROJECT_SOURCE_DIR}/graph
		      ${PROJECT_SOURCE_DIR}/graph/GTL/include)

  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
# ----------------------------------------------------------------------------
# и Lib-ы opencv 
# ----------------------------------------------------------------------------
  TARGET_LINK_LIBRARIES(MultitargetTracker ${OpenCV_LIBS})
