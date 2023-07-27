cmake_minimum_required(VERSION 3.9)

project(mtracking)

set(main_sources
             ../common/nms.h
             ../common/defines.h
             ../common/object_types.h
             ../common/object_types.cpp)

set(tracker_sources
             Ctracker.cpp
             ShortPathCalculator.cpp
             track.cpp
             Kalman.cpp
             TrackerSettings.cpp

             HungarianAlg/HungarianAlg.cpp

             VOTTracker.hpp
             dat/dat_tracker.cpp
             )
set(tracker_headers
             Ctracker.h
             ShortPathCalculator.h
             track.h
             trajectory.h
             Kalman.h
             TrackerSettings.h
             HungarianAlg/HungarianAlg.h
             EmbeddingsCalculator.hpp
             dat/dat_tracker.hpp
)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm|ARM|aarch64|AARCH64")

else()
    set(tracker_sources ${tracker_sources}
             staple/fhog.cpp
             staple/staple_tracker.cpp

             ldes/correlation.cpp
             ldes/fft_functions.cpp
             ldes/fhog.cpp
             ldes/hann.cpp
             ldes/ldes_tracker.cpp
             ldes/ldes_tracker.h
             )
    set(tracker_headers ${tracker_headers}
             staple/fhog.h
             staple/sse.hpp
             staple/staple_tracker.hpp
             ldes/correlation.h
             ldes/fft_functions.h
             ldes/fhog.hpp
             ldes/hann.h
    )
    add_definitions(-DUSE_STAPLE_TRACKER)
endif()

  set(graph_source
             graph/tokenise.cpp
#            graph/script.cpp
#            graph/rings.cpp
             graph/mytree.cpp
             graph/mygraph.cpp
             graph/mwbmatching.cpp
             graph/mincut.cpp
             graph/gport.cpp
#            graph/gml2nestedsql.cpp
#            graph/gml2dot.cpp
#            graph/components.cpp
             graph/fheap.c
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
             )

  set(gtl_source
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

  set(gtl_header
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

  SOURCE_GROUP("Src" FILES ${main_sources})
  SOURCE_GROUP("Tracker" FILES ${tracker_sources} ${tracker_headers})

  SOURCE_GROUP("Tracker/graph" FILES ${graph_source} ${graph_header})
  SOURCE_GROUP("Tracker/GTL" FILES ${gtl_source} ${gtl_header})

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
include_directories(${PROJECT_SOURCE_DIR}/../common)
include_directories(${PROJECT_SOURCE_DIR}/graph)
include_directories(${PROJECT_SOURCE_DIR}/graph/GTL/include)
include_directories(${PROJECT_SOURCE_DIR}/../../thirdparty)

if (CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME} SHARED
        ${main_sources}
	${tracker_sources}
	${graph_source} ${graph_header}
	${gtl_source} ${gtl_header})
    set(LIBS
        ${OpenCV_LIBS}
        inih
        #iconv
    )
else(CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME}
        ${main_sources}
        ${tracker_sources}
        ${graph_source} ${graph_header}
        ${gtl_source} ${gtl_header})
    set(LIBS
        # ${OpenCV_LIBS}
        inih
    )
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${tracker_headers};../common/defines.h;../common/object_types.h")
install(TARGETS ${PROJECT_NAME}
        EXPORT MTTrackingExports
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")