cmake_minimum_required(VERSION 3.5)

project(mtracking)

set(main_sources nms.h defines.h)

  set(detector_sources
             Detector/BaseDetector.cpp
             Detector/MotionDetector.cpp
             Detector/BackgroundSubtract.cpp
             Detector/vibe_src/vibe.cpp
             Detector/Subsense/BackgroundSubtractorLBSP.cpp
             Detector/Subsense/BackgroundSubtractorLOBSTER.cpp
             Detector/Subsense/BackgroundSubtractorSuBSENSE.cpp
             Detector/Subsense/LBSP.cpp
             Detector/FaceDetector.cpp
             Detector/PedestrianDetector.cpp
             Detector/pedestrians/c4-pedestrian-detector.cpp
             Detector/SSDMobileNetDetector.cpp
             Detector/YoloDetector.cpp

             Detector/BaseDetector.h
             Detector/MotionDetector.h
             Detector/BackgroundSubtract.h
             Detector/vibe_src/vibe.hpp
             Detector/Subsense/BackgroundSubtractorLBSP.h
             Detector/Subsense/BackgroundSubtractorLOBSTER.h
             Detector/Subsense/BackgroundSubtractorSuBSENSE.h
             Detector/Subsense/DistanceUtils.h
             Detector/Subsense/LBSP.h
             Detector/Subsense/RandUtils.h
             Detector/FaceDetector.h
             Detector/PedestrianDetector.h
             Detector/pedestrians/c4-pedestrian-detector.h
             Detector/SSDMobileNetDetector.h
             Detector/YoloDetector.h
)

if (BUILD_YOLO_LIB)
    set(detector_sources ${detector_sources} Detector/YoloDarknetDetector.cpp Detector/YoloDarknetDetector.h)
endif()

  set(tracker_sources
             Tracker/Ctracker.cpp
             Tracker/track.cpp
             Tracker/HungarianAlg/HungarianAlg.cpp
             Tracker/LocalTracker.cpp
             Tracker/Kalman.cpp

             Tracker/Ctracker.h
             Tracker/track.h
             Tracker/HungarianAlg/HungarianAlg.h
             Tracker/LocalTracker.h
             Tracker/Kalman.h

             Tracker/VOTTracker.hpp
             Tracker/dat/dat_tracker.cpp
             Tracker/dat/dat_tracker.hpp
)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm|ARM|aarch64|AARCH64") 

else()
    set(tracker_sources ${tracker_sources}
             Tracker/staple/fhog.cpp
             Tracker/staple/fhog.h
             Tracker/staple/sse.hpp
             Tracker/staple/staple_tracker.cpp
             Tracker/staple/staple_tracker.hpp
    )
    add_definitions(-DUSE_STAPLE_TRACKER)
endif()

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

  SOURCE_GROUP("Src" FILES ${main_sources})
  SOURCE_GROUP("Detector" FILES ${detector_sources})
  SOURCE_GROUP("Tracker" FILES ${tracker_sources})

  SOURCE_GROUP("Tracker/graph" FILES ${graph_source} ${graph_header})
  SOURCE_GROUP("Tracker/GTL" FILES ${gtl_source} ${gtl_header})

include(CheckIncludeFileCXX)
check_include_file_cxx(opencv2/bgsegm.hpp HAVE_OPENCV_CONTRIB)
if(HAVE_OPENCV_CONTRIB)
    add_definitions(-DHAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" ON)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" ON)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" ON)

else(HAVE_OPENCV_CONTRIB)

    option(USE_OCV_BGFG "Should use the bgfg algorithms from opencv_contrib?" OFF)
    option(USE_OCV_KCF "Should use the KCF tracker from opencv_contrib for collisions resolving?" OFF)
    option(USE_OCV_UKF "Should use the Unscented Kalman Filterr from opencv_contrib?" OFF)
endif(HAVE_OPENCV_CONTRIB)

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

if(USE_OCV_BGFG)
    add_definitions(-DUSE_OCV_BGFG)
else()
    remove_definitions(-DUSE_OCV_BGFG)
endif(USE_OCV_BGFG)


INCLUDE_DIRECTORIES(
                    ${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/../src
                    ${PROJECT_SOURCE_DIR}/Tracker
                    ${PROJECT_SOURCE_DIR}/Tracker/graph
                    ${PROJECT_SOURCE_DIR}/Tracker/graph/GTL/include
)

if (CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME} SHARED 
        ${main_sources}
	${detector_sources}
	${tracker_sources}
	${graph_source} ${graph_header}
	${gtl_source} ${gtl_header})
else(CMAKE_COMPILER_IS_GNUCXX)
    add_library(${PROJECT_NAME}
        ${main_sources}
	${detector_sources}
	${tracker_sources}
	${graph_source} ${graph_header}
	${gtl_source} ${gtl_header})
endif()

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

if (BUILD_YOLO_LIB)
    set(LIBS ${LIBS} yolo_lib)
endif(BUILD_YOLO_LIB)

target_link_libraries(${PROJECT_NAME} ${LIBS})

