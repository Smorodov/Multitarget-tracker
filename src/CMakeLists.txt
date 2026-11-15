# Shared type registry (object_types.cpp). Both mtracking and mdetection use
# TypeConverter, so the registry must live in exactly one library: type IDs
# have to be identical across the tracker and the detectors (roadmap H22/M21).
add_library(mtracking_core ${MT_LIBRARY_TYPE} mtracking/object_types.cpp)
target_include_directories(mtracking_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
if (MT_SHARED)
    # object_types.h uses MT_CORE_API to dllexport/dllimport the registry data.
    target_compile_definitions(mtracking_core PUBLIC MT_CORE_SHARED)
    # The registry symbols are exported explicitly above, so the generated
    # export-all .def is not needed (and would duplicate them).
    set_target_properties(mtracking_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
endif()
install(TARGETS mtracking_core
        EXPORT MTTrackingExports
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)

add_subdirectory(Detector)
add_subdirectory(Tracker)

if(MTRACKER_PYTHON)
    file(GLOB_RECURSE mtracker_python_src python_bind/*.cpp)
    file(GLOB_RECURSE mtracker_python_inc python_bind/*.h)

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/mtracking
        ${CMAKE_CURRENT_SOURCE_DIR}/Tracker
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector
        ${CMAKE_CURRENT_SOURCE_DIR}/Detector/vibe_src
        ${CMAKE_SOURCE_DIR}/thirdparty
        ${CMAKE_SOURCE_DIR}/thirdparty/pybind11/include)
    if (NUMPY_INCLUDE_DIR)
        include_directories(${NUMPY_INCLUDE_DIR})
    endif()

    pybind11_add_module(pymtracking ${mtracker_python_src} ${mtracker_python_inc})
    target_link_libraries(pymtracking PRIVATE mtracking mdetection ${OpenCV_LIBS} pybind11::module Threads::Threads)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(pymtracking PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # set_target_properties(pymtracking PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

    # target_compile_definitions(pymtracking PRIVATE BGS_PYTHON_SUPPORT=1)

    set_property(TARGET pymtracking PROPERTY OUTPUT_NAME "pymtracking")
endif()
