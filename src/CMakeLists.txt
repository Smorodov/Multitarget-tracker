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
    target_link_libraries(pymtracking PRIVATE mtracking mdetection ${OpenCV_LIBS} pybind11::module)

    # set_target_properties(pymtracking PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

    # target_compile_definitions(pymtracking PRIVATE BGS_PYTHON_SUPPORT=1)

    set_property(TARGET pymtracking PROPERTY OUTPUT_NAME "pymtracking")
endif()
