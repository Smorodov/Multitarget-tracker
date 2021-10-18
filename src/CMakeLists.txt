add_subdirectory(Detector)
add_subdirectory(Tracker)

if(MTRACKER_PYTHON)
    file(GLOB_RECURSE mtracker_python_src python_bind/*.cpp)
    file(GLOB_RECURSE mtracker_python_inc python_bind/*.h)

    include_directories(${CMAKE_SOURCE_DIR}/thirdparty/pybind11/include)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${NUMPY_INCLUDE_DIR})

if (MSVC)
    if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
        set(BIT_SYSTEM x32)
    else()
        set(BIT_SYSTEM x64)
    endif()

    include_directories(Detector/darknet/3rdparty/include)
    link_directories(Detector/darknet/3rdparty/lib/${BIT_SYSTEM})
    set(LIB_PTHREAD pthreadVC2)
else()
    set(LIB_PTHREAD pthread)
endif()

    include_directories(common)

    pybind11_add_module(pymtracking ${mtracker_python_src} ${mtracker_python_inc})
    target_link_libraries(pymtracking PRIVATE mtracking mdetection ${OpenCV_LIBS} ${PYTHON_LIBRARY} pybind11::module)

    # set_target_properties(pymtracking PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

    # target_compile_definitions(pymtracking PRIVATE BGS_PYTHON_SUPPORT=1)

    set_property(TARGET pymtracking PROPERTY OUTPUT_NAME "pymtracking")
endif()
