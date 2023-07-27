cmake_minimum_required(VERSION 3.9)

project(MTTracking VERSION 1.1.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

unset(CMAKE_C_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS_RELEASE CACHE)
# unset(CMAKE_CXX_FLAGS_DEBUG CACHE)

find_package(OpenMP)
if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors -fPIC" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native -mtune=native -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /Ox /Ob2 /Oi /Ot /arch:AVX2 /fp:fast /DNDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /Od /Ob0 /DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)

    add_definitions(-D_USE_MATH_DEFINES -DNOMINMAX)
endif()

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

FIND_PACKAGE(OpenCV REQUIRED)

option(SILENT_WORK "Remove all imshow and waitKey functions?" OFF)
if (SILENT_WORK)
    add_definitions(-DSILENT_WORK)
endif(SILENT_WORK)

include(CheckIncludeFileCXX)
check_include_file_cxx(filesystem HAVE_FILESYSTEM)
if(HAVE_FILESYSTEM)
    add_definitions(-DHAVE_FILESYSTEM)
    message("Founded filesystem header")
else(HAVE_FILESYSTEM)
    message("Do not found filesystem header")
endif(HAVE_FILESYSTEM)

option(BUILD_ABANDONED_DETECTOR "Should compiled abandoned detector example?" OFF)
if (BUILD_ABANDONED_DETECTOR)
    add_definitions(-DBUILD_ABANDONED_DETECTOR)
    add_subdirectory(combined)
endif(BUILD_ABANDONED_DETECTOR)

option(BUILD_EXAMPLES "Should compiled examples (motion detection, pedestrians, faces, DNNs etc)?" ON)
if (BUILD_EXAMPLES)
    add_subdirectory(example)
endif(BUILD_EXAMPLES)

option(BUILD_CARS_COUNTING "Should compiled Cars counting example?" OFF)
if (BUILD_CARS_COUNTING)
    add_definitions(-DBUILD_CARS_COUNTING)
endif(BUILD_CARS_COUNTING)

option(BUILD_ASYNC_DETECTOR "Should compiled async example with low fps Detector?" OFF)
if (BUILD_ASYNC_DETECTOR)
    add_subdirectory(async_detector)
endif(BUILD_ASYNC_DETECTOR)

option(BUILD_YOLO_LIB "Should compiled standalone yolo_lib with original darknet?" OFF)
if (BUILD_YOLO_LIB)
    add_subdirectory(src/Detector/darknet)
    add_definitions(-DBUILD_YOLO_LIB)

if (MSVC)
    if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
        set(BIT_SYSTEM x32)
    else()
        set(BIT_SYSTEM x64)
    endif()

    set(LIB_PTHREAD pthreadVC2)
else()
    set(LIB_PTHREAD pthread)
endif()

if (MSVC)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Detector/darknet/3rdparty/dll/${BIT_SYSTEM}/pthreadVC2.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/Detector/darknet/3rdparty/dll/${BIT_SYSTEM}/pthreadVC2.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)

    file(GLOB CUDNN_DLL ${CUDNN_DIR}/bin/*.dll)
    file(COPY ${CUDNN_DLL} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    file(COPY ${CUDNN_DLL} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
endif()

endif(BUILD_YOLO_LIB)

option(BUILD_YOLO_TENSORRT "Should compiled TensorRT binding for YOLO?" OFF)
if (BUILD_YOLO_TENSORRT)
    add_subdirectory(src/Detector/tensorrt_yolo)
    add_definitions(-DBUILD_YOLO_TENSORRT)
endif(BUILD_YOLO_TENSORRT)

option(MTRACKER_PYTHON "Build mtracking Python bindings?" OFF)
if(MTRACKER_PYTHON)
    set(NUMPY_INCLUDE_DIR "" CACHE FILEPATH "Path to numpy header if cmake can't find them.")
    if (NOT ${NUMPY_INCLUDE_DIR} STREQUAL "")
      message( " *** NUMPY_INCLUDE_DIR : ${NUMPY_INCLUDE_DIR}" )
      if(NOT EXISTS ${NUMPY_INCLUDE_DIR}/numpy/ndarrayobject.h)
          message(SEND_ERROR "Can't find numpy/ndarrayobject.h in ${NUMPY_INCLUDE_DIR}")
      endif()
    include_directories(${NUMPY_INCLUDE_DIR})
endif()

    set(PYBIND11_LTO_CXX_FLAGS "")
    set(PYBIND11_PYTHON_VERSION 3)
    add_subdirectory(thirdparty/pybind11)
endif(MTRACKER_PYTHON)


add_subdirectory(thirdparty)
add_subdirectory(src)


# Create CMake config files for distribution
set(INCLUDE_INSTALL_DIR include/ )
set(LIB_INSTALL_DIR lib/ )

install(EXPORT MTTrackingExports
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${LIB_INSTALL_DIR}/${PROJECT_NAME}/cmake
)

include(CMakePackageConfigHelpers)

set(CONFIG_FILENAME ${PROJECT_NAME}Config.cmake)

configure_package_config_file(${CONFIG_FILENAME}.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILENAME}
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/${PROJECT_NAME}/cmake
    PATH_VARS INCLUDE_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILENAME}
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${LIB_INSTALL_DIR}/${PROJECT_NAME}/cmake )
