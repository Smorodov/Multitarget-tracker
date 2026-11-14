cmake_minimum_required(VERSION 3.16)

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 /utf-8" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /Ox /Ob2 /Oi /Ot /arch:AVX2 /fp:fast /DNDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /Od /Ob0 /DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)

    add_definitions(-D_USE_MATH_DEFINES -DNOMINMAX)
endif()

# Keep the outputs of the conventional in-repo build tree (cmake -B build)
# in <repo>/build. Do NOT re-set CMAKE_BINARY_DIR for other (out-of-source)
# build trees: manually re-pointing it makes every build write into the same
# directory and mix artifacts configured with different options (e.g. the
# SILENT_WORK=ON python-wheel build vs the normal tree) - roadmap H16.
if (CMAKE_BINARY_DIR STREQUAL "${CMAKE_SOURCE_DIR}/build")
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
    set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
endif()

FIND_PACKAGE(OpenCV REQUIRED)

option(SILENT_WORK "Remove all imshow and waitKey functions?" OFF)
if (SILENT_WORK)
    add_definitions(-DSILENT_WORK)
endif(SILENT_WORK)

include(CheckIncludeFileCXX)
check_include_file_cxx(filesystem HAVE_FILESYSTEM)
if(HAVE_FILESYSTEM)
    message("Founded filesystem header")
else(HAVE_FILESYSTEM)
    add_definitions(-DHAVE_EXPERIMENTAL_FILESYSTEM)
    message("Do not found filesystem header")
endif(HAVE_FILESYSTEM)

option(BUILD_EXAMPLES "Should compiled examples (motion detection, pedestrians, faces, DNNs etc)?" ON)
option(BUILD_CARS_COUNTING "Should compiled Cars counting example?" OFF)
option(USE_GEOGRAPHICLIB "Should be used GeographicLib for local cartesian (ENU) geo calculations?" OFF)

if (USE_GEOGRAPHICLIB)
    find_package(GeographicLib CONFIG REQUIRED)
    add_definitions(-DUSE_GEOGRAPHICLIB)
endif(USE_GEOGRAPHICLIB)

if (BUILD_EXAMPLES)
    add_subdirectory(example)
endif(BUILD_EXAMPLES)


option(USE_CLIP "Should be used RuCLIP|CLIP for objects classification?" OFF)
if (USE_CLIP)
    add_definitions(-DUSE_CLIP)
endif(USE_CLIP)

option(BUILD_ASYNC_DETECTOR "Should compiled async example with low fps Detector?" OFF)
if (BUILD_ASYNC_DETECTOR)
    add_subdirectory(async_detector)
endif(BUILD_ASYNC_DETECTOR)

option(BUILD_ONNX_TENSORRT "Should compiled TensorRT binding for ONNX models?" OFF)
if (BUILD_ONNX_TENSORRT)
    add_subdirectory(src/Detector/tensorrt_onnx)
    add_definitions(-DBUILD_ONNX_TENSORRT)
endif(BUILD_ONNX_TENSORRT)

option(MTRACKER_PYTHON "Build mtracking Python bindings?" OFF)
if(MTRACKER_PYTHON)
    set(NUMPY_INCLUDE_DIR "" CACHE FILEPATH "Path to numpy header if cmake can't find them.")
    if (NUMPY_INCLUDE_DIR)
      message( " *** NUMPY_INCLUDE_DIR : ${NUMPY_INCLUDE_DIR}" )
      if(NOT EXISTS "${NUMPY_INCLUDE_DIR}/numpy/ndarrayobject.h")
          message(SEND_ERROR "Can't find numpy/ndarrayobject.h in ${NUMPY_INCLUDE_DIR}")
      endif()
      include_directories(${NUMPY_INCLUDE_DIR})
    endif()

    set(PYBIND11_LTO_CXX_FLAGS "")
    set(PYBIND11_PYTHON_VERSION 3)

    # The vendored pybind11 2.5 still declares compatibility with CMake < 3.5,
    # which CMake 4 removed. Allow configuring the old pybind11 scripts.
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
    endif()

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
