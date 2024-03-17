cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ruclip)

find_package(Torch REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/youtokentome)
include_directories(${CMAKE_SOURCE_DIR}/youtokentome/third_party)

set(RUCLIP_SOURCE_FILES RuCLIP.cpp
	RuCLIPProcessor.cpp
	youtokentome/utf8.cpp
	youtokentome/utils.cpp
	youtokentome/bpe.cpp
	ClipAPI.cpp)
set(RUCLIP_HEADER_FILES TorchHeader.h
	RuCLIP.h
	json.hpp
	json_fwd.hpp
	youtokentome/utf8.h
	youtokentome/utils.h
	youtokentome/bpe.h
	RuCLIPProcessor.h
        ClipAPI.h)

add_library(${PROJECT_NAME} SHARED ${RUCLIP_SOURCE_FILES} ${RUCLIP_HEADER_FILES})

if (MSVC)
    get_filename_component(full_path_nvtools_lib "C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/nvToolsExt64_1.lib" ABSOLUTE)
    message("${full_path_nvtools_lib}")
    list (REMOVE_ITEM TORCH_LIBRARIES "${full_path_nvtools_lib}")
endif(MSVC)
message("Torch libs: ${TORCH_LIBRARIES}")

set(RUCLIP_LIBS
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME} ${RUCLIP_LIBS})

install(TARGETS ${PROJECT_NAME}
        EXPORT MTTrackingExports
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")
