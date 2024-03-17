cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ruclip)

find_package(Torch REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/youtokentome)
include_directories(${CMAKE_SOURCE_DIR}/youtokentome/third_party)

file(GLOB RUCLIP_SOURCE_FILES *.cpp *.cpp)
file(GLOB RUCLIP_HEADER_FILES *.h* *.h*)

add_library(${PROJECT_NAME}  SHARED ${RUCLIP_SOURCE_FILES} ${RUCLIP_HEADER_FILES})

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
