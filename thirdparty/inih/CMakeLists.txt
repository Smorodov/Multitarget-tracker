CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

PROJECT(inih)

set(INIH_SOURCE ini.c INIReader.cpp)

set(INIH_HEADERS ini.h INIReader.h)

ADD_LIBRARY(inih ${INIH_SOURCE} ${INIH_HEADERS})

set_target_properties(inih PROPERTIES FOLDER "libs")

install(TARGETS ${PROJECT_NAME}
        EXPORT MTTrackingExports
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")