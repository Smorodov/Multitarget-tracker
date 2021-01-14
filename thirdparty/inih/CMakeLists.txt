CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

PROJECT(inih)

set(INIH_SOURCE ini.c INIReader.cpp)

set(INIH_HEADERS ini.h INIReader.h)

ADD_LIBRARY(inih ${INIH_SOURCE} ${INIH_HEADERS})

set_target_properties(inih PROPERTIES FOLDER "libs")
