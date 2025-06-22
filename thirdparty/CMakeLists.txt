add_subdirectory(inih)
#add_subdirectory(Circular_Code)
add_subdirectory(spdlog)

if (USE_CLIP)
    add_subdirectory(ruclip)
endif(USE_CLIP)
