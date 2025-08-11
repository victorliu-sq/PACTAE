# Get list of all source files with given pattern
file(GLOB_RECURSE bamboosmp_sources "${PROJECT_SOURCE_DIR}/src/*.cu")

# iterate each source file to create object files individually
add_library(bamboosmp_lib SHARED ${bamboosmp_sources})
target_link_libraries(bamboosmp_lib PUBLIC
        cuda
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES})

#target_compile_options(bamboosmp_lib PUBLIC
#        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)