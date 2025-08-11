cmake_minimum_required(VERSION 3.8)

set(project_name bambooSMP)
set(version 2025.8)

project(${project_name}
        VERSION ${version}
        LANGUAGES C CXX CUDA
)

# ---------- Optional AE deps auto-pickup ----------
# Prefer env AE_DEPS_DIR; else use repo-local .ae/deps
set(_AE_DEPS "$ENV{AE_DEPS_DIR}")
if (NOT _AE_DEPS)
    set(_AE_DEPS "${CMAKE_SOURCE_DIR}/.ae/deps")
endif ()
if (EXISTS "${_AE_DEPS}")
    # Make packages installed there discoverable (e.g., glog-config.cmake)
    list(APPEND CMAKE_PREFIX_PATH "${_AE_DEPS}")
    message(STATUS "Added to CMAKE_PREFIX_PATH: ${_AE_DEPS}")

    # Help your custom FindGlog.cmake too (it looks at GLOG_ROOT_DIR)
    if (NOT DEFINED GLOG_ROOT_DIR)
        set(GLOG_ROOT_DIR "${_AE_DEPS}" CACHE PATH "Hint path for glog root" FORCE)
    endif ()
endif ()

# ---------- Build type / language flags ----------
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(CMAKE_CXX_FLAGS_RELEASE " -O3 -DNDEBUG -fno-tree-vectorize -fno-slp-vectorize -fno-aggressive-loop-optimizations -fno-unroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --extended-lambda -Xptxas -O0 -Xcompiler -fno-strict-aliasing -fwrapv")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# ================== Dependencies ======================
# CUDA
include(cmake/FindCUDA.cmake)

# glog
include("cmake/FindGlog.cmake")
if (NOT GLOG_FOUND)
    message(FATAL_ERROR "glog not found, please install the glog library")
else ()
    include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
endif ()

# Add this line right before the add_subdirectory commands:
add_compile_definitions(GLOG_USE_GLOG_EXPORT)

# both source files and test files
# will try to include header files from this directory
set(CUDA_HOME $ENV{CUDA_HOME})
set(HOME $ENV{HOME})

include_directories(${PROJECT_SOURCE_DIR}/include)
# export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# src files
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)

