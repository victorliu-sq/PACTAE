## Allow CUDA_ARCH to be specified externally
#if(DEFINED ENV{CUDA_ARCH})
#    set(CUDA_COMPUTE_CAPABILITIES $ENV{CUDA_ARCH})
#else()
#    message(STATUS "CUDA_ARCH not set, using default compute capability 89 (for RTX 4070)")
#    set(CUDA_COMPUTE_CAPABILITIES 89)
#endif()
#
#set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITIES})
#message(STATUS "Using CUDA Compute Capability: ${CUDA_COMPUTE_CAPABILITIES}")
#
## Find CUDA Toolkit
#find_package(CUDAToolkit REQUIRED)
#
#if (CUDAToolkit_FOUND)
#    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
#    message(STATUS "CUDA Toolkit root path: ${CUDAToolkit_ROOT}")
#    message(STATUS "nvcc executable path: ${CUDAToolkit_NVCC_EXECUTABLE}")
#    message(STATUS "CUDA include directory: ${CUDAToolkit_INCLUDE_DIRS}")
#    message(STATUS "CUDA library directories: ${CUDAToolkit_LIBRARY_DIR}")
#
#    # Print Thrust header path (included in CUDA Toolkit)
#    set(THRUST_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/thrust")
#    if(EXISTS ${THRUST_INCLUDE_DIR})
#        message(STATUS "Thrust Include Directory: ${THRUST_INCLUDE_DIR}")
#    else()
#        message(WARNING "Thrust directory not found at expected path: ${THRUST_INCLUDE_DIR}")
#    endif()
#
#    include_directories(${CUDAToolkit_INCLUDE_DIRS})
#else ()
#    message(FATAL_ERROR "CUDA Toolkit is not found! Please verify your CUDA installation and envrionment setting!")
#endif ()

# ================== CUDA Architecture Detection ==================
# Respect explicit user choice first:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        # Auto-detect from currently visible GPUs
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        # Fallback for older CMake: use env CUDA_ARCH if present, else default
        if(DEFINED ENV{CUDA_ARCH})
            set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH})
        else()
            message(STATUS "CMAKE<3.24 and CUDA_ARCH not set; defaulting to 89")
            set(CMAKE_CUDA_ARCHITECTURES 89)
        endif()
    endif()
endif()

# Always print what we're using
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ================== CUDA Toolkit Discovery ==================
find_package(CUDAToolkit REQUIRED)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit root path: ${CUDAToolkit_ROOT}")
    message(STATUS "nvcc executable path: ${CUDAToolkit_NVCC_EXECUTABLE}")
    message(STATUS "CUDA include directory: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA library directories: ${CUDAToolkit_LIBRARY_DIR}")

    # Thrust path check
    set(THRUST_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/thrust")
    if(EXISTS ${THRUST_INCLUDE_DIR})
        message(STATUS "Thrust Include Directory: ${THRUST_INCLUDE_DIR}")
    else()
        message(WARNING "Thrust directory not found at expected path: ${THRUST_INCLUDE_DIR}")
    endif()

    include_directories(${CUDAToolkit_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "CUDA Toolkit not found! Please verify your CUDA installation and environment settings!")
endif()