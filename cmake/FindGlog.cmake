# This file is used to find glog library in CMake script, based on code
# from https://github.com/BVLC/caffe/blob/master/cmake/Modules/FindGlog.cmake
# (2-Clause BSD License).
#
# - Try to find Glog
#
# Optionally searched for defaults:
#   GLOG_ROOT_DIR: Base directory where all GLOG components are found
#
# Set after configuration:
#   GLOG_FOUND
#   GLOG_INCLUDE_DIRS
#   GLOG_LIBRARIES
#   GLOG_LIBRARY_DIRS     # <--- added: directory/directories that contain libglog

include(FindPackageHandleStandardArgs)

set(GLOG_ROOT_DIR "" CACHE PATH "Folder contains Google glog")

# --- headers ---
if (WIN32)
    find_path(GLOG_INCLUDE_DIR glog/logging.h
            PATHS ${GLOG_ROOT_DIR}/src/windows)
else ()
    find_path(GLOG_INCLUDE_DIR glog/logging.h
            PATHS ${GLOG_ROOT_DIR})
endif ()

# --- library ---
if (MSVC)
    find_library(GLOG_LIBRARY_RELEASE libglog_static
            PATHS ${GLOG_ROOT_DIR}
            PATH_SUFFIXES Release lib lib64)
    find_library(GLOG_LIBRARY_DEBUG libglog_static
            PATHS ${GLOG_ROOT_DIR}
            PATH_SUFFIXES Debug lib lib64)

    # For MSVC, express the multi-config pair
    set(GLOG_LIBRARY optimized ${GLOG_LIBRARY_RELEASE} debug ${GLOG_LIBRARY_DEBUG})

    # Derive library dirs (remove duplicates)
    set(GLOG_LIBRARY_DIRS "")
    if (GLOG_LIBRARY_RELEASE)
        get_filename_component(_glog_lib_dir_rel "${GLOG_LIBRARY_RELEASE}" DIRECTORY)
        list(APPEND GLOG_LIBRARY_DIRS "${_glog_lib_dir_rel}")
    endif ()
    if (GLOG_LIBRARY_DEBUG)
        get_filename_component(_glog_lib_dir_dbg "${GLOG_LIBRARY_DEBUG}" DIRECTORY)
        list(APPEND GLOG_LIBRARY_DIRS "${_glog_lib_dir_dbg}")
    endif ()
    list(REMOVE_DUPLICATES GLOG_LIBRARY_DIRS)
else ()
    find_library(GLOG_LIBRARY glog
            PATHS ${GLOG_ROOT_DIR}
            PATH_SUFFIXES lib lib64)
    # Derive single library dir
    if (GLOG_LIBRARY)
        get_filename_component(GLOG_LIBRARY_DIRS "${GLOG_LIBRARY}" DIRECTORY)
    endif ()
endif ()

# --- result ---
find_package_handle_standard_args(Glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

set(CMAKE_BUILD_TYPE Debug)

if (GLOG_FOUND)
    set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
    set(GLOG_LIBRARIES ${GLOG_LIBRARY})

    message(STATUS "Found glog")
    message(STATUS "  include dir : ${GLOG_INCLUDE_DIRS}")
    message(STATUS "  library     : ${GLOG_LIBRARIES}")
    message(STATUS "  library dir : ${GLOG_LIBRARY_DIRS}")

    mark_as_advanced(GLOG_ROOT_DIR GLOG_LIBRARY_RELEASE GLOG_LIBRARY_DEBUG
            GLOG_LIBRARY GLOG_INCLUDE_DIR)
endif ()
