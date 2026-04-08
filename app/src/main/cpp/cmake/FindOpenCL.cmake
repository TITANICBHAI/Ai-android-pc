#
# FindOpenCL.cmake — ARIA custom override
#
# Used instead of CMake's built-in FindOpenCL.cmake so that ggml-opencl's
# find_package(OpenCL REQUIRED) resolves to the inline-built stub target
# (aria_opencl_stub) rather than searching the host system (which fails in
# cross-compilation mode with the NDK).
#
# This file is found first because the parent CMakeLists.txt prepends
# ${CMAKE_CURRENT_SOURCE_DIR}/cmake to CMAKE_MODULE_PATH before
# add_subdirectory(llama_build).
#
# Outputs (match the variables CMake's FindOpenCL.cmake would produce):
#   OpenCL_FOUND           BOOL
#   OpenCL_INCLUDE_DIRS    list of include paths
#   OpenCL_LIBRARIES       link target / list
#   OpenCL::OpenCL         IMPORTED or ALIAS target
#

if(TARGET aria_opencl_stub)
    set(OpenCL_FOUND TRUE)
    set(OpenCL_INCLUDE_DIR  "${ARIA_OPENCL_INCLUDE_DIR}")
    set(OpenCL_INCLUDE_DIRS "${ARIA_OPENCL_INCLUDE_DIR}")
    set(OpenCL_LIBRARY      aria_opencl_stub)
    set(OpenCL_LIBRARIES    aria_opencl_stub)
    set(OpenCL_VERSION_STRING "2.0")

    if(NOT TARGET OpenCL::OpenCL)
        add_library(OpenCL::OpenCL ALIAS aria_opencl_stub)
    endif()

    if(NOT OpenCL_FIND_QUIETLY)
        message(STATUS "FindOpenCL (ARIA): using inline stub → aria_opencl_stub")
    endif()
else()
    set(OpenCL_FOUND FALSE)
    if(OpenCL_FIND_REQUIRED)
        message(FATAL_ERROR
            "FindOpenCL (ARIA): aria_opencl_stub target not defined. "
            "Ensure GGML_OPENCL is ON and opencl_stub.c is present in stubs/.")
    else()
        message(STATUS "FindOpenCL (ARIA): aria_opencl_stub not available — OpenCL disabled")
    endif()
endif()
