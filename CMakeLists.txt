cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
project(gustgrid LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA library dirs: ${CUDAToolkit_LIBRARY_DIR}")

find_package(freetype CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP support found: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP_CXX not found. Building without OpenMP.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
if(UNIX)
  set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native -funroll-loops -ftree-vectorize -fprefetch-loop-arrays -flto -ffast-math -DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE  "/O2/ /DNDEBUG")
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")

set(SOURCES
  src/main.cpp
  src/renderer.cpp
  src/simulator.cpp
  src/glad.c
)
set(CUDA_SOURCES
  src/fluidSimulator.cu
)

add_executable(${PROJECT_NAME}
  ${SOURCES}
  ${CUDA_SOURCES}
)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/freetype/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${FREETYPE_INCLUDE_DIRS}
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY      ${CMAKE_SOURCE_DIR}/bin
)

if(NOT CUDAToolkit_INCLUDE_DIRS)
    message(WARNING "CUDAToolkit_INCLUDE_DIRS is empty, trying common paths")
    find_path(CUDA_INCLUDE_DIR cuda_runtime.h
        PATHS
        /usr/local/cuda/include
        /opt/cuda/include
        /usr/include/cuda
        ENV CUDA_PATH
        PATH_SUFFIXES include
    )
    if(CUDA_INCLUDE_DIR)
        include_directories(${CUDA_INCLUDE_DIR})
        message(STATUS "Found CUDA headers at: ${CUDA_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Could not find cuda_runtime.h")
    endif()
endif()

link_directories(${CMAKE_SOURCE_DIR}/lib)

target_link_libraries(${PROJECT_NAME} PRIVATE
    freetype
    glfw
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::curand
    CUDA::cublas
    CUDA::cufft
    CUDA::cusparse
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
if(UNIX)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    dl
    m
  )
endif()

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)