cmake_minimum_required(VERSION 3.10)
project(neurominisat)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS                "-Wall -Wextra -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG          "-g3")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")

set(PYBIND11_CPP_STANDARD -std=c++14)
set(PYBIND11_CPP_STANDARD -O3)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

find_package(ZLIB)
include_directories(${ZLIB_INCLUDE_DIR})

find_package(minisat)
include_directories(${MINISAT_INCLUDE_DIR})

find_package(pybind11 REQUIRED)
include_directories(/usr/local/include/eigen3)

pybind11_add_module(neurominisat neurominisat.cxx neurosolver.cxx)
target_link_libraries(neurominisat PUBLIC ${ZLIB_LIBRARY})
target_link_libraries(neurominisat PUBLIC ${MINISAT_LIBRARIES})
