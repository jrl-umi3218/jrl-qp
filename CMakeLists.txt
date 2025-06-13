cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME jrl-qp)
set(PROJECT_DESCRIPTION "...")
set(PROJECT_URL "")
set(PROJECT_DEBUG_POSTFIX "_d")
set(PROJECT_VERSION 0.1.0)
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_GENERATED_HEADERS_SKIP_DEPRECATED ON)
set(PROJECT_GENERATED_HEADERS_SKIP_CONFIG ON)
set(PROJECT_GENERATED_HEADERS_SKIP_WARNING ON)

set(CMAKE_CXX_STANDARD 17)

include(cmake/base.cmake)
include(cmake/msvc-specific.cmake)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(DOXYGEN_USE_MATHJAX "YES")
# allow to document a group of items and generate a documentation for each of
# them
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)

project(jrl-qp LANGUAGES CXX)

# If we were given coverage options, we will only pass them to the core library.
set(GENERATE_COVERAGE
    FALSE
    CACHE BOOL "Generate coverage data")
if("${CMAKE_CXX_FLAGS}" MATCHES "--coverage")
  set(GENERATE_COVERAGE
      TRUE
      CACHE BOOL "Generate coverage data" FORCE)
  string(REPLACE "--coverage" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "--coverage" "" CMAKE_EXE_LINKER_FLAGS
                 "${CMAKE_EXE_LINKER_FLAGS}")
  string(REPLACE "--coverage" "" CMAKE_MODULE_LINKER_FLAGS
                 "${CMAKE_MODULE_LINKER_FLAGS}")
endif()

# ##############################################################################
# * External dependencies - #
# ##############################################################################

# ##############################################################################
# * Eigen - #
# ##############################################################################
find_package(Eigen3 QUIET NO_CMAKE_PACKAGE_REGISTRY)
if(Eigen3_FOUND)
  add_project_dependency(Eigen3 REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
else()
  add_project_dependency(Eigen3 MODULE REQUIRED)
endif()

# For MSVC, set local environment variable to enable finding the built dll of
# the main library when launching ctest with RUN_TESTS and use solution folders.
if(MSVC)
  set(CMAKE_MSVCIDE_RUN_PATH
      "\$(SolutionDir)/src/\$(Configuration);${CMAKE_INSTALL_PREFIX}/bin")
endif(MSVC)

# allow for solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(BUILD_BENCHMARKS "Build benchmarks" ON)

add_subdirectory(3rd-party)
add_subdirectory(src)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
