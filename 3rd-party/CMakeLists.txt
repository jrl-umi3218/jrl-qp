##########################
# -- Google benchmark -- #
##########################

# Disable tests
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
if(NOT WIN32)
  add_definitions(-Wno-error)
endif()
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmark EXCLUDE_FROM_ALL)
endif()

install(FILES ${PROJECT_SOURCE_DIR}/3rdParty/effolkronium/random.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include/jrl-qp/3rdParty/effolkronium/)
