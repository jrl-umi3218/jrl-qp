set(SOLVER_FLAGS "")
set(SOLVER_LIBS "")

###################
# - eigen-lssol - #
###################
add_project_dependency(eigen-lssol QUIET)
if(${eigen-lssol_FOUND})
  list(APPEND SOLVER_FLAGS "-DJRLQP_USE_LSSOL")
  list(APPEND SOLVER_LIBS eigen-lssol::eigen-lssol)
endif()

###################
# -  eigen-qld  - #
###################
add_project_dependency(eigen-qld QUIET)
if(${eigen-qld_FOUND})
  list(APPEND SOLVER_FLAGS "-DJRLQP_USE_QLD")
  list(APPEND SOLVER_LIBS eigen-qld::eigen-qld)
endif()

###################
#- eigen-quadprog-#
###################
add_project_dependency(eigen-quadprog QUIET)
if(${eigen-quadprog_FOUND})
  list(APPEND SOLVER_FLAGS "-DJRLQP_USE_QUADPROG")
  list(APPEND SOLVER_LIBS eigen-quadprog::eigen-quadprog)
endif()


add_custom_target(jrlqp_benchmarks)

macro(addBenchmark name)
  set(benchName ${name}_Bench)
  add_executable(${benchName} ${name}.cpp ${ARGN})
  add_custom_command(TARGET jrlqp_benchmarks
    COMMAND ${benchName} --benchmark_out="${CMAKE_CURRENT_SOURCE_DIR}/out/${name}.csv" --benchmark_out_format=csv
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ${name} benchmark"
    )
  target_link_libraries(${benchName} PUBLIC JRLQP benchmark ${SOLVER_LIBS})
  if(NOT WIN32)
    target_link_libraries(${benchName} PUBLIC pthread)
  else()
    target_link_libraries(${benchName} PUBLIC shlwapi)
  endif()
  target_compile_definitions(${benchName} PUBLIC ${SOLVER_FLAGS})
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${benchName} "${CMAKE_INSTALL_PREFIX}/bin")
  set_target_properties(${benchName} PROPERTIES FOLDER "Benchmarks")
endmacro()

macro(addEigenBenchmark name)
  set(benchName ${name}_Bench)
  add_executable(${benchName} ${name}.cpp ${ARGN})
  add_custom_command(TARGET jrlqp_benchmarks
    COMMAND ${benchName} --benchmark_out="${CMAKE_CURRENT_SOURCE_DIR}/out/${name}.csv" --benchmark_out_format=csv
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ${name} benchmark"
    )
  target_link_libraries(${benchName} PUBLIC Eigen3::Eigen benchmark)
  if(NOT WIN32)
    target_link_libraries(${benchName} PUBLIC pthread)
  else()
    target_link_libraries(${benchName} PUBLIC shlwapi)
  endif()
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${benchName} "${CMAKE_INSTALL_PREFIX}/bin")
  set_target_properties(${benchName} PROPERTIES FOLDER "Benchmarks")
endmacro()

file(MAKE_DIRECTORY out)

addEigenBenchmark(BasicEigen)
addEigenBenchmark(Decomposition)
addEigenBenchmark(LinearSystemSolving)

addBenchmark(Solvers problemAdaptors.cpp)
addBenchmark(BoxAndSingleConstraintSolver)

add_custom_command(TARGET jrlqp_benchmarks
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/generatePlot.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)
