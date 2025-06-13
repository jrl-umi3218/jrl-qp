if(WIN32)
  # This is one of the way to avoid link errors related to static variables in
  # program_options
  add_definitions(/bigobj)
endif(WIN32)

option(WITH_QPS_TESTS "Use QPS tests" OFF)
if(WITH_QPS_TESTS)
  set(QPS_DIR
      ""
      CACHE PATH "Path to the QPS data directory")
  if(QPS_DIR STREQUAL "")
    message(FATAL_ERROR "You need to specify a path to the QPS data directory.")
  endif()
endif()

macro(addUnitTest name)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    add_executable(${name} ${name}.cpp ${ARGN})
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.in.cpp)
    configure_file(${name}.in.cpp ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp)
    add_executable(${name} ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(
      SEND_ERROR
        "Cannot add a test ${name}, neither ${name}.cpp nor ${name}.in.cpp exists in ${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()
  if(WITH_QPS_TESTS)
    target_compile_definitions(${name} PUBLIC "-DQPS_TESTS_DIR=\"${QPS_DIR}/\"")
  endif()
  target_link_libraries(${name} jrl-qp)
  add_test(${name}Unit ${name})
  # Adding a project configuration file (for MSVC only)
  generate_msvc_dot_user_file(${name} "${CMAKE_INSTALL_PREFIX}/bin")
  set_target_properties(${name} PROPERTIES FOLDER "Tests")
endmacro(addUnitTest)

set(MultiIK_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MultiIK/arrowAllData.txt
    ${CMAKE_CURRENT_BINARY_DIR}/MultiIK/triBlockDiag_a.txt
    ${CMAKE_CURRENT_BINARY_DIR}/MultiIK/triBlockDiag_C.txt
    ${CMAKE_CURRENT_BINARY_DIR}/MultiIK/triBlockDiag_G.txt
    ${CMAKE_CURRENT_BINARY_DIR}/MultiIK/triBlockDiag_sol.txt
    ${CMAKE_CURRENT_BINARY_DIR}/MultiIK/triBlockDiag_u.txt)

add_custom_command(
  OUTPUT ${MultiIK_FILES}
  COMMAND ${CMAKE_COMMAND} -E tar xf ${CMAKE_CURRENT_SOURCE_DIR}/MultiIK.zip
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/MultiIK.zip
  COMMENT "Extract MultiIK archive")
add_custom_target(extra-multiik-archive DEPENDS ${MultiIK_FILES})

addunittest(ActiveSetTest)
addunittest(blockArrowLLTTest)
addunittest(BlockGISolverTest IKmatReader.cpp)
add_dependencies(BlockGISolverTest extra-multiik-archive)
addunittest(BoxAndSingleConstraintSolverTest)
addunittest(GoldfarbIdnaniSolverTest QPSReader.cpp)
addunittest(InternalTest)
addunittest(RandomProblemsTest)
addunittest(StructuredTest)
addunittest(triBlockDiagLLTTest)

set(TESTSOURCES main.cpp)

set(TESTHEADERS)

add_executable(main ${TESTSOURCES} ${TESTHEADERS} QPSReader.cpp)
target_link_libraries(main jrl-qp)
generate_msvc_dot_user_file(main "${CMAKE_INSTALL_PREFIX}/bin")
