set(JRLQP_SOURCES
  DualSolver.cpp
  GoldfarbIdnaniSolver.cpp
  SolverOptions.cpp
  experimental/BoxAndSingleConstraintSolver.cpp
  internal/ActiveSet.cpp
  internal/memoryChecks.cpp
  test/kkt.cpp
  test/problems.cpp
  test/randomProblems.cpp
  utils/Logger.cpp
)

set(JRLQP_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/jrl-qp)
set(JRLQP_HEADERS
  ${JRLQP_INCLUDE_DIR}/api.h
  ${JRLQP_INCLUDE_DIR}/defs.h
  ${JRLQP_INCLUDE_DIR}/DualSolver.h
  ${JRLQP_INCLUDE_DIR}/enums.h
  ${JRLQP_INCLUDE_DIR}/GoldfarbIdnaniSolver.h
  ${JRLQP_INCLUDE_DIR}/SolverOptions.h
  ${JRLQP_INCLUDE_DIR}/experimental/BoxAndSingleConstraintSolver.h
  ${JRLQP_INCLUDE_DIR}/internal/ActiveSet.h
  ${JRLQP_INCLUDE_DIR}/internal/meta.h
  ${JRLQP_INCLUDE_DIR}/internal/memoryChecks.h
  ${JRLQP_INCLUDE_DIR}/internal/ConstraintNormal.h
  ${JRLQP_INCLUDE_DIR}/internal/TerminationType.h
  ${JRLQP_INCLUDE_DIR}/internal/Workspace.h
  ${JRLQP_INCLUDE_DIR}/test/kkt.h
  ${JRLQP_INCLUDE_DIR}/test/problems.h
  ${JRLQP_INCLUDE_DIR}/test/randomMatrices.h
  ${JRLQP_INCLUDE_DIR}/test/randomProblems.h
  ${JRLQP_INCLUDE_DIR}/utils/debug.h
  ${JRLQP_INCLUDE_DIR}/utils/Logger.h
  ${JRLQP_INCLUDE_DIR}/utils/toMatlab.h
)

#Helper functions to assign files to groups in IDE, mimicking the organization of files on disk
#Adapted from https://stackoverflow.com/a/31423421
function(assign_source_group folder)
  foreach(_source IN ITEMS ${ARGN})
    if (IS_ABSOLUTE "${_source}")
        file(RELATIVE_PATH _source_rel "${CMAKE_SOURCE_DIR}" "${_source}")
    else()
        set(_source_rel "${_source}")
    endif()
    #remove include/jrl-qp/ if present in the path
    string(REPLACE "include/jrl-qp/" "" _source_rel_modified "${_source_rel}")
    get_filename_component(_source_path "${_source_rel_modified}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${folder}\\${_source_path_msvc}" FILES "${_source}")
  endforeach()
endfunction(assign_source_group)

assign_source_group("sources" ${JRLQP_SOURCES})
assign_source_group("headers" ${JRLQP_HEADERS})

add_library(jrl-qp SHARED ${JRLQP_SOURCES} ${JRLQP_HEADERS})
target_include_directories(jrl-qp PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<INSTALL_INTERFACE:include>)
if(TARGET Eigen3::Eigen)
  target_link_libraries(jrl-qp PUBLIC Eigen3::Eigen)
else()
  target_include_directories(jrl-qp SYSTEM PUBLIC "${EIGEN3_INCLUDE_DIR}")
endif()
set_target_properties(jrl-qp PROPERTIES COMPILE_FLAGS "-DJRLQP_EXPORTS -DEIGEN_RUNTIME_NO_MALLOC")
set_target_properties(jrl-qp PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR} VERSION ${PROJECT_VERSION})
set_target_properties(jrl-qp PROPERTIES CXX_STANDARD 17)
target_compile_definitions(jrl-qp PUBLIC ${SOLVER_FLAGS})
if(NOT WIN32)
  target_compile_options(jrl-qp PUBLIC "-Wall" "-Wextra" "-Wno-sign-conversion" "-pedantic")
else()
  target_compile_options(jrl-qp PUBLIC "-D_ENABLE_EXTENDED_ALIGNED_STORAGE" "-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING")
endif()

if(GENERATE_COVERAGE)
  target_compile_options(jrl-qp PRIVATE "--coverage")
  set_target_properties(jrl-qp PROPERTIES LINK_FLAGS "--coverage")
endif()

install(TARGETS jrl-qp
  EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)
install(DIRECTORY ${JRLQP_INCLUDE_DIR}/ DESTINATION ${INCLUDE_INSTALL_DESTINATION})
pkg_config_append_libs(jrl-qp)
