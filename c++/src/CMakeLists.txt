set(JRLQP_SOURCES
	DualSolver.cpp
	GoldfarbIdnaniSolver.cpp
	SolverOptions.cpp
	internal/ActiveSet.cpp
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
  ${JRLQP_INCLUDE_DIR}/internal/ActiveSet.h
  ${JRLQP_INCLUDE_DIR}/internal/meta.h
  ${JRLQP_INCLUDE_DIR}/internal/ConstraintNormal.h
  ${JRLQP_INCLUDE_DIR}/internal/TerminationType.h
  ${JRLQP_INCLUDE_DIR}/internal/Workspace.h
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

add_library(JRLQP SHARED ${JRLQP_SOURCES} ${JRLQP_HEADERS})
target_include_directories(JRLQP PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include> $<INSTALL_INTERFACE:include>)
target_link_libraries(JRLQP PUBLIC Eigen3::Eigen)
set_target_properties(JRLQP PROPERTIES COMPILE_FLAGS "-DJRLQP_EXPORTS")
set_target_properties(JRLQP PROPERTIES SOVERSION 0 VERSION 0.8.0)
set_target_properties(JRLQP PROPERTIES CXX_STANDARD 17)
target_compile_definitions(JRLQP PUBLIC ${SOLVER_FLAGS})

if(GENERATE_COVERAGE)
  target_compile_options(JRLQP PRIVATE "--coverage")
  set_target_properties(JRLQP PROPERTIES LINK_FLAGS "--coverage")
endif()


install(TARGETS JRLQP
	EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DESTINATION})
