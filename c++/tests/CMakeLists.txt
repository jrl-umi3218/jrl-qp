if(WIN32)
  #This is one of the way to avoid link errors related to static variables in program_options
  add_definitions(/bigobj)
endif(WIN32)

macro(addUnitTest name)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    add_executable(${name} ${name}.cpp ${ARGN})
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.in.cpp)
    configure_file(${name}.in.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp)
    add_executable(${name} ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(SEND_ERROR "Cannot add a test ${name}, neither ${name}.cpp nor ${name}.in.cpp exists in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  target_link_libraries(${name} JRLQP)
  add_test(${name}Unit ${name})
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${name} "${CMAKE_INSTALL_PREFIX}/bin")
endmacro(addUnitTest)


set(TESTSOURCES
    main.cpp
)

set(TESTHEADERS
)

add_executable(main ${TESTSOURCES} ${TESTHEADERS})
target_link_libraries(main JRLQP)
GENERATE_MSVC_DOT_USER_FILE(main "${CMAKE_INSTALL_PREFIX}/bin")

