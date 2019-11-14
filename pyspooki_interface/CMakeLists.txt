
FUNCTION(ADD_BOOST_PYTHON_MODULE name sources)
    PYTHON_ADD_MODULE(${ARGV})
    target_compile_definitions(${name} PRIVATE THIS_PYTHON_MODULE_NAME=${name})
    target_link_libraries(${name} PRIVATE Boost::${BOOST_PYTHON_COMPONENT_NAME} ${PYTHON_LIBRARIES})
    target_include_directories(${name} PRIVATE ${PYTHON_INCLUDE_DIRS})
    install(TARGETS ${name} CONFIGURATIONS Debug LIBRARY DESTINATION ${PYTHON_USER_INSTALL_SITE}/dev)
    install(TARGETS ${name} CONFIGURATIONS Release LIBRARY DESTINATION ${PYTHON_USER_INSTALL_SITE})

    configure_file(test_${name}.in.py test_${name}.out.py)
    add_custom_target(check-${name} COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/test_${name}.out.py)
    add_dependencies(check-${name} ${name})
ENDFUNCTION()

add_custom_target(check-py)

ADD_SUBDIRECTORY(absval)
ADD_SUBDIRECTORY(interface_classes)
ADD_SUBDIRECTORY(receive_numpy_array)
ADD_SUBDIRECTORY(numpy_initial_attempts)
ADD_SUBDIRECTORY(numpy_legit_attempts)
ADD_SUBDIRECTORY(numpy_capsule_way)

configure_file(setup_test_environment.in.sh ${CMAKE_CURRENT_BINARY_DIR}/setup_test_environment.out.sh)


