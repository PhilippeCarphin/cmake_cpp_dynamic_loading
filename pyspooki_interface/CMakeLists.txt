
FUNCTION(ADD_BOOST_PYTHON_MODULE name sources)
    PYTHON_ADD_MODULE(${ARGV})
    target_compile_definitions(${name} PRIVATE THIS_PYTHON_MODULE_NAME=${name})
    target_link_libraries(${name} PRIVATE Boost::${BOOST_PYTHON_COMPONENT_NAME} ${PYTHON_LIBRARIES})
    target_include_directories(${name} PRIVATE ${PYTHON_INCLUDE_DIRS})

    configure_file(test_${name}.in.py test_${name}.out.py)
    add_custom_target(check-${name} COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/test_${name}.out.py)
    add_dependencies(check-${name} ${name})
ENDFUNCTION()

add_custom_target(check-py)

ADD_BOOST_PYTHON_MODULE(absval absval.cpp)
ADD_DEPENDENCIES(absval AbsoluteValue)
ADD_DEPENDENCIES(check-py check-absval)

ADD_BOOST_PYTHON_MODULE(interface_classes interface_classes.cpp)
ADD_DEPENDENCIES(check-py check-interface_classes)

ADD_BOOST_PYTHON_MODULE(receive_numpy_array receive_numpy_array.cpp)
target_link_libraries(receive_numpy_array PRIVATE Boost::${BOOST_PYTHON_NUMPY_COMPONENT_NAME})
add_dependencies(check-py check-receive_numpy_array)


ADD_BOOST_PYTHON_MODULE(numpy_initial_attempts numpy_initial_attempts.cpp)
target_link_libraries(numpy_initial_attempts PRIVATE Boost::${BOOST_PYTHON_NUMPY_COMPONENT_NAME})
ADD_DEPENDENCIES(check-py check-numpy_initial_attempts)

ADD_BOOST_PYTHON_MODULE(numpy_legit_attempts numpy_legit_attempts.cpp)
target_include_directories(numpy_legit_attempts PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
ADD_DEPENDENCIES(check-py check-numpy_legit_attempts)

ADD_BOOST_PYTHON_MODULE(numpy_capsule_way numpy_capsule_way.cpp)
target_link_libraries(numpy_capsule_way PRIVATE Boost::${BOOST_PYTHON_NUMPY_COMPONENT_NAME})
ADD_DEPENDENCIES(check-py check-numpy_capsule_way)

configure_file(setup_test_environment.in.sh ${CMAKE_CURRENT_BINARY_DIR}/setup_test_environment.out.sh)


