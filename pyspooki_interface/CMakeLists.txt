PYTHON_ADD_MODULE(pyspooki_interface pyspooki_interface.cpp)

list(GET Boost_LIBRARY_DIRS 0 Boost_LD_PATH)

target_include_directories(
        pyspooki_interface
    PRIVATE
        ${PYTHON_INCLUDE_DIRS}
)

if(${APPLE})
    target_include_directories(
            pyspooki_interface
        PRIVATE
            ${Python3_NumPy_INCLUDE_DIRS}
    )
endif()

target_link_libraries(
        pyspooki_interface
    PRIVATE
        meteo_operations
        Boost::${BOOST_PYTHON_NUMPY_COMPONENT_NAME}
        Boost::${BOOST_PYTHON_COMPONENT_NAME}
        ${PYTHON_LIBRARIES}
)

if(NOT ON_UBUNTU_14_PPP)
    target_compile_definitions(pyspooki_interface PUBLIC "USE_BOOST_NUMPY")
endif()
target_compile_options(pyspooki_interface PRIVATE -Wpedantic)




PYTHON_ADD_MODULE(absval absval.cpp)
target_include_directories(absval PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(absval PRIVATE Boost::${BOOST_PYTHON_COMPONENT_NAME} ${PYTHON_LIBRARIES})
add_dependencies(absval AbsoluteValue)

add_custom_target(check-absval COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/test_absval.out.py)
configure_file(test_absval.in.py test_absval.out.py)
add_dependencies(check-absval absval)


PYTHON_ADD_MODULE(interface_classes interface_classes.cpp)
target_compile_definitions(interface_classes PRIVATE THIS_PYTHON_MODULE_NAME=interface_classes)
target_link_libraries(interface_classes PRIVATE Boost::${BOOST_PYTHON_COMPONENT_NAME} ${PYTHON_LIBRARIES})
target_include_directories(interface_classes PRIVATE ${PYTHON_INCLUDE_DIRS})

add_custom_target(check-interface_classes COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/test_interface_classes.out.py)
configure_file(test_interface_classes.in.py test_interface_classes.out.py)
add_dependencies(check-interface_classes interface_classes)


PYTHON_ADD_MODULE(numpy_initial_attempts numpy_initial_attempts.cpp)
target_compile_definitions(numpy_initial_attempts PRIVATE THIS_PYTHON_MODULE_NAME=numpy_initial_attempts)
target_link_libraries(numpy_initial_attempts PRIVATE Boost::${BOOST_PYTHON_COMPONENT_NAME} Boost::${BOOST_PYTHON_NUMPY_COMPONENT_NAME} ${PYTHON_LIBRARIES})
target_include_directories(numpy_initial_attempts PRIVATE ${PYTHON_INCLUDE_DIRS})

add_custom_target(check-numpy_initial_attempts COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/test_numpy_initial_attempts.out.py)
configure_file(test_numpy_initial_attempts.in.py test_numpy_initial_attempts.out.py)
add_dependencies(check-numpy_initial_attempts numpy_initial_attempts)




PYTHON_ADD_MODULE(numpy_legit_attempts numpy_legit_attempts.cpp)
target_compile_definitions(numpy_legit_attempts PRIVATE THIS_PYTHON_MODULE_NAME=numpy_legit_attempts)
target_link_libraries(numpy_legit_attempts PRIVATE Boost::${BOOST_PYTHON_COMPONENT_NAME} Boost::${BOOST_PYTHON_NUMPY_COMPONENT_NAME} ${PYTHON_LIBRARIES})
target_include_directories(numpy_legit_attempts PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(numpy_legit_attempts PRIVATE ${Python3_NumPy_INCLUDE_DIRS})

add_custom_target(check-numpy_legit_attempts COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/test_numpy_legit_attempts.out.py)
configure_file(test_numpy_legit_attempts.in.py test_numpy_legit_attempts.out.py)
add_dependencies(check-numpy_legit_attempts numpy_legit_attempts)


configure_file(test_python.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh)
configure_file(test_loading.py.in ${CMAKE_CURRENT_BINARY_DIR}/test_loading.out.py)
configure_file(test_numpy.in.py ${CMAKE_CURRENT_BINARY_DIR}/test_numpy.out.py)
configure_file(test_numpy.in.sh ${CMAKE_CURRENT_BINARY_DIR}/test_numpy.out.sh)
configure_file(setup_test_environment.in.sh ${CMAKE_CURRENT_BINARY_DIR}/setup_test_environment.out.sh)
add_custom_target(
        python_interface_test
    COMMAND
        # The key that was holding me back was this:  The boost libraries were
        # not in LD_LIBRARY_PATH
        env -i ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh
)

add_dependencies(python_interface_test
    pyspooki_interface
    AbsoluteValue
)

add_custom_target(
        check-numpy
        COMMAND
        env -i ${CMAKE_CURRENT_BINARY_DIR}/test_numpy.out.sh
)

add_dependencies( check-numpy
        pyspooki_interface
)
