PYTHON_ADD_MODULE(pyspooki_interface pyspooki_interface.cpp)

list(GET Boost_LIBRARY_DIRS 0 Boost_LD_PATH)
configure_file(test_python.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh)
configure_file(test_loading.py.in ${CMAKE_CURRENT_BINARY_DIR}/test_loading.out.py)
configure_file(test_numpy.in.py ${CMAKE_CURRENT_BINARY_DIR}/test_numpy.out.py)
configure_file(test_numpy.in.sh ${CMAKE_CURRENT_BINARY_DIR}/test_numpy.out.sh)
configure_file(setup_test_environment.in.sh ${CMAKE_CURRENT_BINARY_DIR}/setup_test_environment.out.sh)

target_include_directories(
        pyspooki_interface
    PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
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

add_dependencies(
    pyspooki_interface
    meteo_operations
)

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
