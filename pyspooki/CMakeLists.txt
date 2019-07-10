
add_library(pyspooki_interface SHARED pyspooki_interface.cpp)
configure_file(test_python.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh)

target_include_directories(
        pyspooki_interface
        PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

target_link_libraries(
        pyspooki_interface
        PRIVATE
        meteo_operations
        ${Boost_LIBRARIES}
        # Note the non-necessity of ${PYTHON_LIBRARIES} because BOOS
)

add_dependencies(
    pyspooki_interface
    meteo_operations
)

add_custom_target(
        python_interface_test
        COMMAND
        env -i LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${Boost_LIBRARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh
)

add_dependencies(python_interface_test
    pyspooki_interface
)

add_dependencies(
    check
    python_interface_test
)
