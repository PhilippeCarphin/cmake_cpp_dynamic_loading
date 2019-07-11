
# add_library(pyspooki_interface SHARED pyspooki_interface.cpp)
PYTHON_ADD_MODULE(pyspooki_interface pyspooki_interface.cpp)

list(GET Boost_LIBRARY_DIRS 0 Boost_LD_PATH)
configure_file(test_python.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh)
configure_file(test_loading.py.in ${CMAKE_CURRENT_BINARY_DIR}/test_loading.out.py)

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

if(${APPLE})
  target_link_libraries(
      pyspooki_interface
    PRIVATE
      ${PYTHON_LIBRARIES}
  )
endif()

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

add_dependencies(
    check
    python_interface_test
)
