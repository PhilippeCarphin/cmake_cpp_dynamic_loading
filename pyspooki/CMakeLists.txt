
add_library(pyspooki_interface SHARED pyspooki_interface.cpp)


FIND_PACKAGE(Boost COMPONENTS python-py36)
message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
message("Boost_LIBRARIES=${Boost_LIBRARIES}")

FIND_PACKAGE(PythonLibs 3.6)

set(PYSPOOKI_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(test_python.sh.in ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh)

message("PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES=${PYTHON_LIBRARIES}")

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

add_custom_target(
        python_interface_test
        COMMAND
        ${CMAKE_CURRENT_BINARY_DIR}/test_python.out.sh
)

add_dependencies(check python_interface_test)
