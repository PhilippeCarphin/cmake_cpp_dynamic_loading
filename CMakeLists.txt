cmake_minimum_required (VERSION 2.8)
project (Tutorial CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)


# NOTE: This variable must be declared before the find_package
# for the call to work with version 3.4.
# The find_package doesn't use the variable as an argument
# to make clear the fact that the the variable is there for
# another reason.
SET(Python_ADDITIONAL_VERSIONS 3.4)
FIND_PACKAGE(PythonLibs 3.4 REQUIRED)

# Note the way the version is written, I'm using the least amount of
# variables so that we see the values themselves in the calls
FIND_PACKAGE(Boost COMPONENTS python-py34)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        .
)

add_executable(spooki_run.out main.cpp)
target_link_libraries(spooki_run.out dl)

add_custom_target(check)

# Or we could do this in plugins/AbsoluteValue/CMakeLists
add_dependencies(spooki_run.out AbsoluteValue)

add_subdirectory(meteo_operations)
add_subdirectory(plugins)

add_subdirectory(pyspooki)

add_dependencies(check
   python_interface_test
   pyspooki_interface
)

configure_file(cmake_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config.out.h)


