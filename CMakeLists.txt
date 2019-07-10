cmake_minimum_required (VERSION 2.8)
project (Tutorial CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)


# SET(PYTHON_VERSION_MAJOR_DOT_MINOR 3.4)
# SET(PYTHON_VERSION_MAJOR_MINOR 34)
SET(Python_ADDITIONAL_VERSIONS 3.4)
FIND_PACKAGE(PythonLibs 3.4 REQUIRED)

FIND_PACKAGE(Boost COMPONENTS python-py34)
message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
message("Boost_LIBRARIES=${Boost_LIBRARIES}")


set(PYSPOOKI_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
enable_testing()


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        .
)

add_executable(spooki_run.out main.cpp)

add_custom_target(check)

target_link_libraries(
    spooki_run.out
        dl
)

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

