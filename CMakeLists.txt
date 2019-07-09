cmake_minimum_required (VERSION 2.6)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)

project (Tutorial)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        .
)

add_executable(spooki_run.out main.cpp)

add_subdirectory(meteo_operations)
add_subdirectory(plugins)

configure_file(cmake_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config.out.h)

