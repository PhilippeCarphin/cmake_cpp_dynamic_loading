cmake_minimum_required (VERSION 2.6)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)

project (Tutorial)
enable_testing()
add_custom_target(check)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        .
)

add_executable(spooki_run.out main.cpp)
target_link_libraries(
    spooki_run.out
        dl
)

# Or we could do this in plugins/AbsoluteValue/CMakeLists
add_dependencies(spooki_run.out AbsoluteValue)

add_subdirectory(meteo_operations)
add_subdirectory(plugins)

add_subdirectory(pyspooki)

configure_file(cmake_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config.out.h)

