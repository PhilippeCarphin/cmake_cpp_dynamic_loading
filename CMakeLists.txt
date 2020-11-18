cmake_minimum_required (VERSION 2.8)

project (Tutorial CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################# Finding python ############################################

find_package(PythonLibs 3.7 REQUIRED)

execute_process(COMMAND python3 -m site --user-site OUTPUT_VARIABLE PYTHON_USER_INSTALL_SITE)
STRING(REPLACE "\n" "" PYTHON_USER_INSTALL_SITE ${PYTHON_USER_INSTALL_SITE})

################################# Finding the right version of boost-python #################

set(BOOST_PYTHON_LIBRARY_SUFFIX 37)

# There is this name difference between Apple and Linux :(
if(${APPLE})
    set(BOOST_PYTHON_COMPONENT_NAME python${BOOST_PYTHON_LIBRARY_SUFFIX})
    set(BOOST_PYTHON_NUMPY_COMPONENT_NAME numpy${BOOST_PYTHON_LIBRARY_SUFFIX})
else()
    set(BOOST_PYTHON_COMPONENT_NAME python-py${BOOST_PYTHON_LIBRARY_SUFFIX})
    set(BOOST_PYTHON_NUMPY_COMPONENT_NAME numpy3-py${BOOST_PYTHON_LIBRARY_SUFFIX})
endif()

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/usr/local/Cellar/boost-python3/1.71.0/lib")
find_package(
        Boost
    COMPONENTS
        log
        log_setup
        thread
        ${BOOST_PYTHON_COMPONENT_NAME}
    REQUIRED
        ${BOOST_PYTHON_NUMPY_COMPONENT_NAME}
)

##################### Global options, an include_directories, so sue me ! ###################

add_compile_options(-g)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
)

################################## TARGET spooki_run.out ####################################

add_executable(spooki_run.out main.cpp)
target_link_libraries(spooki_run.out PRIVATE dl)

# NOTE:
#     simply linking to the boost-python library without also linking to the
#     python library will cause a runtime loading error.
#
# target_link_libraries(spooki_run.out ${Boost_LIBRARIES})
#
#     but also linking to the python library will fix this
#
# target_link_libraries(spooki_run.out ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#
#     allthough spoorki_run.out doesn't use the library.


# Or we could do this in plugins/AbsoluteValue/CMakeLists
add_dependencies(spooki_run.out AbsoluteValue)

################################## SUBDIRECTORIES ############################################

add_subdirectory(meteo)
add_subdirectory(pymeteo)
add_custom_target(check)

################################## CONFIGURE_FILE ############################################

# We're going to hard-code the path to plugins directly
# into the C++ code.  This value is going to go in an .h file
set(CMAKE_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)
configure_file(cmake_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config.out.h)


