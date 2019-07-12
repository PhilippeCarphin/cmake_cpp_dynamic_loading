cmake_minimum_required (VERSION 2.8)
project (Tutorial CXX)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)


################################# Finding python ############################################

if("$ENV{ORDENV_PLAT}" STREQUAL "ubuntu-14.04-amd64-64")
    set(SPECIAL_CASE_PYTHON_34 TRUE)
endif()

if(SPECIAL_CASE_PYTHON_34)
    SET(Python_ADDITIONAL_VERSIONS 3.4)
    SET(Python_DESIRED_VERSION 3.4)
    SET(PYTHON_FOUND_VERSION 34)
else()
    SET(Python_DESIRED_VERSION 3)
endif()


FIND_PACKAGE(
    PythonLibs
        ${Python_DESIRED_VERSION}
    REQUIRED
)

if(PYTHONLIBS_FOUND)
    message(STATUS "Found PythonLibs version ${PYTHONLIBS_VERSION_STRING}")
endif()

################################# Finding the right version of boost-python #################

# Change PYTHONLIBS_VERSION_STRING=3.4.5 to PYTHON_FOUND_VERSION=3
string(REGEX MATCH "3.[1-9]" BOOST_PYTHON_LIBRARY_SUFFIX ${PYTHONLIBS_VERSION_STRING})
string(REPLACE "." "" BOOST_PYTHON_LIBRARY_SUFFIX ${BOOST_PYTHON_LIBRARY_SUFFIX})

# There is this name difference between Apple and Linux :(
if(${APPLE})
    set(BOOST_PYTHON_LIBRARY_NAME python${BOOST_PYTHON_LIBRARY_SUFFIX})
else()
    set(BOOST_PYTHON_LIBRARY_NAME python-py${BOOST_PYTHON_LIBRARY_SUFFIX})
endif()

find_package(
        Boost
    COMPONENTS
        ${BOOST_PYTHON_LIBRARY_NAME}
    REQUIRED
)

################################## An include_directories ... so sue me #####################

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
)

################################## TARGET spooki_run.out ####################################

add_executable(spooki_run.out main.cpp)
target_link_libraries(spooki_run.out dl)

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

add_subdirectory(meteo_operations)
add_subdirectory(plugins)
add_subdirectory(pyspooki)

################################## TARGET CHECK ##############################################

add_custom_target(check)
add_dependencies(check
    python_interface_test
    pyspooki_interface
)

################################## CONFIGURE_FILE ############################################

# We're going to hard-code the path to plugins directly
# into the C++ code.  This value is going to go in an .h file
set(CMAKE_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)
configure_file(cmake_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config.out.h)


