cmake_minimum_required (VERSION 2.8)
project (Tutorial CXX)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


################################# Finding python ############################################

if("$ENV{ORDENV_PLAT}" STREQUAL "ubuntu-14.04-amd64-64")
    set(ON_UBUNTU_14_PPP TRUE)
endif()

if(ON_UBUNTU_14_PPP)
    # if("$ENV{INTELCOMP_HOME}" STREQUAL "")
        # MESSAGE(FATAL_ERROR "ON UBUNTU 14, On PPP1, finding PythonLibs requires intelcomp environment")
    # endif()
    MESSAGE(STATUS "ON UBUNTU 14, ...")
    if("$ENV{CXX}" STREQUAL "/usr/bin/g++" OR "$ENV{CXX}" STREQUAL "/usr/bin/g++-4.8")
       MESSAGE(STATUS "--> Environment variable CXX has an acceptable value for PPP1/2")
    else()
       MESSAGE(STATUS "--> POSSIBLE ERROR: On PPP1/2, environment variable CXX must be /usr/bin/g++ or /usr/bin/g++-4.8")
    endif()
    if(${CMAKE_MAJOR_VERSION} LESS 3)
       SET(Python_ADDITIONAL_VERSIONS 3.4)
       MESSAGE(STATUS "--> On Ubuntu 14 with CMake 2, setting Python_ADDITIONAL_VERSIONS to 3.4")
    else()
       MESSAGE(STATUS "--> On Ubuntu 14 with CMake 3, no need for Python_ADDITIONAL_VERSION")
    endif()
else()
   MESSAGE("Not on Ubuntu 14, no special settings necessary to find PythonLibs")
endif()

FIND_PACKAGE(
    PythonLibs
        3
        COMPONENTS
)

if(${APPLE})
    # Because Numpy header files are in a weird place on Apple
    # /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/numpy/core/include
    # which contains a directory called numpy which has the headers.
    # On Ubuntu 18, we have it in the regular python include directory:
    # /usr/include/python3.6m contains a directory (a link) called numpy, it points to the numpy directory
    # in the .../numpy/core/include
    FIND_PACKAGE(Python3 COMPONENTS NumPy)
    message(STATUS "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3_NumPy_INCLUDE_DIRS=${Python3_NumPy_INCLUDE_DIRS}")
endif()

if(PYTHONLIBS_FOUND)
    message(STATUS "Found PythonLibs version ${PYTHONLIBS_VERSION_STRING} at path ${PYTHON_LIBRARIES}")
 else()
   if("$ENV{ORDENV_PLAT}" STREQUAL "")
      message(STATUS "POSSIBLE ERROR: Environment variable ORDENV_PLAT is not defined, it is
      necessary to finding PythonLibs if we are on PPP1/2")
      message(FATAL_ERROR "PythonLibs not found If, on PPP1, note
      that ordenv must be loaded and CXX environment variable
      must be either /usr/bin/g++ or /usr/bin/g++-4.8, Also, you
      may have to clear CMake's cache after correcting these
      issues.")
   endif()
endif()

################################# Finding the right version of boost-python #################

# Change PYTHONLIBS_VERSION_STRING=3.4.5 to PYTHON_FOUND_VERSION=3
string(REGEX MATCH "3.[1-9]" BOOST_PYTHON_LIBRARY_SUFFIX ${PYTHONLIBS_VERSION_STRING})
string(REPLACE "." "" BOOST_PYTHON_LIBRARY_SUFFIX ${BOOST_PYTHON_LIBRARY_SUFFIX})

# There is this name difference between Apple and Linux :(
if(${APPLE})
    set(BOOST_PYTHON_COMPONENT_NAME python${BOOST_PYTHON_LIBRARY_SUFFIX})
    set(BOOST_PYTHON_NUMPY_COMPONENT_NAME numpy${BOOST_PYTHON_LIBRARY_SUFFIX})
else()
    set(BOOST_PYTHON_COMPONENT_NAME python-py${BOOST_PYTHON_LIBRARY_SUFFIX})
    set(BOOST_PYTHON_NUMPY_COMPONENT_NAME numpy3-py${BOOST_PYTHON_LIBRARY_SUFFIX})
endif()

find_package(
        Boost
    COMPONENTS
        log
        ${BOOST_PYTHON_COMPONENT_NAME}
        ${BOOST_PYTHON_NUMPY_COMPONENT_NAME}
)
if(${APPLE})
    if(${Boost_PYTHON37_FOUND})
        message(STATUS "PYTHON37 FOUND!!!!!")
        set(Boost_PYTHON_LIBRARIES ${Boost_PYTHON37_LIBRARIES})
        set(Boost_NUMPY_LIBRARIES ${Boost_NUMPY37_LIBRARIES})
    elseif(${Boost_PYTHON_PY37_FOUND})
        message(STATUS "PYTHON_PY37 FOUND !!!!!!!!!!!!!!!!")
        set(Boost_PYTHON_LIBRARIES ${Boost_PYTHON_PY37_LIBRARIES})
        set(Boost_NUMPY_LIBRARIES ${Boost_NUMPY3_PY37_LIBRARIES})
    endif()
else()
    if(${Boost_PYTHON-PY36_FOUND})
        message(STATUS "Boost_PYTHON-PY36_FOUND !!!!!!!!!!!!")
    endif()
    if(${Boost_NUMPY3-PY36_FOUND})
        message(STATUS "Boost_NUMPY3-PY36_FOUND !!!!!!!!!!!!!")
    endif()
endif()

message(STATUS "------- PYTHON ")
message(STATUS "Boost_PYTHON37_LIBRARIES = ${Boost_PYTHON37_LIBRARIES}")
message(STATUS "Boost_PYTHON37_LIBRARY = ${Boost_PYTHON37_LIBRARY}")
message(STATUS "Boost_PYTHON36_LIBRARIES = ${Boost_PYTHON37_LIBRARIES}")
message(STATUS "Boost_PYTHON36_LIBRARY = ${Boost_PYTHON37_LIBRARY}")
message(STATUS "Boost_PYTHON-PY36_LIBRARIES = ${Boost_PYTHON37_LIBRARIES}")
message(STATUS "Boost_PYTHON-PY36_LIBRARY = ${Boost_PYTHON37_LIBRARY}")
message(STATUS "Boost_PYTHON_PY36_LIBRARIES = ${Boost_PYTHON37_LIBRARIES}")
message(STATUS "Boost_PYTHON_PY36_LIBRARY = ${Boost_PYTHON37_LIBRARY}")
message(STATUS "Boost_PYTHON_LIBRARIES = ${Boost_PYTHON37_LIBRARIES}")
message(STATUS "Boost_PYTHON_LIBRARY = ${Boost_PYTHON37_LIBRARY}")
message(STATUS "Boost_PYTHON_LIBRARIES = ${Boost_PYTHON37_LIBRARIES}")
message(STATUS "Boost_PYTHON_LIBRARY = ${Boost_PYTHON37_LIBRARY}")

message(STATUS "------- ")
message(STATUS "Boost_NUMPY37_LIBRARIES = ${Boost_NUMPY37_LIBRARIES}")
message(STATUS "Boost_NUMPY37_LIBRARY = ${Boost_NUMPY37_LIBRARY}")

message(STATUS "Boost_LOG_LIBRARIES = ${Boost_LOG_LIBRARIES}")
message(STATUS "Boost_LOG_LIBRARY = ${Boost_LOG_LIBRARY}")

message(STATUS "Boost_LOG_SETUP_LIBRARIES = ${Boost_LOG_SETUP_LIBRARIES}")
message(STATUS "Boost_LOG_SETUP_LIBRARY = ${Boost_LOG_SETUP_LIBRARY}")

message(STATUS "Boost_THREAD_LIBRARIES = ${Boost_THREAD_LIBRARIES}")
message(STATUS "Boost_THREAD_LIBRARY = ${Boost_THREAD_LIBRARY}")

message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
# message(STATUS "Boost_PYTHON37_INCLUDE_DIRS = ${Boost_PYTHON37_INCLUDE_DIRS}")
# message(STATUS "Boost_NUMPY37_INCLUDE_DIRS = ${Boost_NUMPY37_INCLUDE_DIRS}")
# message(STATUS "Boost_PYTHON37_INCLUDE_DIR = ${Boost_PYTHON37_INCLUDE_DIR}")
# message(STATUS "Boost_NUMPY37_INCLUDE_DIR = ${Boost_NUMPY37_INCLUDE_DIR}")
message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")

# message(FATAL_ERROR "Stopping early")
if(ON_UBUNTU_14_PPP)
   MESSAGE(STATUS "PPP1 and PPP2 do not have numpy,
   you won't be able to build anything that uses it, and anything that uses it
   will likely cause a segfault when running")
endif()

################################## An include_directories ... so sue me #####################

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

add_subdirectory(meteo_operations)
add_subdirectory(plugins)
add_subdirectory(pyspooki_interface)
add_subdirectory(spooki_logging)

################################## TARGET CHECK ##############################################

add_custom_target(check)

################################## CONFIGURE_FILE ############################################

# We're going to hard-code the path to plugins directly
# into the C++ code.  This value is going to go in an .h file
set(CMAKE_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)
configure_file(cmake_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config.out.h)


