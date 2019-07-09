
add_library(pyspooki_interface SHARED pyspooki_interface.cpp)


FIND_PACKAGE(Boost COMPONENTS python-py36)
message("Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")

FIND_PACKAGE(PythonLibs 3.6)

message("PythonInterp_INCLUDE_DIRS=${Python_INCLUDE_DIRS}")
message("PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}")
# message("PYTHON_LIBRARIES=${PYTHON_LIBRARIES}")

target_include_directories(
        pyspooki_interface
    PUBLIC
        ${PYTHON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

target_link_libraries(
        pyspooki_interface
    PUBLIC
        meteo_operations
        ${Boost_LIBRARIES}
        ${Python_LIBRARIES}
)


