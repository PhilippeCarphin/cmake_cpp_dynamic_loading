add_library(spooki_logging SHARED spooki_logging.cpp)

target_include_directories(spooki_logging PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(spooki_logging PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_link_libraries(spooki_logging INTERFACE ${Boost_LIBRARIES})

if(${CMAKE_MAJOR_VERSION} LESS 3)
    target_compile_options(spooki_logging PUBLIC "-std=c++11")
endif()

if(${CMAKE_MAJOR_VERSION} LESS 3)
    target_compile_definitions(spooki_logging PUBLIC "BOOST_LOG_DYN_LINK")
else()
    set_target_properties(spooki_logging PROPERTIES COMPILE_DEFINITIONS "BOOST_LOG_DYN_LINK")
endif()

