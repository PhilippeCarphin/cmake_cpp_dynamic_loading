add_library(spooki_logging SHARED spooki_logging.cpp)

target_include_directories(spooki_logging PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(spooki_logging ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_link_libraries(spooki_logging INTERFACE ${Boost_LIBRARIES})
target_compile_definitions(spooki_logging PUBLIC "BOOST_LOG_DYN_LINK")
# set_target_properties(spooki_logging PROPERTIES COMPILE_DEFINITIONS "BOOST_LOG_DYN_LINK")