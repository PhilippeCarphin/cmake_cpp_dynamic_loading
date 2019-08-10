add_library(spooki_logging SHARED spooki_logging.cpp)

target_include_directories(spooki_logging PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(spooki_logging
    PUBLIC
        Boost::log
)
target_compile_definitions(spooki_logging PUBLIC "BOOST_LOG_DYN_LINK")


add_executable(test_spooki_logging test_spooki_logging.cpp)
target_link_libraries(test_spooki_logging spooki_logging)
add_custom_target(check-spooki_logging COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_spooki_logging)
add_dependencies(check-spooki_logging test_spooki_logging)
