add_library(spooki_logging SHARED spooki_logging.cpp)

target_include_directories(spooki_logging PRIVATE ${Boost_INCLUDE_DIRS})


# log, logsetup and thread don't get added to Boost_LIBRARIES on Apple.
# I should just always explicitly link with the boost libraries that I want
# It's better this way
target_link_libraries(spooki_logging
    PRIVATE
        Boost::log
)

if(${CMAKE_MAJOR_VERSION} LESS 3)
    target_compile_options(spooki_logging PUBLIC "-std=c++11")
endif()

if(${CMAKE_MAJOR_VERSION} LESS 3)
    target_compile_definitions(spooki_logging PUBLIC "BOOST_LOG_DYN_LINK")
else()
    set_target_properties(spooki_logging PROPERTIES COMPILE_DEFINITIONS "BOOST_LOG_DYN_LINK")
endif()

