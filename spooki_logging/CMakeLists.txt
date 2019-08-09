add_library(spooki_logging SHARED spooki_logging.cpp)

target_include_directories(spooki_logging PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(spooki_logging PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#target_compile_features(spooki_logging PUBLIC cxx_std_11)
target_link_libraries(spooki_logging PUBLIC
    ${Boost_LOG_SETUP_LIBRARY}
    ${Boost_LOG_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
)

# target_compile_options(spooki_logging PUBLIC "-std=c++11")
target_compile_definitions(spooki_logging PUBLIC "-DBOOST_LOG_DYN_LINK")

