
add_library(AbsoluteValue MODULE AbsoluteValue.cpp)

target_link_libraries(
        AbsoluteValue
    PRIVATE
        meteo_operations
)
