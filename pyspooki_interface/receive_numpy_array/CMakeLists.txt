
ADD_BOOST_PYTHON_MODULE(receive_numpy_array receive_numpy_array.cpp)
target_link_libraries(receive_numpy_array PRIVATE Boost::${BOOST_PYTHON_NUMPY_COMPONENT_NAME})
add_dependencies(check-py check-receive_numpy_array)

