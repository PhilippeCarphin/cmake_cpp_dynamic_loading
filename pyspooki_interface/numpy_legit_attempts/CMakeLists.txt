ADD_BOOST_PYTHON_MODULE(numpy_legit_attempts numpy_legit_attempts.cpp)
target_include_directories(numpy_legit_attempts PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
ADD_DEPENDENCIES(check-py check-numpy_legit_attempts)
