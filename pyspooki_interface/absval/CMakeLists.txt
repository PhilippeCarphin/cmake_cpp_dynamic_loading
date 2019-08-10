
ADD_BOOST_PYTHON_MODULE(absval absval.cpp)
ADD_DEPENDENCIES(absval AbsoluteValue)
ADD_DEPENDENCIES(check-py check-absval)
