//
// Created by Philippe Carphin on 2019-08-10.
//

#include <boost/python.hpp>
#include <iostream>
#include <boost/python/numpy.hpp>

typedef long int my_data_type;

boost::python::numpy::ndarray get_array_that_owns_through_default_object()
{
    // Change this to see how the adresses change.
    unsigned int last_dim = 6000;
    boost::python::object shape = boost::python::make_tuple(4, 5, last_dim);

    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<my_data_type>();

    auto * const data_ptr = new my_data_type[4*5*last_dim];

    const size_t s = sizeof(my_data_type);
    boost::python::object strides = boost::python::make_tuple(5*last_dim*s, last_dim*s, s);

    for(int i = 1; i <= 4*5*last_dim; ++i){ data_ptr[i-1] = i; }

    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "data_ptr = " << data_ptr << std::endl;

    return boost::python::numpy::from_data( data_ptr, dt, shape, strides, boost::python::object());
}


using namespace boost::python;
BOOST_PYTHON_MODULE(THIS_PYTHON_MODULE_NAME)
{
    boost::python::numpy::initialize();
    def("get_array_that_owns_through_default_object", get_array_that_owns_through_default_object);
}