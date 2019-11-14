//
// Created by Philippe Carphin on 2019-08-10.
//
#include <stdlib.h>

#include <boost/python.hpp>
#include <iostream>
#include <boost/python/numpy.hpp>

typedef long int my_data_type;
inline void destroyManagerCObject(PyObject* self) {
    auto * b = reinterpret_cast<my_data_type*>( PyCapsule_GetPointer(self, NULL) );
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << " free(" << b << ")" << std::endl;
    // Clang-tidy : No need for if; deleting NULL has no effect
    delete [] b;
}

boost::python::numpy::ndarray get_array_that_owns_through_capsule()
{
    // Change this to see how the adresses change.
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "Before calling boost::python::numpy::initialize()" << std::endl;
    unsigned int last_dim = 6000;
    boost::python::object shape = boost::python::make_tuple(4, 5, last_dim);

    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<my_data_type>();

    my_data_type * const data_ptr = (my_data_type *)malloc(4*5*last_dim * sizeof(*data_ptr));

    const size_t s = sizeof(my_data_type);
    boost::python::object strides = boost::python::make_tuple(5*last_dim*s, last_dim*s, s);

    for(int i = 1; i <= 4*5*last_dim; ++i){ data_ptr[i-1] = i; }

    // This sets up a python object whose destructio will free data_ptr
    PyObject *capsule = ::PyCapsule_New((void *)data_ptr, NULL, (PyCapsule_Destructor)&destroyManagerCObject);
    boost::python::handle<> h_capsule{capsule};
    boost::python::object owner_capsule{h_capsule};

    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "data_ptr = " << data_ptr << std::endl;

    return boost::python::numpy::from_data( data_ptr, dt, shape, strides, owner_capsule);
}

using namespace boost::python;
BOOST_PYTHON_MODULE(THIS_PYTHON_MODULE_NAME)
{
    Py_Initialize();
    boost::python::numpy::initialize();
    def("get_array_that_owns_through_capsule", get_array_that_owns_through_capsule);
}