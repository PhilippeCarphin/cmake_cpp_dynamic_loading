//
// Created by Philippe Carphin on 2019-08-10.
//
#include "interface_classes.h"
#include "cmake_config.out.h"
#include <iostream>
#include <boost/python.hpp>

InterfaceClass::InterfaceClass(std::string name) :name(name)
{
    std::cout << "C++      : " <<  __PRETTY_FUNCTION__ << "[" << name << "]" << std::endl;
}


InterfaceClass::~InterfaceClass()
{
    std::cout << "C++      : " <<  __PRETTY_FUNCTION__ << "[" << name << "]" << std::endl;
}

void InterfaceClass::method()
{
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}


std::shared_ptr<InterfaceClass> returning_shared_ptr_test(std::string name){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "[" << name << "]" << std::endl;
    return std::shared_ptr<InterfaceClass>(new InterfaceClass(name));
}

std::shared_ptr<InterfaceClass> copy(std::shared_ptr<InterfaceClass> the_ptr){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << the_ptr << std::endl;
    return the_ptr;
}



void interface_function(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}


const double e = 2.7182818284590452353602874713527;
double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

/*
 * Here Boost is just going to see a function that takes a PyObject* and
 * returns a PyObject* so it will do none of it's magic.  Therefore, even
 * though we are exposing this function through boost, this is what you
 * would do without boost.
 *
 * Also, the magic error handling stuff that boost does won't work as well
 * but we do get more than if we were doing a pure python extension without
 * boost at all.
 *
 * >>> absval.tanh_impl('not a float')
 * C++      : PyObject *tanh_impl(PyObject *)
 * TypeError: must be real number, not str
 *
 * The above exception was the direct cause of the following exception:
 *
 * Traceback (most recent call last):
 *   File "<stdin>", line 1, in <module>
 * SystemError: <Boost.Python.function object at 0x7fbf49500e30> returned a result with an error set
 */
PyObject* tanh_impl(PyObject *x) {
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    double x_val = PyFloat_AsDouble(x);
    double tanh_x = sinh_impl(x_val) / cosh_impl(x_val);
    return PyFloat_FromDouble(tanh_x);
}

/*
 * >>> absval.tanh_impl_better('not a float')
 * Traceback (most recent call last):
 *   File "<stdin>", line 1, in <module>
 * Boost.Python.ArgumentError: Python argument types in
 *     absval.tanh_impl_better(str)
 * did not match C++ signature:
 *     tanh_impl_better(float)
 */
double tanh_impl_better(float x) {
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    return sinh_impl(x) / cosh_impl(x);
}

void internal_initializations(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    std::cerr << "C++      : [info] This interface was compiled for loading by Python " << BOOST_PYTHON_VERSION << std::endl;
    std::cerr << "C++      : [info] This test is development on SPOOKI" << std::endl;
    std::cerr << "C++      : [info] This was compiled with PLUGIN_PATH=" << PLUGIN_PATH << std::endl;
    // std::cerr << "C++      : "; BOOST_LOG_INFO << "Testing boost log info";
}

using namespace boost::python;
BOOST_PYTHON_MODULE(THIS_PYTHON_MODULE_NAME)
{
        class_<std::shared_ptr<InterfaceClass>>("InterfaceClass_shared_ptr")
                .def("sh_ptr_use_count", &std::shared_ptr<InterfaceClass>::use_count);

        class_<InterfaceClass>("InterfaceClass", init<std::string>()).def("method", &InterfaceClass::method);
        def("copy", copy);
        def("pyspooki_interface_function", interface_function);
        def("returning_shared_ptr_test", returning_shared_ptr_test);
        def("tanh_impl", tanh_impl);
        def("tanh_impl_better", tanh_impl_better);
        internal_initializations();
}

// TODO: Create a function that returns a python object that I create by hand in the CPP world
