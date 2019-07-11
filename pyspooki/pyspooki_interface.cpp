//
// Created by afsmpca on 19-07-09.
//

#include <iostream>
#include <boost/python.hpp>

#include "pyspooki_interface.h"
#include "cmake_config.out.h"

pyspooki_interface_class::pyspooki_interface_class(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}

void pyspooki_interface_class::method(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}

void pyspooki_interface_function(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}

void internal_initializations(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    std::cerr << "C++      : [info] This interface was compiled for loading by Python " << BOOST_PYTHON_VERSION << std::endl;
    std::cerr << "C++      : [info] This test is development on SPOOKI" << std::endl;
}

const double e = 2.7182818284590452353602874713527;
double sinh_impl(double x) {
    return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
    return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

PyObject* tanh_impl(float x) {
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    double tanh_x = sinh_impl(x) / cosh_impl(x);
    return PyFloat_FromDouble(tanh_x);
}

using namespace boost::python;
BOOST_PYTHON_MODULE(libpyspooki_interface)
{
    class_<pyspooki_interface_class>("pyspooki_interface_class", init<>())
            .def("method", &pyspooki_interface_class::method);
    def("pyspooki_interface_function", pyspooki_interface_function);
    def("tanh_impl", tanh_impl);
    internal_initializations();
}

// TODO: Create a function that returns a python object that I create by hand in the CPP world
