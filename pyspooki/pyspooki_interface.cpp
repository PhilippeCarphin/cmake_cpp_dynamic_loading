//
// Created by afsmpca on 19-07-09.
//

#include <iostream>
#include <boost/python.hpp>

#include "pyspooki_interface.h"

pyspooki_interface_class::pyspooki_interface_class(){
    std::cerr << __PRETTY_FUNCTION__ << std::endl;
}

void pyspooki_interface_class::method(){
    std::cerr << __PRETTY_FUNCTION__ << std::endl;
}

void pyspooki_interface_function(){
    std::cerr << __PRETTY_FUNCTION__ << std::endl;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(libpyspooki_interface)
{
    class_<pyspooki_interface_class>("pyspooki_interface_class", init<>())
            .def("method", &pyspooki_interface_class::method);
    def("pyspooki_interface_function", pyspooki_interface_function);


}

// TODO: Create a function that returns a python object that I create by hand in the CPP world
