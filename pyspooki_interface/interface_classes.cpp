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
        internal_initializations();
}

// TODO: Create a function that returns a python object that I create by hand in the CPP world
