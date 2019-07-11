//
// Created by afsmpca on 19-07-09.
//

#include <iostream>
#include <boost/python.hpp>

#include "pyspooki_interface.h"
#include "cmake_config.out.h"

#include <dlfcn.h>
#include <meteo_operations/OperationBase.h>

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

void run_absolute_value_plugin(){

    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;

    std::string absolute_value_path;

    absolute_value_path += std::string(PLUGIN_PATH) + "/AbsoluteValue/libAbsoluteValue.so";

    void *plugin = dlopen(absolute_value_path.c_str(), RTLD_NOW);

    if(plugin){
        std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : Plugin loaded successfully" << std::endl;
    } else {
        std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : ERROR loading plugin : " << dlerror() << std::endl;
        return;
    }

    void *maker = dlsym(plugin, "maker");

    if(maker){
        std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : Symbol loaded successfully" << std::endl;
    } else {
        std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : ERROR loading symbol: " << dlerror() << std::endl;
        return;
    }

    /*
     * We have a pointer to the symbol but only the programmer knows
     * what that symbol is.
     */
    typedef OperationBase *plugin_maker_t();
    plugin_maker_t *absolute_value_maker = reinterpret_cast<plugin_maker_t *>(maker);

    OperationBase *absolute_value_instance_ptr = absolute_value_maker();

    absolute_value_instance_ptr->algo();
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
BOOST_PYTHON_MODULE(pyspooki_interface)
{
    class_<pyspooki_interface_class>("pyspooki_interface_class", init<>())
            .def("method", &pyspooki_interface_class::method);
    def("pyspooki_interface_function", pyspooki_interface_function);
    def("tanh_impl", tanh_impl);
    def("run_absolute_value_plugin", run_absolute_value_plugin);
    internal_initializations();
}

// TODO: Create a function that returns a python object that I create by hand in the CPP world
