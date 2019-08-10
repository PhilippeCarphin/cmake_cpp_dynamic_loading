//
// Created by Philippe Carphin on 2019-08-10.
//

//
//
// Created by afsmpca on 19-07-09.
//

#include <iostream>
#include <boost/python.hpp>
#include <Python.h>

#include "absval.h"
#include "cmake_config.out.h"

#include <dlfcn.h>
#include <cstdlib>
#include <meteo_operations/OperationBase.h>

void internal_initializations(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    std::cerr << "C++      : [info] This interface was compiled for loading by Python " << BOOST_PYTHON_VERSION << std::endl;
    std::cerr << "C++      : [info] This test is development on SPOOKI" << std::endl;
    std::cerr << "C++      : [info] This was compiled with PLUGIN_PATH=" << PLUGIN_PATH << std::endl;
    // std::cerr << "C++      : "; BOOST_LOG_INFO << "Testing boost log info";
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
    plugin_maker_t *absolute_value_maker = dlsymAs<plugin_maker_t *>(maker);
    //plugin_maker_t *absolute_value_maker = reinterpret_cast<plugin_maker_t *>(maker);

    OperationBase *absolute_value_instance_ptr = absolute_value_maker();

    absolute_value_instance_ptr->algo();
}

using namespace boost::python;
BOOST_PYTHON_MODULE(absval)
{
    def("run_absolute_value_plugin", run_absolute_value_plugin);
    internal_initializations();
}

