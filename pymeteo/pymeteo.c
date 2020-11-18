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

#include "pymeteo.h"

//#include <dlfcn.h>
#include <cstdlib>
#include "meteo.h"

void internal_initializations(){
    printf("C++      : %s\n", __PRETTY_FUNCTION__);
}

void run_absolute_value_plugin(){

    printf("C++      : %s\n", __PRETTY_FUNCTION__);

    OperationBase *absolute_value_instance_ptr = absolute_value_maker();

    absolute_value_instance_ptr->algo();
}

using namespace boost::python;
BOOST_PYTHON_MODULE(absval)
{
    def("run_absolute_value_plugin", run_absolute_value_plugin);
    internal_initializations();
}

