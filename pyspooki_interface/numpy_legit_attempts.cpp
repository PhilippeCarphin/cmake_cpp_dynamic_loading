//
// Created by Philippe Carphin on 2019-08-10.
//

#include "numpy_legit_attempts.h"
#include "cmake_config.out.h"
#include <boost/python.hpp>

#include <numpy/arrayobject.h>
// #include <numpy/ndarraytypes.h>

#include <iostream>

boost::python::object get_numpy_array_owning_data()
{
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    int nd = 4;
    npy_intp npy_dims[] = {20,30,40,50};
    int * data_ptr = (int*)(malloc(20 * 30 * 40 * 50 * sizeof(int)));
    for(int i = 1; i <= 20*30*40*50; ++i){
        data_ptr[i-1] = i;
    }
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "   Attempting to call PyArray_SimpleNewFromData(" << std::endl
              << "          " << "nd=" << nd << std::endl
              << "          " << "npy_dims=" << npy_dims << std::endl
              << "          " << "NPY_INT32=" << NPY_INT32 << std::endl
              << "          " << "data_ptr=" << data_ptr << std::endl
              << "          " << "&data_ptr=" << &data_ptr << std::endl
              << std::endl;

    PyObject *array = PyArray_SimpleNewFromData(nd, npy_dims, NPY_INT32, (void*)(data_ptr));
    PyArray_ENABLEFLAGS((PyArrayObject *)array, NPY_OWNDATA);

    std::cout << "C++      : " << __PRETTY_FUNCTION__ << " [[ END ]] " << std::endl;
    return boost::python::object(boost::python::handle<>(array));
}

// #define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return NUMPY_IMPORT_ARRAY_RETVAL; } }
int import_array_wrapper(){
    // import_array(); (a macro for doing this:
    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }
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
    import_array_wrapper();
    def("get_numpy_array_owning_data", get_numpy_array_owning_data);
    internal_initializations();
}