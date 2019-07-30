//
// Created by afsmpca on 19-07-09.
//

#include <iostream>
#include <boost/python.hpp>

#include "pyspooki_interface.h"
#include "cmake_config.out.h"

#include <dlfcn.h>
#include <meteo_operations/OperationBase.h>
// #include "spooki_logging/spooki_logging.hpp"

#ifdef USE_BOOST_NUMPY
#include <boost/python/numpy.hpp>
#endif

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

class TestObject {
public:
    TestObject(std::string name):name(name){std::cout << "C++      : " <<  __PRETTY_FUNCTION__ << "[" << name << "]" << std::endl;}
    void method(){std::cout << "C++      : " << __PRETTY_FUNCTION__ << "[" << name << "]" << std::endl;}
    ~TestObject(){std::cout << "C++      : " << __PRETTY_FUNCTION__ << "[" << name << "]" << std::endl;}
    std::string name;
};

std::shared_ptr<TestObject> returning_shared_ptr_test(std::string name){
std::cout << "C++      : " << __PRETTY_FUNCTION__ << "[" << name << "]" << std::endl;
    return std::shared_ptr<TestObject>(new TestObject(name));
}

std::shared_ptr<TestObject> copy(std::shared_ptr<TestObject> the_ptr){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << the_ptr << std::endl;
    return the_ptr;
}

#ifdef USE_BOOST_NUMPY
std::string bnp_array_to_string(boost::python::numpy::ndarray const &a){
    return std::string(boost::python::extract<char const*>(boost::python::str(a)));
}

std::string bnp_array_to_shape_string(boost::python::numpy::ndarray const &a){
    std::ostringstream oss;
    oss << "(";
    int nd = a.get_nd();
    const Py_intptr_t * iptr = a.get_shape();
    int dim = 0;
    for(;dim < nd-1; dim++){
        oss << iptr[dim] << ", ";
    }
    oss << iptr[nd-1] << ")" << std::endl;
    return oss.str();
}

void massage_numpy_array(boost::python::numpy::ndarray const &a){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    std::cout << "C++      : " << " printing str(a) ..." << std::endl << bnp_array_to_string(a) << std::endl;
    std::cout << "C++      : " << " SHAPE IS : (" << a.shape(0) << ", " << a.shape(1) << ", " << a.shape(2) << ")" << std::endl;
#ifndef __APPLE__
    // On My imac, this raises an exception that is caught as
    // an IndexError in python.  There are only three dimensions
    // to this array.
    std::cout << "C++      : " << " a.shape(3) : " << a.shape(8) << std::endl;
    // on kano, this gave 48 which is a.strides(0) and this is
    // probably due to memory layout and is implementation defined.
#endif
    const Py_intptr_t * iptr = a.get_shape();
    std::cout << "C++      : " << "shape of a : " << bnp_array_to_shape_string(a) << std::endl;
    std::cout << "C++      : " << "strides of a : [" << a.strides(0) << ", " << a.strides(1) << ", " << a.strides(2) << "]" << std::endl;
    std::cout << "C++      : " << "sizeof(int) = " << sizeof(int) << std::endl;
    unsigned long int *data = reinterpret_cast<unsigned long int *>(a.get_data());
    for(int i = 0; i < 1*2*3; i++){
        data[i] = i;
    }
    std::cout << "C++      : " << " printing str(a) ..." << std::endl << bnp_array_to_string(a) << std::endl;
}
#else
void massage_numpy_array(PyObject *a){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}
#endif


using namespace boost::python;
BOOST_PYTHON_MODULE(pyspooki_interface)
{
    // Py_Initialize();
#ifdef USE_BOOST_NUMPY
    boost::python::numpy::initialize();
#endif
    class_<std::shared_ptr<TestObject>>("TestObject_shared_ptr").def("ref_count", &std::shared_ptr<TestObject>::use_count);
    class_<TestObject>("TestObject", init<std::string>()).def("method", &TestObject::method);
    class_<pyspooki_interface_class>("pyspooki_interface_class", init<>())
            .def("method", &pyspooki_interface_class::method);
    def("copy", copy);
    def("pyspooki_interface_function", pyspooki_interface_function);
    def("tanh_impl", tanh_impl);
    def("run_absolute_value_plugin", run_absolute_value_plugin);
    def("returning_shared_ptr_test", returning_shared_ptr_test);
    def("massage_numpy_array", massage_numpy_array);
    internal_initializations();
}

// TODO: Create a function that returns a python object that I create by hand in the CPP world
