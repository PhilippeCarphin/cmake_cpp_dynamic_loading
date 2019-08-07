//
// Created by afsmpca on 19-07-09.
//

#include <iostream>
#include <boost/python.hpp>
#include <python3.6m/numpy/arrayobject.h>


#include "pyspooki_interface.h"
#include "cmake_config.out.h"

#include <dlfcn.h>
#include <cstdlib>
#include <meteo_operations/OperationBase.h>
// #include "spooki_logging/spooki_logging.hpp"

#ifdef USE_BOOST_NUMPY
#include <boost/python/numpy.hpp>
#endif


pyspooki_interface_class::pyspooki_interface_class(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}

void pyspooki_interface_class::method(){std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
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

static int *g_int_ptr = nullptr;
void delete_g_int_ptr(){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "delete [] g_int_ptr" << std::endl;
    delete [] g_int_ptr;
}
void free_g_int_ptr(){
    free(g_int_ptr);
}

void print_g_int_ptr(int i){

    if(!g_int_ptr)
        return;
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "g_int_ptr[" << i << "] = " << g_int_ptr[i] << std::endl;
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "g_int_ptr = " << g_int_ptr << std::endl;
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << "&g_int_ptr = " << &g_int_ptr << std::endl;
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
    // VALGRIND does spot this on Linux, it just doesn't cause a crash
    // std::cout << "C++      : " << " a.shape(3) : " << a.shape(8) << std::endl;
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


    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<float>();
    std::cout << "C++      : " << " dtype::get_builtin<float>().get_itemsize() = " << dt.get_itemsize() << std::endl;
}
boost::python::numpy::ndarray cook_up_a_numpy_array()
{
    boost::python::object shape = boost::python::make_tuple(10, 20, 30, 40);
    boost::python::object strides = boost::python::make_tuple(192000, 9600, 320, 8);

    boost::python::object own;
    static int static_dat[10*20*30] = {11,22,33,44,55,66};

    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<int>();
    int * data_ptr = static_cast<int*>(malloc(10 * 20 * 30 * 40 * sizeof(int)));

    for(int i = 1; i <= 10*20*30; ++i){
        data_ptr[i-1] = i;
        // static_dat[i-1] = i;
    }

    // std::cout << "C++      : " << __PRETTY_FUNCTION__ << "own.is_none() : " << own.is_none() << std::endl;
    // PyArray_ENABLEFLAGS(boost::python::extract<PyArrayObject*>(nda), NPY_ARRAY_OWNDATA);
    return boost::python::numpy::from_data(data_ptr, dt, shape, strides, own);
}
#else
void massage_numpy_array(PyObject *a){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}
boost::python::numpy::ndarray cook_up_a_numpy_array(){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}
#endif

class WrappedNDArray
{
private:
    boost::python::numpy::ndarray _nda;
    void *_raw_data = nullptr;
    bool _owns_data = true;

public:
    boost::python::numpy::ndarray get_nda()
    {
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << std::endl;
        return _nda;
    }
    void set_owns_data(bool owns_data){_owns_data = owns_data;}
    bool owns_data(){return _owns_data;}
    WrappedNDArray(void *data_ptr, boost::python::numpy::dtype dt, boost::python::object shape, boost::python::object strides, boost::python::object own);
    // WrappedNDArray(const WrappedNDArray &other)
    //         :_nda(other._nda)
    // {
    //     std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << std::endl;
    //     _raw_data = other._raw_data;
    //     _owns_data = false;
    // }
    ~WrappedNDArray(){
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << std::endl;
        if(_owns_data){
            free(_raw_data);
        }
        _raw_data = nullptr;
    }

    boost::python::str __str__()
    {
        return boost::python::str(_nda);
    }

};

class ExtNdArray : public boost::python::numpy::ndarray
{
public:
    ExtNdArray(void *data_ptr, boost::python::numpy::dtype dt, boost::python::object shape, boost::python::object strides, boost::python::object own)
    : _shape(shape), boost::python::numpy::ndarray(boost::python::numpy::from_data(data_ptr, dt, shape, strides, own)) {
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << std::endl;
        this->_raw_data = data_ptr;
    }
    ~ExtNdArray(){
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << std::endl;
        free(_raw_data);
    }

    boost::python::object shape()
    {
        return _shape;
    }
private:
    void *_raw_data;
    bool _owns_data;
    boost::python::object _shape;
};

ExtNdArray get_ext_nd_array()
{
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    boost::python::object shape = boost::python::make_tuple(10, 20, 30, 40);
    boost::python::object strides = boost::python::make_tuple(192000, 9600, 320, 8);
    boost::python::object own;
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<int>();

    int * data_ptr = static_cast<int*>(malloc(10 * 20 * 30 * 40 * sizeof(int)));
    for(int i = 1; i <= 10*20*30; ++i){
        data_ptr[i-1] = i;
    }

    // std::shared_ptr<ExtNdArray> enda(new ExtNdArray(data_ptr, dt, shape, strides, own));

    return ExtNdArray(data_ptr, dt, shape, strides, own);

}

boost::python::numpy::ndarray get_ext_nd_array_polymorphic(){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    boost::python::object shape = boost::python::make_tuple(10, 20, 30, 40);
    boost::python::object strides = boost::python::make_tuple(192000, 9600, 320, 8);
    boost::python::object own;
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<int>();

    int * data_ptr = static_cast<int*>(malloc(10 * 20 * 30 * 40 * sizeof(int)));
    for(int i = 1; i <= 10*20*30; ++i){
        data_ptr[i-1] = i;
    }

    return static_cast<boost::python::numpy::ndarray>(ExtNdArray(data_ptr, dt, shape, strides, own));
}

WrappedNDArray::WrappedNDArray(void *data_ptr, boost::python::numpy::dtype dt, boost::python::object shape, boost::python::object strides, boost::python::object own)
        :_raw_data(data_ptr), _nda(boost::python::numpy::from_data(data_ptr, dt, shape, strides, own))
{
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
}

std::shared_ptr<WrappedNDArray> cook_up_wrapped_ndarray()
{
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    boost::python::object shape = boost::python::make_tuple(10, 20, 30, 40);
    boost::python::object strides = boost::python::make_tuple(192000, 9600, 320, 8);
    boost::python::object own;
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<int>();

    int * data_ptr = static_cast<int*>(malloc(10 * 20 * 30 * 40 * sizeof(int)));
    for(int i = 1; i <= 10*20*30; ++i){
        data_ptr[i-1] = i;
    }

    std::shared_ptr<WrappedNDArray> wnda(new WrappedNDArray(data_ptr, dt, shape, strides, own));
    wnda->set_owns_data(true);

    return wnda;
}

WrappedNDArray cook_up_wrapped_ndarray_no_ptr()
{
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    boost::python::object shape = boost::python::make_tuple(10, 20, 30, 40);
    boost::python::object strides = boost::python::make_tuple(192000, 9600, 320, 8);
    boost::python::object own;
    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<int>();

    int * data_ptr = static_cast<int*>(malloc(10 * 20 * 30 * 40 * sizeof(int)));
    for(int i = 1; i <= 10*20*30; ++i){
        data_ptr[i-1] = i;
    }

    // This now causes two constructors and destructors.
    WrappedNDArray nda = WrappedNDArray(data_ptr, dt, shape, strides, own);
    return nda;
}
// THIS IS THE SHIT!  This is what I wanted to know all along!
// http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
// This is cool too : https://gist.github.com/malcolmgreaves/e784281c3b0c0bad9d7224b9c58f7b75
// Also see this: https://groups.google.com/d/msg/cython-users/AG-8G_E0TRQ/D3YrbgxyG3YJ
// And this one is really good, it explains a lot and is exactly what I need https://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/
using namespace boost::python;
BOOST_PYTHON_MODULE(pyspooki_interface)
{
    // Py_Initialize();
#ifdef USE_BOOST_NUMPY
    boost::python::numpy::initialize();
    class_<WrappedNDArray>("WrappedNDArray",init<void *, boost::python::numpy::dtype, boost::python::object, boost::python::object, boost::python::object>() )
            .add_property("inner_nda", &WrappedNDArray::get_nda)
            .def("__str__", &WrappedNDArray::__str__);
    class_<std::shared_ptr<WrappedNDArray>>("WrappedNDARRAY_shared_ptr")
            .add_property("inner_nda", &WrappedNDArray::get_nda)
            .def("__str__", &WrappedNDArray::__str__);
    def("cook_up_wrapped_ndarray", cook_up_wrapped_ndarray);
    def("cook_up_wrapped_ndarray_no_ptr", cook_up_wrapped_ndarray_no_ptr);
    // class_<std::shared_ptr<ExtNdArray>>("ExtNdArray_shared_ptr").def("sh_ptr_use_count", &std::shared_ptr<TestObject>::use_count)
    //     ;
    class_<ExtNdArray>("ExtNdArray", init<void *, boost::python::numpy::dtype, boost::python::object, boost::python::object, boost::python::object>())
            .add_property("shape", &ExtNdArray::shape)
            ;
    def("get_ext_nd_array", get_ext_nd_array);
    def("get_ext_nd_array_polymorphic", get_ext_nd_array_polymorphic);
#endif
    class_<std::shared_ptr<TestObject>>("TestObject_shared_ptr").def("sh_ptr_use_count", &std::shared_ptr<TestObject>::use_count);
    class_<TestObject>("TestObject", init<std::string>()).def("method", &TestObject::method);
    class_<pyspooki_interface_class>("pyspooki_interface_class", init<>())
            .def("method", &pyspooki_interface_class::method);
    def("copy", copy);
    def("pyspooki_interface_function", pyspooki_interface_function);
    def("tanh_impl", tanh_impl);
    def("run_absolute_value_plugin", run_absolute_value_plugin);
    def("returning_shared_ptr_test", returning_shared_ptr_test);
    def("massage_numpy_array", massage_numpy_array);
    def("cook_up_a_numpy_array", cook_up_a_numpy_array);
    def("delete_g_int_ptr", delete_g_int_ptr);
    def("print_g_int_ptr", print_g_int_ptr);
    def("free_g_int_ptr", free_g_int_ptr);
    internal_initializations();
}

// TODO: Create a function that returns a python object that I create by hand in the CPP world
