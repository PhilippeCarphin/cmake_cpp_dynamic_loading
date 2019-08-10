//
// Created by Philippe Carphin on 2019-08-10.
//
#include <boost/python.hpp>
#include <iostream>
#include <cmake_config.out.h>

#include <boost/python/numpy.hpp>

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
    WrappedNDArray(const WrappedNDArray &other)
            :_nda(other._nda)
    {
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << " COPYING object at " << &other <<std::endl;
        _raw_data = other._raw_data;
        _owns_data = true;
    }
    WrappedNDArray(WrappedNDArray &&other)
            :_raw_data(other._raw_data), _nda(other._nda)
    {
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << " MOVING object at " << &other <<std::endl;
        other._raw_data = nullptr;
        other._owns_data = false;
        this->_owns_data = false;
    }
    ~WrappedNDArray(){
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this <<  ", owns_data=" << _owns_data << std::endl;
        if(_owns_data){

            free(_raw_data);
        }
        _raw_data = nullptr;
    }


    void dealloc(){
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << std::endl;
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
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << " giving ownership of data at " << data_ptr << std::endl;
        this->_raw_data = data_ptr;
    }
    ~ExtNdArray(){
        std::cout << "C++      : " << __PRETTY_FUNCTION__ << " C++ object at " << this << " freeing owned data " << _raw_data << std::endl;
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
    return std::move(WrappedNDArray(data_ptr, dt, shape, strides, own));
}

void internal_initializations(){
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    std::cerr << "C++      : [info] This interface was compiled for loading by Python " << BOOST_PYTHON_VERSION << std::endl;
    std::cerr << "C++      : [info] This test is development on SPOOKI" << std::endl;
    std::cerr << "C++      : [info] This was compiled with PLUGIN_PATH=" << PLUGIN_PATH << std::endl;
}
using namespace boost::python;
BOOST_PYTHON_MODULE(THIS_PYTHON_MODULE_NAME)
        {
    boost::python::numpy::initialize();
    class_<WrappedNDArray, std::shared_ptr<WrappedNDArray>>("WrappedNDArray",init<void *, boost::python::numpy::dtype, boost::python::object, boost::python::object, boost::python::object>() )
            .add_property("inner_nda", &WrappedNDArray::get_nda)
            .def("__str__", &WrappedNDArray::__str__)
            .def("dealloc", &WrappedNDArray::dealloc);
    def("cook_up_wrapped_ndarray", cook_up_wrapped_ndarray);
    def("cook_up_wrapped_ndarray_no_ptr", cook_up_wrapped_ndarray_no_ptr);
    // class_<std::shared_ptr<ExtNdArray>>("ExtNdArray_shared_ptr").def("sh_ptr_use_count", &std::shared_ptr<TestObject>::use_count)
    //     ;
    class_<ExtNdArray>("ExtNdArray", init<void *, boost::python::numpy::dtype, boost::python::object, boost::python::object, boost::python::object>())
            .add_property("shape", &ExtNdArray::shape)
            ;
    def("get_ext_nd_array", get_ext_nd_array);
    def("get_ext_nd_array_polymorphic", get_ext_nd_array_polymorphic);
        def("cook_up_a_numpy_array", cook_up_a_numpy_array);
        def("delete_g_int_ptr", delete_g_int_ptr);
        def("print_g_int_ptr", print_g_int_ptr);
        def("free_g_int_ptr", free_g_int_ptr);
        internal_initializations();
        }

// TODO: Create a function that returns a python object that I create by hand in the CPP world
