//
// Created by Philippe Carphin on 2019-08-10.
//

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

std::string bnp_array_to_string(boost::python::numpy::ndarray const &a){
    return std::string(boost::python::extract<char const*>(boost::python::str(a)));
}

std::string bnp_array_to_shape_string(boost::python::numpy::ndarray const &a)
{
    int nd = a.get_nd();
    auto *dims = a.get_shape();

    std::ostringstream oss;
    oss << "(";
    for(int i = 0;i < nd-1; i++){
        oss << dims[i] << ", ";
    }
    oss << dims[nd-1] << ")" << std::endl;

    return oss.str();
}

void massage_numpy_array(boost::python::numpy::ndarray const &a){
    std::cout << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    std::cout << "C++      : " << " printing str(a) ..." << std::endl << bnp_array_to_string(a) << std::endl;
    std::cout << "C++      : " << " shape is : (" << a.shape(0) << ", " << a.shape(1) << ", " << a.shape(2) << ")" << std::endl;
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
    auto *data = reinterpret_cast<int *>(a.get_data());
    int z = 0;

    for(int i = 0; i < a.shape(0); i++)
        for(int j = 0; j < a.shape(1); j++)
            for(int k = 0; k < a.shape(2); k++){
                z++;
                char * const base = (char * const)data;
                char * const element = base + i * a.strides(0) + j * a.strides(1) + k * a.strides(2);
                int * const int_p = (int * const)element;
                *int_p = z;
            }
    std::cout << "C++      : " << " printing str(a) ..." << std::endl << bnp_array_to_string(a) << std::endl;


    boost::python::numpy::dtype dt = boost::python::numpy::dtype::get_builtin<int>();
    std::cout << "C++      : " << " dtype::get_builtin<int>().get_itemsize() = " << dt.get_itemsize() << std::endl;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(THIS_PYTHON_MODULE_NAME)
{
    boost::python::numpy::initialize();
   def("massage_numpy_array", massage_numpy_array);
}