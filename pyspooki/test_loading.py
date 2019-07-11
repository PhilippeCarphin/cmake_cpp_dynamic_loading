print()
print("PYTHON   : Importing libpyspooki_interface")
import sys
sys.path.append("/Users/pcarphin/Documents/GitHub/cmake_cpp_dynamic_loading/cmake-build-debug/pyspooki/")
import pyspooki_interface as interface

print()
print("PYTHON   : Creating an instance of a class")
my_class = interface.pyspooki_interface_class()

print()
print("PYTHON   : Calling a method of the class")
my_class.method()

print()
print("PYTHON   : Calling a regular exported function")
interface.pyspooki_interface_function()

print()
print("PYTHON   : Calling interface.tanh_impl(2.0)")
result = interface.tanh_impl(2.0)
print("PYTHON   : Result of tanh_impl: {}".format(result))
