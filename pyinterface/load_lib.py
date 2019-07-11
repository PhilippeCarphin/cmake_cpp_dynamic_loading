import sys

sys.path.append("/Users/pcarphin/Documents/GitHub/cmake_cpp_dynamic_loading/cmake-build-debug/pyinterface")

import pyinter

print(dir(pyinter))

my_class_instance = pyinter.PyInterface()
my_class_instance.method()
pyinter.regular_function()