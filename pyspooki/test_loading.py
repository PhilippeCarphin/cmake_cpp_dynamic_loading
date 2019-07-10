import libpyspooki_interface as interface

print("PYTHON   : Creating an instance of a class")
my_class = interface.pyspooki_interface_class()
print("PYTHON   : Calling a method of the class")
my_class.method()

print("PYTHON   : Calling a regular exported function")
interface.pyspooki_interface_function()
