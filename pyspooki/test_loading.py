
import sys

sys.path.append("../cmake-build-debug/pyspooki")

import libpyspooki_interface as interface

my_class = interface.pyspooki_interface_class()
my_class.method()

interface.pyspooki_interface_function()
