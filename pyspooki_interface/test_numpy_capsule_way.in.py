#!/usr/bin/env python3
import numpy_capsule_way as interface

i = 0
while i < 10:
    print("PYTHON   : ---------------- While iteration ------------------- ({})".format(i))
    print("PYTHON   : BEFORE calling test_capsule_way()")
    arr = interface.get_array_that_owns_through_capsule()
    print("PYTHON   : AFTER calling test_capsule_way()")
    i += 1
    if i % 1000 == 0:
        print("PYTHON   : Nb arrays created and destroyed : {}".format(i))

    print("PYTHON   : ----------- End while iteration")
print("PYTHON   : SCRIPT END")