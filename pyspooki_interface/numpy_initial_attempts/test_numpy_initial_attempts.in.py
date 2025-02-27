#!/usr/bin/env python3

import numpy_initial_attempts as interface
import numpy as np
import tracemalloc

def pyprint(msg):
    print("PYTHON   : " + str(msg))


def test_memory():
    # pyprint("Calling cook_up_a_numpy_array()")
    arr = interface.cook_up_a_numpy_array()
    # pyprint("Back from calling cook_up_a_numpy_array()")
    # pyprint("arr : ... \n{}".format(arr))
    # interface.print_g_int_ptr(3)
    # pyprint("typeof arr = {}".format(type(arr)))
    return arr

def test_numpy_straight_up():
    arr = np.zeros((10, 20, 30, 40))
    return arr

def trace_malloc_test_memory():
    tracemalloc.start()
    pyprint("Before calling test_memory()")
    arr = test_memory()
    pyprint("After calling test_memory()")
    snapshot = tracemalloc.take_snapshot()
    for snap in snapshot.statistics('lineno'):
        pyprint(str(snap))

def test_wrapped_array():
    pyprint("FIRST LINE of test_wrapped_array()")
    wrapped = interface.cook_up_wrapped_ndarray()
    pyprint("LAST LINE of test_wrapped_array()")
    # inner_nda = wrapped.inner_nda
    # print(inner_nda)
    # return wrapped

print()
pyprint("BEFORE calling test_wrapped_array()")
test_wrapped_array()
pyprint("AFTER calling test_wrapped_array()")

def test_wrapped_ndarray_no_ptr():
    pyprint("FIRST LINE of test_wrapped_array_no_ptr()")
    wrapped_no_ptr = interface.cook_up_wrapped_ndarray_no_ptr()
    pyprint("LAST LINE of test_wrapped_array_no_ptr()")

print()
pyprint("BEFORE calling test_wrapped_array_no_ptr()")
test_wrapped_ndarray_no_ptr()
pyprint("AFTER calling test_wrapped_array_no_ptr()")


print()
my_array = interface.get_ext_nd_array()
pyprint("my_array.shape = {}".format(my_array.shape))
# pyprint("my_array.strides = {}".format(my_array.strides))
arr = interface.cook_up_a_numpy_array()
pyprint("arr.shape = {}".format(arr.shape))
pyprint("arr.strides = {}".format(arr.strides))

def test_inheritance():
    my_array = interface.get_ext_nd_array_polymorphic()
    pyprint(my_array.strides)
    pyarray = np.zeros(my_array.shape)
    # print(pyarray[:,:,:])

    # print("before doing a slice")
    # print(my_array[:, :, :])

print()
test_inheritance()

