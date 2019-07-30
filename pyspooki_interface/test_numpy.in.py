import pyspooki_interface as interface
import numpy as np

def send_numpy_array():
    my_array = np.zeros((1, 2, 3), dtype='uint64')
    print(my_array)
    my_array[0,1] = 5;
    interface.massage_numpy_array(my_array)
    print("PYTHON   : my_array.shape = {}".format(my_array.shape))
    return my_array

arr = send_numpy_array()
print(arr)