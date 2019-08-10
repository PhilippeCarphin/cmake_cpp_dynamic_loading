import numpy as np
import receive_numpy_array as interface

def pyprint(message):
    print("PYTHON   : " + str(message))
def send_numpy_array():
    my_array = np.zeros((1, 2, 3), dtype='uint64')
    my_array[0,1] = 2**16 - 1;
    interface.massage_numpy_array(my_array)
    pyprint("PYTHON   : my_array.shape = {}".format(my_array.shape))
    return my_array

def test_send_numpy_array():
    pyprint("Calling send_numpy_array()")
    arr = send_numpy_array()
    pyprint("Back from calling send_numpy_array()\n")

test_send_numpy_array()
