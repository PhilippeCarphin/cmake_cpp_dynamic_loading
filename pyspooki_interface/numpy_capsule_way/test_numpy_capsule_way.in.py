#!/usr/bin/env python3
import numpy_capsule_way as interface
import os
import psutil
import time

def get_process_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

one_mb = 1000000

MEMORY_MAX = 100 * one_mb
i = 0
max_nb_arrays = 10000
while True:
    j = 4
    print("PYTHON   : ---------------- While iteration ------------------- ({})".format(i))
    print("PYTHON   : BEFORE calling test_capsule_way()")
    arr = interface.get_array_that_owns_through_capsule()
    print("PYTHON   : AFTER calling test_capsule_way()")
    i += 1
    if i % 1000 == 0:
        mem = get_process_memory_usage()
        if mem > MEMORY_MAX:
            print("PYTHON   : Bro chill with the memory, you're using {}MB over here!".format(mem/1000000))
            quit()
        print("PYTHON   : Nb arrays created and destroyed : {}".format(i))

    if i > max_nb_arrays and get_process_memory_usage() < 100 * one_mb:
        print("With the arrays we're creating, if we created {} of them and we're still not using 100MB, then I think it's safe to say that the memory gets freed".format(max_nb_arrays))
        quit(0)

    print("PYTHON   : ----------- End while iteration\n")
    # time.sleep(0.1)

print("PYTHON   : SCRIPT END")