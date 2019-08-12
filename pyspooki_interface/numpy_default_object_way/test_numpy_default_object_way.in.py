import numpy_default_object_way as interface
import os
import psutil

def get_process_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

one_mb = 1000000

MEMORY_MAX = 100 * one_mb
i = 0
while True:
    j = 4
    print("PYTHON   : ---------------- While iteration ------------------- ({})".format(i))
    print("PYTHON   : BEFORE calling test_capsule_way()")
    arr = interface.get_array_that_owns_through_default_object()
    print("PYTHON   : AFTER calling test_capsule_way()")
    i += 1
    if i % 1000 == 0:
        mem = get_process_memory_usage()
        if mem > MEMORY_MAX:
            print("PYTHON   : Bro chill with the memory, you're using {}MB over here!".format(mem/one_mb))
            quit()
        print("PYTHON   : Nb arrays created and destroyed : {}".format(i))

    print("PYTHON   : ----------- End while iteration\n")
    # time.sleep(0.1)

print("PYTHON   : SCRIPT END")
