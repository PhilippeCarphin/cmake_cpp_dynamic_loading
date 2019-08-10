import numpy_legit_attempts as nla


i = 0;
try:
    while True:
        i += 1
        nla.get_numpy_array_owning_data()
        if i % 1000 == 0:
            print("PYTHON   : Nb arrays created = {}".format(i))
except KeyboardInterrupt as e:
    print("\nPYTHON   : Nb arrays created = {}\n".format(i))
    raise e
