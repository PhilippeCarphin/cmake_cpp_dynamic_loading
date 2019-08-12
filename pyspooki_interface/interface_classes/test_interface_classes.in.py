import interface_classes as interface
import sys

def pyprint(message):
    print("PYTHON   : " + message)

def py_obj_ref_count(x):
    return sys.getrefcount(x) - 3

print("==== create_a_shared_ptr =====")
def create_a_shared_ptr():
    print("PYTHON   : Calling returing_shared_ptr_test ...")
    a = interface.get_object_shared_ptr("create_a_shared_ptr")
    print("PYTHON   : back from returning_shared_ptr_test")

pyprint("create_a_shared_ptr()")
create_a_shared_ptr()
pyprint("back from create_a_shared_ptr")

print("========= return_a_shared_ptr() =======")
def return_a_shared_ptr():
    return interface.get_object_shared_ptr("get_object_shared_ptr")


pyprint("b = return_a_shared_ptr()")
b = return_a_shared_ptr()
pyprint("b.sh_ptr_use_count() : {}, py_obj_ref_count(b) : {}"
        .format(b.sh_ptr_use_count(), py_obj_ref_count(b)))

print("========== REFERENCE COUNTING ===========")
pyprint("c = b")
c = b
pyprint("b.sh_ptr_use_count() : {}, py_obj_ref_count(b) : {} (we have two python variables pointing to the same python object which has a std::shared_ptr)"
        .format(b.sh_ptr_use_count(), py_obj_ref_count(b)))

pyprint("d = interface.copy(b)")
d = interface.copy(b)
pyprint("b.sh_ptr_use_count() : {}, py_obj_ref_count(b) : {}  (we now have two std::shared_ptr sharing the TestObject)"
        .format(b.sh_ptr_use_count(), py_obj_ref_count(b)))
pyprint("d.sh_ptr_use_count() : {}, py_obj_ref_count(d) : {}  (and two different python objects and d's object is only referenced by d)"
        .format(d.sh_ptr_use_count(), py_obj_ref_count(d)))

pyprint("c = 4")
c = 4
pyprint("b.sh_ptr_use_count() : {}, py_obj_ref_count(b) : {}".format(b.sh_ptr_use_count(), py_obj_ref_count(b)))

pyprint("d = 4")
d = 4
pyprint("b.sh_ptr_use_count() : {}, py_obj_ref_count(b) : {}".format(b.sh_ptr_use_count(), py_obj_ref_count(b)))

pyprint("b = interface.copy(b)")
b = interface.copy(b)
pyprint("b.sh_ptr_use_count() : {}, py_obj_ref_count(b) : {}".format(b.sh_ptr_use_count(), py_obj_ref_count(b)))


pyprint("b = 4")
b = 4
pyprint("no more TestObject")

print("getting test object manual")
c = interface.InterfaceClass("manual")











print("PYTHON   : SCRIPT END")
