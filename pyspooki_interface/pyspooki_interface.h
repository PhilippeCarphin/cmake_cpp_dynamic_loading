//
// Created by afsmpca on 19-07-09.
//

#ifndef TUTORIAL_PYSPOOKI_INTERFACE_H
#define TUTORIAL_PYSPOOKI_INTERFACE_H

#ifdef __GNUC__
/*
 * This stops a warning that results from the reinterpret_cast below
 *
 * ... warning: ISO C++ forbids casting between pointer-to-function and pointer-to-object [-Wpedantic]
 *      return reinterpret_cast<plugin_maker_t*>( dlsym(__handle, __name) );
 *
 * clang doesn't produce this warning even though I give it the -Wpedantic flag
 */
#pragma GCC system_header
template <typename RetType>
RetType dlsymAs(void *symbol_ptr)
{
   return reinterpret_cast<RetType>(symbol_ptr);
}
#endif


class pyspooki_interface_class
{
public:
    pyspooki_interface_class();
    void method();
};

void pyspooki_interface_function();


#endif //TUTORIAL_PYSPOOKI_INTERFACE_H
