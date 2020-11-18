//
// Created by Philippe Carphin on 2019-08-10.
//

#ifndef TUTORIAL_ABSVAL_H
#define TUTORIAL_ABSVAL_H

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


#endif //TUTORIAL_ABSVAL_H
