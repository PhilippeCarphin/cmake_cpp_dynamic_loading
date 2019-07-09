//
// Created by afsmpca on 19-07-09.
//

#include <iostream>

#include "AbsoluteValue.h"

void AbsoluteValue::algo(){
    std::cerr << __PRETTY_FUNCTION__ << " : Hello ..." << std::endl;
    base_method();
}

extern "C" {
    OperationBase *maker()
    {
        return new AbsoluteValue();
    }
}
