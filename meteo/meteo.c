//
// Created by afsmpca on 19-07-09.
//
#include <iostream>

#include "OperationBase.h"

void OperationBase::algo()
{
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;
    base_method();
}

void OperationBase::base_method()
{
    std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : ... world!" << std::endl;
}
