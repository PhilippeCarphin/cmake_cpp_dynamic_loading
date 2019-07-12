//
// Created by afsmpca on 19-07-09.
//

#ifndef TUTORIAL_ABSOLUTEVALUE_H
#define TUTORIAL_ABSOLUTEVALUE_H

#include "OperationBase.h"

class AbsoluteValue : public OperationBase{

    virtual void algo();

};

extern "C" OperationBase *maker();

#endif //TUTORIAL_ABSOLUTEVALUE_H
