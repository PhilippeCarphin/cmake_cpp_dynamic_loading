//
// Created by afsmpca on 19-07-09.
//

#ifndef TUTORIAL_ABSOLUTEVALUE_H
#define TUTORIAL_ABSOLUTEVALUE_H

#include "meteo_operations/OperationBase.h"

class AbsoluteValue : public OperationBase{

    virtual void algo();

};

OperationBase *maker();

#endif //TUTORIAL_ABSOLUTEVALUE_H
