//
// Created by Philippe Carphin on 2019-08-10.
//

#ifndef TUTORIAL_INTERFACE_CLASSES_H
#define TUTORIAL_INTERFACE_CLASSES_H

#include <string>

class InterfaceClass {
public:
    InterfaceClass(std::string name);
    void method();
    ~InterfaceClass();
    std::string name;
};


#endif //TUTORIAL_INTERFACE_CLASSES_H
