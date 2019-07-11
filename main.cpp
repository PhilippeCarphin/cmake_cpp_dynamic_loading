#include <cstdlib>
#include <iostream>
#include <dlfcn.h>
#include <meteo_operations/OperationBase.h>
#include "cmake_config.out.h"

int main(void)
{
  std::cerr << "C++      : " << __PRETTY_FUNCTION__ << std::endl;

  std::string absolute_value_path;

  absolute_value_path += std::string(PLUGIN_PATH) + "/AbsoluteValue/libAbsoluteValue.so";

  void *plugin = dlopen(absolute_value_path.c_str(), RTLD_NOW);

  if(plugin){
      std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : Plugin loaded successfully" << std::endl;
  } else {
      std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : ERROR loading plugin : " << dlerror() << std::endl;
  }

  void *maker = dlsym(plugin, "maker");

  if(maker){
      std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : Symbol loaded successfully" << std::endl;
  } else {
      std::cerr << "C++      : " << __PRETTY_FUNCTION__ << " : ERROR loading symbol: " << dlerror() << std::endl;
  }

  /*
   * We have a pointer to the symbol but only the programmer knows
   * what that symbol is.
   */
  typedef OperationBase *plugin_maker_t();
  plugin_maker_t *absolute_value_maker = reinterpret_cast<plugin_maker_t *>(maker);

  OperationBase *absolute_value_instance_ptr = absolute_value_maker();

  absolute_value_instance_ptr->algo();

  return 0;
}
