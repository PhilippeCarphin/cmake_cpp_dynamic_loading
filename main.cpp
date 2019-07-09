#include <iostream>
#include <dlfcn.h>
#include <meteo_operations/OperationBase.h>
#include "cmake_config.out.h"

int main(void)
{
  std::cerr << __PRETTY_FUNCTION__ << " ..." << std::endl;

  std::string absolute_value_path;

  absolute_value_path += std::string(PLUGIN_PATH) + "/AbsoluteValue/libAbsoluteValue.so";

  void *plugin = dlopen(absolute_value_path.c_str(), RTLD_NOW);

  if(plugin){
      std::cerr << "Plugin loaded successfully" << std::endl;
  } else {
      std::cerr << "ERROR loading plugin : " << dlerror() << std::endl;
      exit(1);
  }

  exit(0);

  void *maker = dlsym(plugin, "maker");

  if(maker){
      std::cerr << "Symbol loaded successfully" << std::endl;
  } else {
      std::cerr << "ERROR loading symbol: " << dlerror() << std::endl;
      exit(2);
  }

  /*
   * We have a pointer to the symbol but only the programmer knows
   * what that symbol is.
   */
  // typedef OperationBase *plugin_maker_t();
  // plugin_maker_t *absolute_value_maker = reinterpret_cast<plugin_maker_t *>(maker);

  // OperationBase *absolute_value_instance_ptr = absolute_value_maker();

  // absolute_value_instance_ptr->algo();

  return 0;
}
