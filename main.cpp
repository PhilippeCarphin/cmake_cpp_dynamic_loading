#include <iostream>
#include <dlfcn.h>
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
  }

  void *maker = dlsym(plugin, "maker");

  if(maker){
      std::cerr << "Symbol loaded successfully" << std::endl;
  } else {
      std::cerr << "ERROR loading symbol: " << dlerror() << std::endl;
  }

  return 0;
}
