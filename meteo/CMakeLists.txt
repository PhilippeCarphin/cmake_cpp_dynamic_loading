
add_library(meteo SHARED meteo.c)

# This will cause ' -I ${CMAKE_CURRENT_SOURCE_DIR} to go on the command line
# for things that link with this target.
# that is why we can do
#    #include "OperationBase.h"
# in plugins/AbsoluteValue/AbsoluteValue.h, however I prefer to have a toplevel
# include_directories(Project root) and have includes like
#    #include "meteo_operations/OperationBase.h"
# because it's easier to spot where what you're incuding comes from.
target_include_directories(
        meteo_operations
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)