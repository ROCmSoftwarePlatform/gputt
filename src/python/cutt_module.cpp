#include "cutt.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

extern "C" CUTT_API void cutt_init_python(void* parent, int submodule, const char* apikey);

PYBIND11_MODULE(cutt, cutt)
{
    // TODO Read the license file.
    cutt_init_python(&cutt, 0 /* no submodule */, nullptr);
}

