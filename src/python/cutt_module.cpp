#include "hiptt.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

extern "C" CUTT_API void hiptt_init_python(void* parent, int submodule, const char* apikey);

PYBIND11_MODULE(hiptt, hiptt)
{
    // TODO Read the license file.
    hiptt_init_python(&hiptt, 0 /* no submodule */, nullptr);
}

