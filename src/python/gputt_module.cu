#include "gputt.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

extern "C" GPUTT_API void gputt_init_python(void* parent, int submodule, const char* apikey);

PYBIND11_MODULE(gputt, gputt)
{
    // TODO Read the license file.
    gputt_init_python(&gputt, 0 /* no submodule */, nullptr);
}

