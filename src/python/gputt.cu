#include "gputt.h"

#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

const char *gputtErrorString(gputtResult result) {
  const char *gputtSuccess = "Success";
  const char *gputtInvalidPlan = "Invalid plan handle";
  const char *gputtInvalidParameter = "Invalid input parameter";
  const char *gputtInvalidDevice =
      "Execution tried on device different than where plan was created";
  const char *gputtInternalError = "Internal error";
  const char *gputtUndefinedError = "Undefined error";
  const char *gputtUnknownError = "Unknown error";

  switch (result) {
  case GPUTT_SUCCESS:
    return gputtSuccess;
  case GPUTT_INVALID_PLAN:
    return gputtInvalidPlan;
  case GPUTT_INVALID_PARAMETER:
    return gputtInvalidParameter;
  case GPUTT_INVALID_DEVICE:
    return gputtInvalidDevice;
  case GPUTT_INTERNAL_ERROR:
    return gputtInternalError;
  case GPUTT_UNDEFINED_ERROR:
    return gputtUndefinedError;
  }

  return gputtUnknownError;
}

class gpuTT {
  gputtHandle plan;
  gputtResult initStatus;
  bool planInitialized = false;

  int rank;
  const std::vector<int> dim;
  const std::vector<int> permutation;
  gpuStream stream;

public:
  gpuTT(int rank_, const std::vector<int> &dim_,
        const std::vector<int> &permutation_, const py::object &stream_)
      : rank(rank_), dim(dim_), permutation(permutation_) {
    // Defer gputtPlan to the first use, as we don't know the type yet.

    py::object pygpu = py::module::import("pygpu");
    py::object pystream = pygpu.attr("driver").attr("Stream");

    if (stream_.is_none())
      stream = 0;
    else {
      if (!py::isinstance(stream_, pystream)) {
        std::stringstream ss;
        ss << "Stream argument must be a pygpu.driver.stream, got: ";
        ss << py::str(stream_);
        throw std::invalid_argument(ss.str());
      }

      stream = (gpuStream)stream_.attr("handle_int").cast<intptr_t>();
    }
  }

  gpuTT(int rank_, const std::vector<int> &dim_,
        const std::vector<int> &permutation_, const py::object &stream_,
        const py::object &idata, py::object &odata, const void *alpha = NULL,
        const void *beta = NULL)
      : rank(rank_), dim(dim_), permutation(permutation_) {
    py::object pygpu = py::module::import("pygpu");
    py::object pygpuarray = pygpu.attr("gpuarray");
    py::object pystream = pygpu.attr("driver").attr("Stream");

    if (!py::isinstance(idata, pygpuarray)) {
      std::stringstream ss;
      ss << "Input array must be a pygpu.gpuarray, got: ";
      ss << py::str(idata);
      throw std::invalid_argument(ss.str());
    }
    if (!py::isinstance(odata, pygpuarray)) {
      std::stringstream ss;
      ss << "Output array must be a pygpu.gpuarray, got: ";
      ss << py::str(odata);
      throw std::invalid_argument(ss.str());
    }

    if (stream_.is_none())
      stream = 0;
    else {
      if (!py::isinstance(stream_, pystream)) {
        std::stringstream ss;
        ss << "Stream argument must be a pygpu.driver.stream, got: ";
        ss << py::str(stream_);
        throw std::invalid_argument(ss.str());
      }

      stream = (gpuStream)stream_.attr("handle_int").cast<intptr_t>();
    }

    if (!idata.attr("dtype").cast<pybind11::dtype>().is(
            odata.attr("dtype").cast<pybind11::dtype>()))
      throw std::invalid_argument(
          "Input and output array must have the same type");

    const void *igpuarray =
        reinterpret_cast<const void *>(idata.attr("ptr").cast<intptr_t>());
    void *ogpuarray =
        reinterpret_cast<void *>(odata.attr("ptr").cast<intptr_t>());

    size_t sizeofType = idata.attr("itemsize").cast<size_t>();
    initStatus =
        gputtPlanMeasure(&plan, rank, reinterpret_cast<const int *>(&dim[0]),
                         reinterpret_cast<const int *>(&permutation[0]),
                         sizeofType, stream, igpuarray, ogpuarray, alpha, beta);
    planInitialized = true;
  }

  ~gpuTT() { gputtDestroy(plan); }

  void execute(const py::object &idata, py::object &odata,
               const py::object &pyalpha, const py::object &pybeta) {
    py::object pygpu = py::module::import("pygpu");
    py::object pygpuarray = pygpu.attr("gpuarray").attr("GPUArray");
    py::object pystream = pygpu.attr("driver").attr("Stream");

    if (!py::isinstance(idata, pygpuarray)) {
      std::stringstream ss;
      ss << "Input array must be a pygpu.gpuarray, got: ";
      ss << py::str(idata);
      throw std::invalid_argument(ss.str());
    }
    if (!py::isinstance(odata, pygpuarray)) {
      std::stringstream ss;
      ss << "Output array must be a pygpu.gpuarray, got: ";
      ss << py::str(odata);
      throw std::invalid_argument(ss.str());
    }

    if (!idata.attr("dtype").cast<pybind11::dtype>().is(
            odata.attr("dtype").cast<pybind11::dtype>())) {
      std::stringstream ss;
      ss << "Input and output array must have the same type, got: ";
      ss << py::str(idata.attr("dtype").cast<pybind11::dtype>());
      ss << " and ";
      ss << py::str(odata.attr("dtype").cast<pybind11::dtype>());
      throw std::invalid_argument(ss.str());
    }

    const void *igpuarray =
        reinterpret_cast<const void *>(idata.attr("ptr").cast<intptr_t>());
    void *ogpuarray =
        reinterpret_cast<void *>(odata.attr("ptr").cast<intptr_t>());

    size_t sizeofType = idata.attr("itemsize").cast<size_t>();

    if (!planInitialized) {
      // Now we know the sizeofType, and can initialize the plan handle.
      initStatus = gputtPlan(
          &plan, rank, reinterpret_cast<const int *>(&dim[0]),
          reinterpret_cast<const int *>(&permutation[0]), sizeofType, stream);
      planInitialized = true;
    }

    if (initStatus != GPUTT_SUCCESS) {
      std::stringstream ss;
      ss << "gpuTT error: ";
      ss << gputtErrorString(initStatus);
      throw std::invalid_argument(ss.str());
    }

    double *alpha = NULL, valpha;
    if (!pyalpha.is_none()) {
      valpha = pyalpha.cast<double>();
      alpha = &valpha;
    }
    double *beta = NULL, vbeta;
    if (!pybeta.is_none()) {
      vbeta = pybeta.cast<double>();
      beta = &vbeta;
    }
    gputtResult status = gputtExecute(plan, igpuarray, ogpuarray, alpha, beta);
    if (status != GPUTT_SUCCESS) {
      std::stringstream ss;
      ss << "gpuTT error: ";
      ss << gputtErrorString(status);
      throw std::invalid_argument(ss.str());
    }
  }
};

} // namespace

extern "C" GPUTT_API void gputt_init_python(void *parent_, int submodule,
                                            const char *apikey) {
  if (!parent_)
    return;

  py::module &parent = *reinterpret_cast<py::module *>(parent_);
  py::module gputt = submodule ? parent.def_submodule("gputt") : parent;

  py::class_<gpuTT>(gputt, "gpuTT")
      .def(py::init<int, const std::vector<int> &, const std::vector<int> &,
                    const py::object &>(),
           R"doc(Create plan

Parameters
handle            = Returned handle to gpuTT plan
rank              = Rank of the tensor
dim[rank]         = Dimensions of the tensor
permutation[rank] = Transpose permutation
sizeofType        = Size of the elements of the tensor in bytes (=2, 4 or 8)
stream            = CUDA stream (0 if no stream is used)

Returns
Success/unsuccess code)doc")
      .def(py::init<int, const std::vector<int> &, const std::vector<int> &,
                    const py::object &, const py::object &, py::object &,
                    const void *, const void *>(),
           R"doc(Create plan and choose implementation by measuring performance

Parameters
handle            = Returned handle to gpuTT plan
rank              = Rank of the tensor
dim[rank]         = Dimensions of the tensor
permutation[rank] = Transpose permutation
sizeofType        = Size of the elements of the tensor in bytes (=2, 4 or 8)
stream            = CUDA stream (0 if no stream is used)
idata             = Input data size product(dim)
odata             = Output data size product(dim)

Returns
Success/unsuccess code)doc")
      .def(
          "execute", &gpuTT::execute,
          R"doc(Execute plan out-of-place; performs a tensor transposition of the form \f[ \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})} \gets \alpha * \mathcal{A}_{i_0,i_1,...,i_{d-1}} + \beta * \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})}, \f]

Parameters
handle            = Returned handle to gpuTT plan
idata             = Input data size product(dim)
odata             = Output data size product(dim)
alpha             = scalar for input
beta              = scalar for output
 
Returns
Success/unsuccess code)doc",
          py::arg().noconvert(), py::arg().noconvert(),
          py::arg("pyalpha") = py::none(), py::arg("pybeta") = py::none());
}
