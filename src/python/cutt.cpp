#include "cutt.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

static std::once_flag cuttInitialized;

class cuTT
{
	cuttHandle plan;
	cuttResult initStatus;
	bool planInitialized = false;

	int rank;
	const std::vector<int>& dim;
	const std::vector<int>& permutation;
	cudaStream_t stream;

public :

	cuTT(int rank_, const std::vector<int>& dim_, const std::vector<int>& permutation_, const py::object& stream_) :
		rank(rank_), dim(dim_), permutation(permutation_)
	{
		// This is only needed for the Umpire allocator's lifetime management:
		// - if CUTT_HAS_UMPIRE is defined, will grab Umpire's allocator;
		// - otherwise this is a no-op
		std::call_once(cuttInitialized, [](){ cuttInitialize(); });

		// Defer cuttPlan to the first use, as we don't know the type yet.

		py::object pycudadriver = py::module::import("pycuda.driver");
                py::object pystream = pycudadriver.attr("stream");

                if (!py::isinstance(stream_, pystream))
                	throw std::invalid_argument(
                        	"Stream argument must be a pycuda.driver.stream");

                stream = (cudaStream_t)stream_.attr("handle_int").cast<intptr_t>();
	}

	cuTT(int rank_, const std::vector<int>& dim_, const std::vector<int>& permutation_, const py::object& stream_,
		const py::object& idata, py::object& odata, const void* alpha = NULL, const void* beta = NULL) :
		rank(rank_), dim(dim_), permutation(permutation_)
	{
		// This is only needed for the Umpire allocator's lifetime management:
                // - if CUTT_HAS_UMPIRE is defined, will grab Umpire's allocator;
                // - otherwise this is a no-op
                std::call_once(cuttInitialized, [](){ cuttInitialize(); });

		py::object pycuda = py::module::import("pycuda");
		py::object pygpuarray = pycuda.attr("gpuarray");
		py::object pycudadriver = py::module::import("pycuda.driver");
		py::object pystream = pycudadriver.attr("stream");

		if (!py::isinstance(idata, pygpuarray))
			throw std::invalid_argument(
                                "Input array must be a pycuda.gpuarray");
		if (!py::isinstance(odata, pygpuarray))
                        throw std::invalid_argument(
                                "Output array must be a pycuda.gpuarray");
		if (!py::isinstance(stream_, pystream))
                        throw std::invalid_argument(
                                "Stream argument must be a pycuda.driver.stream");

		if (!idata.attr("dtype").cast<pybind11::dtype>().is(
			odata.attr("dtype").cast<pybind11::dtype>()))
			throw std::invalid_argument(
                                "Input and output array must have the same type");

		const void* igpuarray = idata.attr("ptr").cast<const void*>();
		void* ogpuarray = odata.attr("ptr").cast<void*>();
		stream = (cudaStream_t)stream_.attr("handle_int").cast<intptr_t>();

                size_t sizeofType = idata.attr("itemsize").cast<size_t>();
		initStatus = cuttPlanMeasure(&plan, rank, reinterpret_cast<const int*>(&dim[0]),
			reinterpret_cast<const int*>(&permutation[0]), sizeofType, stream,
			igpuarray, ogpuarray, alpha, beta);
		planInitialized = true;
	}

	~cuTT()
	{
		cuttDestroy(plan);
	}

	cuttResult execute(const py::object& idata, py::object& odata, const void* alpha = NULL, const void* beta = NULL)
	{
		py::object pycuda = py::module::import("pycuda");
                py::object pygpuarray = pycuda.attr("gpuarray");
                py::object pycudadriver = py::module::import("pycuda.driver");
                py::object pystream = pycudadriver.attr("stream");

                if (!py::isinstance(idata, pygpuarray))
                        throw std::invalid_argument(
                                "Input array must be a pycuda.gpuarray");
                if (!py::isinstance(odata, pygpuarray))
                        throw std::invalid_argument(
                                "Output array must be a pycuda.gpuarray");

                if (!idata.attr("dtype").cast<pybind11::dtype>().is(
                        odata.attr("dtype").cast<pybind11::dtype>()))
                        throw std::invalid_argument(
                                "Input and output array must have the same type");

                const void* igpuarray = idata.attr("ptr").cast<const void*>();
                void* ogpuarray = odata.attr("ptr").cast<void*>();

                size_t sizeofType = idata.attr("itemsize").cast<size_t>();

                if (!planInitialized)
                {
			// Now we know the sizeofType, and can initialize the plan handle.
                        initStatus = cuttPlan(&plan, rank, reinterpret_cast<const int*>(&dim[0]),
                                reinterpret_cast<const int*>(&permutation[0]), sizeofType, stream);
                        planInitialized = true;
                }

                if (initStatus != CUTT_SUCCESS)
                        return initStatus;

		return cuttExecute(plan, igpuarray, ogpuarray, alpha, beta);
	}
};

} // namespace

extern "C" CUTT_API void cutt_init_python(void* parent_, int submodule, const char* apikey)
{
	if (!parent_) return;

	py::module& parent = *reinterpret_cast<py::module*>(parent_);
	py::module cutt = submodule ? parent.def_submodule("cutt") : parent;

	py::class_<cuTT>(cutt, "cuTT")
		.def(py::init<int, const std::vector<int>&, const std::vector<int>&, const py::object&>(),
R"doc(Create plan

Parameters
handle            = Returned handle to cuTT plan
rank              = Rank of the tensor
dim[rank]         = Dimensions of the tensor
permutation[rank] = Transpose permutation
sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
stream            = CUDA stream (0 if no stream is used)

Returns
Success/unsuccess code)doc"
		)
		.def(py::init<int, const std::vector<int>&, const std::vector<int>&, const py::object&,
			const py::object&, py::object&, const void*, const void*>(),
R"doc(Create plan and choose implementation by measuring performance

Parameters
handle            = Returned handle to cuTT plan
rank              = Rank of the tensor
dim[rank]         = Dimensions of the tensor
permutation[rank] = Transpose permutation
sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
stream            = CUDA stream (0 if no stream is used)
idata             = Input data size product(dim)
odata             = Output data size product(dim)

Returns
Success/unsuccess code)doc"
		)
		.def("execute", &cuTT::execute,
R"doc(Execute plan out-of-place; performs a tensor transposition of the form \f[ \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})} \gets \alpha * \mathcal{A}_{i_0,i_1,...,i_{d-1}} + \beta * \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})}, \f]

Parameters
handle            = Returned handle to cuTT plan
idata             = Input data size product(dim)
odata             = Output data size product(dim)
alpha             = scalar for input
beta              = scalar for output
 
Returns
Success/unsuccess code)doc"
		);
}

